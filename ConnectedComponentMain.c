#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#define MAX_SOURCE_SIZE (0x100000)
#define _CRT_SECURE_NO_WARNINGS
#define MAXPASS 10
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <CL/opencl.h>
#include "stdafx.h"
#include <time.h>

void abortf(const char *mes, ...) 
{
	va_list ap;
	va_start(ap, mes);
	vfprintf(stderr, mes, ap);
	va_end(ap);
	exit(-1);
}





void main(int argc, char **argv)
{
	int x, y, i;
	int npass;
	int rgb;
	float time_spent;
	cl_int error;
	cl_platform_id cp_Platform[2];
	cl_device_id Device_ID;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel Pre_kernel;
	cl_kernel Pro_kernel;
	cl_mem memPix;
	cl_mem memLabel;
	cl_mem memFlags;
	clock_t begin = clock();
	size_t source_size;
	char *kernelSource;
	FILE *printer;
	printer = fopen("printerFile.txt", "a+");

	

	IplImage*img = 0; // Intel Image Processing Library 
	img = cvLoadImage("test.png", CV_LOAD_IMAGE_COLOR); //cvLoadImage provides a pointer to an IplImage, which means it creates an IplImage when it loads it and returns it's emplacement.
	if (!img) abortf("Could not load test.png \n");
	if (img->nChannels != 3) abortf("nChannels != 3\n");


	//My image data data:
	//nsize=144 -> sizeof(intel image processing library)
	//nChannels=3 -> RGB
	//depth=8 -> 8bits per channel (255,255,255)
	//width=160
	//height=160
	//imageSize=76800 -> width * height * 3BytesOfData(1byte,1byte,1byte)
	//widthStep=480 -> width * 3bytesOfdata


	int iw = img->width, ih = img->height; // the size of our matrix depends on image size and so the work_items
	uint8_t *data = (uint8_t *)img->imageData;
	

	cl_int *bufPix = (cl_int *)calloc(iw * ih, sizeof(cl_int));
	cl_int *bufLabel = (cl_int *)calloc(iw * ih, sizeof(cl_int));//output buffer
	cl_int *bufFlags = (cl_int *)calloc(MAXPASS + 1, sizeof(cl_int));//output buffer
	
	for (y = 0; y<ih; y++)
	{
		for (x = 0; x<iw; x++)
		{
			
			fprintf(printer,"%d ", data[y*img->widthStep + x * 3 + 1]);
			if (data[y*img->widthStep + x * 3 + 1] > 127)// (matrix height)*(row size in bytes) + x*(3 bytes of data(RGB)) + 1
				bufPix[y*iw + x] = 1;

			else
				bufPix[y*iw + x] = 0;
		}
	}

    //Kernel File
	FILE* fileKernel;	
	fileKernel = fopen("KernelCode.cl", "r");
	if (!fileKernel)
	{
		printf("Cannot open kernel file!\n");
		exit(1);
	}

	// Read kernel code
	kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(kernelSource, 1, MAX_SOURCE_SIZE, fileKernel);
	fclose(fileKernel);


	error = clGetPlatformIDs(2, cp_Platform, NULL);	//array with two devices

	error = clGetDeviceIDs(cp_Platform[1], CL_DEVICE_TYPE_GPU, 1, &Device_ID, NULL); // cp_platform[1] = Nvidia GPU

	context = clCreateContext(NULL, 1, &Device_ID, NULL, NULL, &error); // creating openCL context ----> error 2

	queue = clCreateCommandQueue(context, Device_ID, 0, &error); // creating command queue, executing openCL context on device cp_Platform[1] ****

	program = clCreateProgramWithSource(context, 1, (const char **)& kernelSource, (const size_t *)&source_size, &error); //this function creates a program object for this specific openCL context

	error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); //compiles and links a program executable from the program source

	Pre_kernel = clCreateKernel(program, "PreparationKernal", &error); //creating kernel object 
	Pro_kernel = clCreateKernel(program, "PropagateKernal", &error); //creating kernel object 

	memPix = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), bufPix, NULL);
	memLabel = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), bufLabel, NULL);
	memFlags = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (MAXPASS + 1) * sizeof(cl_int), bufFlags, NULL);

	
	size_t global_size [2] = { (size_t)((iw + 31) & ~31), (size_t)((ih + 31) & ~31) };// 160,160 in my example. basically it depends on input picture size.
	printf("*NOTE: THE GLOBAL SIZE DEPENDS ON INPUT SIZE\nGlobal size: %d,%d", global_size[0], global_size[1]);

	cl_event events[MAXPASS + 1];

	for (i = 0; i <= MAXPASS; i++) 
	{
		events[i] = clCreateUserEvent(context, NULL);
	}


	
	clSetKernelArg(Pre_kernel, 0, sizeof(cl_mem), (void *)&memLabel);
	clSetKernelArg(Pre_kernel, 1, sizeof(cl_mem), (void *)&memPix);
	clSetKernelArg(Pre_kernel, 2, sizeof(cl_mem), (void *)&memFlags);
	i = MAXPASS;
	clSetKernelArg(Pre_kernel, 3, sizeof(cl_int), (void *)&i);
	i = 0;
	clSetKernelArg(Pre_kernel, 4, sizeof(cl_int), (void *)&i);
	clSetKernelArg(Pre_kernel, 5, sizeof(cl_int), (int *)&iw);
	clSetKernelArg(Pre_kernel, 6, sizeof(cl_int), (int *)&ih);

	size_t local_work_size[2] = { 32, 32 };
	printf("\nLocal Work size: %d,%d", local_work_size[0], local_work_size[1]);
	error |= clEnqueueNDRangeKernel(queue, Pre_kernel, 2, NULL, global_size, local_work_size, 0, NULL, &events[0]);

	for (i = 1; i <= MAXPASS; i++) {
		clSetKernelArg(Pro_kernel, 0, sizeof(cl_mem), (void *)&memLabel);
		clSetKernelArg(Pro_kernel, 1, sizeof(cl_mem), (void *)&memPix);
		clSetKernelArg(Pro_kernel, 2, sizeof(cl_mem), (void *)&memFlags);
		clSetKernelArg(Pro_kernel, 3, sizeof(cl_int), (void *)&i);
		clSetKernelArg(Pro_kernel, 4, sizeof(cl_int), (int *)&iw);
		clSetKernelArg(Pro_kernel, 5, sizeof(cl_int), (int *)&ih);

		clEnqueueNDRangeKernel(queue, Pro_kernel, 2, NULL, global_size, NULL, 0, NULL, &events[i]); // workGroup=null:OpenCL implementation will determine how to be break the global work-items into appropriate work-group instances
	}

	clEnqueueReadBuffer(queue, memLabel, CL_TRUE, 0, iw * ih * sizeof(cl_int), bufLabel, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, memFlags, CL_TRUE, 0, (MAXPASS + 1) * sizeof(cl_int), bufFlags, 0, NULL, NULL);

	clFinish(queue);

	
	for (npass = 0; npass<MAXPASS + 1; npass++)
	{
		if (bufFlags[npass] == 0) break;
	}


	clock_t end = clock();
	time_spent = (float)(end - begin) / CLOCKS_PER_SEC;



	printf("\nTotal program's running time is: %.2f\n", time_spent);

	clReleaseMemObject(memFlags);
	clReleaseMemObject(memLabel);
	clReleaseMemObject(memPix);
	clReleaseKernel(Pro_kernel);
	clReleaseKernel(Pre_kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	
	for (y = 0; y<ih; y++) 
	{
		for (x = 0; x < iw; x++)
		{
			if (bufLabel[y * iw + x] == -1)
				rgb = 0;
			else
				rgb = (bufLabel[y * iw + x] * 1103515245 + 12345);

			data[y * img->widthStep + x * 3 + 0] = rgb & 0xff; rgb >>= 8;
			data[y * img->widthStep + x * 3 + 1] = rgb & 0xff; rgb >>= 8;
			data[y * img->widthStep + x * 3 + 2] = rgb & 0xff; rgb >>= 8;
		}
	}
  
 

  cvSaveImage("output.png", img, NULL);

  free(bufFlags);
  free(bufLabel);
  free(bufPix);

  exit(0);
}
