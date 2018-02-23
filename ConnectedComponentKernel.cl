#pragma OPENCL EXTENSION cl_khr_fp64 : enable      
__kernel void PreparationKernal(__global int *Mlabel, __global int *Mpix, __global int *Mflags, int maxpass, int bgc, int iw, int ih)
{

	const int x = get_global_id(0); //| 0,0 | 0,1 | ...
	const int y = get_global_id(1); //| 1,0 | 1,1 | ...
									//...	...		...
									//...	...		...

	const int p0 = y * iw + x;      //Pixel location

	if (y == 0 && x < maxpass + 1) // single row each time
	{
		if (x == 0)
			Mflags[x] = 1;
		else if (x != 0)
			Mflags[x] = 0;
	}

	if (x >= iw || y >= ih) // If it's not inside image boundaries
		return;

	if (Mpix[p0] == bgc) // if the pixel is turned off = 0, we will give him the value -1
	{
		Mlabel[p0] = -1;
		return;
	}

	//if the pixels are connected we shall give them the same label
	if (y > 0 && Mpix[p0] == Mpix[p0 - iw])// Pixel connection check from above
	{
		Mlabel[p0] = p0 - iw;
		return;
	}

	if (x > 0 && Mpix[p0] == Mpix[p0 - 1])// Pixel connection check from behind 
	{
		Mlabel[p0] = p0 - 1;
		return;
	}

	Mlabel[p0] = p0;
	
}






__kernel void PropagateKernal(global int *label, global int *pix, global int *flags, int pass, int iw, int ih)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int p0 = y * iw + x; // pixel index
	int g = label[p0]; //pixels label
	int og = g;

	if (x >= iw || y >= ih) // If it's not inside image boundaries
		return;
	

	if (flags[pass - 1] == 0) 
		return;

	if (g == -1) //if the pixel is turned off
		return;

	for (int yy = -1; yy <= 1; yy++) // this loop runs from our main pixel to all is surroundings and connects them by value if they are connected.
	{
		for (int xx = -1; xx <= 1; xx++)
		{
			if (0 <= x + xx &&  x + xx < iw && 0 <= y + yy &&  y + yy < ih)
			{
				const int p1 = (y + yy) * iw + x + xx, s = label[p1];
				if (s != -1 && s < g)
					g = s;
			}
		}
	}
	
	for (int j = 0; j<6; j++)
		g = label[g];

	if (g != og) //this operation gives the connected component is last minimun value
	{
		atomic_min(&label[og], g);
		atomic_min(&label[p0], g);
		flags[pass] = 1;
	}
}
