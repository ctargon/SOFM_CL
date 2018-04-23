//
// openCL kernel for vector addition
//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void vecAdd(  __global unsigned short *a, __global unsigned short *b, __global unsigned short *c, const unsigned int n)
{
	// Get global thread ID 
	int id = get_global_id(0);

	// Make sure do not go out of bounds
	if (id < n) 
	{
		c[id] = a[id] + b[id];
	}
}