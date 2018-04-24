//
// openCL kernel for vector addition
//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void vecAdd(  __global float **dataset, 
				__global float **weights, 
				const unsigned int dim_x,
				const unsigned int dim_y,
				const unsigned int num_samples)
{


	
	// Get global thread ID 
	int id = get_global_id(0);

	// Make sure do not go out of bounds
	if (id < n) 
	{
		c[id] = a[id] + b[id];
	}
}