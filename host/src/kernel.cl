//
// openCL kernel for vector addition
//

float get_dist(__global float *x, __global float *w, int n_features)
{
	int i;
	float dist = 0.0;
	for (i = 0; i < n_features; i++)
	{
		dist += pow(w[i] - x[i], 2);
	}

	return sqrt(dist);
}


__kernel void find_bmu(  __global float *dataset, 
				__global float *weights, 
				__global float *distances,
				const unsigned int dim_x,
				const unsigned int dim_y,
				const unsigned int rand_idx,
				const unsigned int n_features,
				__global int *bmu_idx)
{
	__global float *x = &(dataset[rand_idx * n_features]);
	int i, x_c, y_c, min_idx = -1;
	float min = 99999999.9;


	// Get global thread ID 
	int id = get_global_id(0);

	// Make sure do not go out of bounds
	if (id < dim_x * dim_y) 
	{
		x_c = id / dim_x;
		y_c = id % dim_y;
		distances[id] = get_dist(x, &(weights[x_c * n_features * dim_x + y_c * n_features]), n_features);
	}

}