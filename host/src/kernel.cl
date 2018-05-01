//
// openCL kernel for vector addition
//

float get_dist(__global float *x, __global float *w, int n_features)
{
	int i;
	float dist = 0.0;

	#pragma unroll 3
	for (i = 0; i < n_features; i++)
	{
		dist += pow(w[i] - x[i], 2);
	}

	return sqrt(dist);
}


float dist_from_bmu(int x, int y, int idx_x, int idx_y)
{
	return (float) sqrt(pow((float)(idx_x - x), 2) + pow((float)(idx_y - y), 2));
}


float calc_influence(float bmu_dist, float radius)
{
	return exp(-bmu_dist / (2 * pow(radius, 2)));
}


__kernel void find_bmu(  __global float *dataset, 
				__global float *weights, 
				__global float *distances,
				const unsigned int dim_x,
				const unsigned int dim_y,
				const unsigned int rand_idx,
				const unsigned int n_features)
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


__kernel void update_weights( __global float *dataset, 
				__global float *weights, 
				__global float *distances,
				const int bmu_idx,
				const unsigned int dim_x,
				const unsigned int dim_y,
				const unsigned int rand_idx,
				const unsigned int n_features,
				const float radius,
				const float lr)
{
	__global float *w = NULL;
	__local float bmu_dist, influence, alpha;
	__global float *x = &(dataset[rand_idx * n_features]);
	int x_c, y_c, i;

	// Get global thread ID 
	int id = get_global_id(0);

	// Make sure do not go out of bounds
	if (id < dim_x * dim_y) 
	{
		x_c = id / dim_x;
		y_c = id % dim_y;
		
		w = &(weights[x_c * dim_x * n_features + y_c * n_features]);

		bmu_dist = dist_from_bmu(x_c, y_c, bmu_idx / dim_x, bmu_idx % dim_y);

		if (bmu_dist <= radius)
		{
			influence = calc_influence(bmu_dist, radius);

			alpha = lr * influence;

			#pragma unroll 3
			for (i = 0; i < n_features; i++)
			{
				w[i] = w[i] + (alpha * (x[i] - w[i]));
			}
		}
	}
}