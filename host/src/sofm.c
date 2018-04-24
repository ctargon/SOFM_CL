
// 
// this file contains function declarations for SOFM related processes
//


#include "sofm.h"



float **initialize_weights(int dim_x, int dim_y, int n_features)
{
	float **weights = (float **) malloc (dim_x * dim_y * sizeof(float *));
	int i, j;

	for (i = 0; i < dim_x * dim_y; i++)
	{
		weights[i] = (float *) malloc (n_features * sizeof(float));
	}

	for (i = 0; i < dim_x * dim_y; i++)
	{
		for (j = 0; j < n_features; j++)
		{
			weights[i][j] = (float)rand() / (float)RAND_MAX;
		}
	}

	return weights;
}


void save_weights(char *fname, float **w, int x, int y, int n)
{
	int i;
	FILE *f = fopen(fname, "wb");

	for (i = 0; i < x * y; i++)
	{
		fwrite(w[i], sizeof(float), n, f);
	}
}


void print_weights_debug(float **w, int x, int y, int n)
{
	int i, j;

	for (i = 0; i < x * y; i++)
	{
		for (j = 0; j < n; j++)
		{
			printf("%5.3f\t", w[i][j]);
		}
		printf("\n");
	}
}


float **load_rand_colors(int size)
{
	float **X = (float **) malloc (COLOR_D * size * sizeof(float *));
	int i, j;

	for (i = 0; i < size; i++)
	{
		X[i] = (float *) malloc (COLOR_D * sizeof(float));
	}

	for (i = 0; i < size; i++)
	{
		for (j = 0; j < COLOR_D; j++)
		{
			X[i][j] = (float)(rand() % 256);
			X[i][j] /= MAX_RGB;
		}
	}

	return X;
}


struct coords find_bmu(float **weights, float *x, int dim_x, int dim_y, int n_features)
{
	int i, j, min_idx = -1;
	float min = FLT_MAX, dist;
	struct coords out;

	for (i = 0; i < dim_x * dim_y; i++)
	{
		dist = 0.0;
		for (j = 0; j < n_features; j++)
		{
			dist += pow(weights[i][j] - x[j], 2);
		}

		dist = sqrt(dist);

		if (dist < min)
		{
			min = dist;
			min_idx = i;
		}
	}

	out.x = min_idx / dim_x;
	out.y = min_idx % dim_y;

	return out;
}


float decay_radius(float init_r, int iter, float time_delay)
{
	return init_r * exp(-(float) iter / time_delay);
}


float decay_lr(float init_lr, int iter, int total_iters)
{
	return init_lr * exp(-(float) iter / (float) total_iters);
}


float calc_dist_from_bmu(int i, int j, struct coords c)
{
	return (float) sqrt(pow((float)(c.x - i), 2) + pow((float)(c.y - j), 2));
}


float calc_influence(float bmu_dist, float radius)
{
	return exp(-bmu_dist / (2 * pow(radius, 2)));
}


void update_weight_vec(float *w, float *x, float lr, float inf, int n_features)
{
	int i;
	float alpha = lr * inf;

	for (i = 0; i < n_features; i++)
	{
		w[i] = w[i] + (alpha * (x[i] - w[i]));
	}
}

