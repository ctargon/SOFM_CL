
// 
// this file contains function declarations for SOFM related processes
//


#include "sofm.h"



float *initialize_weights(int dim_x, int dim_y, int n_features)
{
	float *weights = (float *) malloc (dim_x * dim_y * n_features * sizeof(float));
	int i, j, k;

	for (i = 0; i < dim_x; i++)
	{
		for (j = 0; j < dim_y; j++)
		{
			for (k = 0; k < n_features; k++)
			{
				weights[i*dim_x*n_features + j*n_features + k] = (float)rand() / (float)RAND_MAX;
			}
		}
	}

	return weights;
}


void save_weights(char *fname, float *w, int x, int y, int n)
{
	int i, j;
	FILE *f = fopen(fname, "wb");

	for (i = 0; i < x; i++)
	{
		for (j = 0; j < y; j++)
		{
			fwrite(&(w[i*x*n + j*n]), sizeof(float), n, f);
		}
	}
}


void print_weights_debug(float *w, int x, int y, int n)
{
	int i, j, k;

	for (i = 0; i < x; i++)
	{
		for (j = 0; j < y; j++)
		{
			for (k = 0; k < n; k++)
			{
				printf("%5.3f\t", w[i*x*n + j*n + k]);
			}
			printf("\n");
		}
	}
}


float *load_data_file(char *file, int n_features, int size)
{
	float *X = (float *) malloc (n_features * size * sizeof(float));

	FILE *fptr = fopen(file, "rb");
	fread(X, sizeof(float), n_features * size, fptr);
	fclose(fptr);

	return X;
}


float *load_rand_colors(int size, int n_features)
{
	float *X = (float *) malloc (n_features * size * sizeof(float));
	int i, j;

	for (i = 0; i < size; i++)
	{
		for (j = 0; j < n_features; j++)
		{
			X[i * n_features + j] = (float)(rand() % 256);
			X[i * n_features + j] /= MAX_RGB;
		}
	}

	return X;
}


struct coords find_bmu(float *weights, float *x, int dim_x, int dim_y, int n_features)
{
	int i, j, k;
	float min = FLT_MAX, dist;
	struct coords out;

	for (i = 0; i < dim_x; i++)
	{
		for (j = 0; j < dim_y; j++)
		{
			dist = 0.0;
			for (k = 0; k < n_features; k++)
			{
				dist += pow(weights[i*dim_x*n_features + j*n_features + k] - x[k], 2);
			}

			dist = sqrt(dist);

			if (dist < min)
			{
				min = dist;
				out.x = i;
				out.y = j;
			}
		}
		
	}

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

