/*
 * main.c
 * Colin Targonski
 * 4/10/2018
 *
 * This is the main function for running a serial implementation of a self organizing
 * feature map. It takes in a data of a specified input and learns over time where
 * each data point can be mapped to on a two dimensional grid for clustering
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>


// custom defines
#define max(x,y) ((x) >= (y)) ? (x) : (y)
#define COLOR_D 3
#define MAX_RGB 255

// dtypes
struct coords
{
	int x;
	int y;
};

// function declarations
void print_weights_debug(float **w, int x, int y, int n);
float **initialize_weights(int dim_x, int dim_y, int n_features);
void save_weights(char *fname, float **w, int x, int y, int n);
float **load_rand_colors(int size);
struct coords find_bmu(float **weights, float *x, int dim_x, int dim_y, int n_features);
float decay_radius(float init_r, int iter, float time_delay);
float decay_lr(float init_lr, int iter, int total_iters);
float calc_dist_from_bmu(int i, int j, struct coords c);
float calc_influence(float bmu_dist, float radius);
void update_weight_vec(float *w, float *x, float lr, float inf, int n_features);


int main (int argc, char **argv)
{
	// helper variables
	int i, ii, j;
	time_t t;
	FILE *weight_f = NULL;

	float *x = NULL; // training example pointer
	int idx = -1; // index for random sample
	struct coords bmu_idx; // location of BMU
	float bmu_dist = 0.0;
	float influence = 0.0;
	float *w = NULL;

	srand((unsigned) time(&t));

	// load data
	int size = 200;
	float **color_data = load_rand_colors(size);

	// define network dimensions
	int net_dim_x = 160;
	int net_dim_y = 160;

	// define hyperparameters
	int iters = 5000;
	float init_lr = 0.1;
	float lr = 0.1;
	int max_dim = max(net_dim_x, net_dim_y);
	float init_radius = (float) max_dim / 2.0;
	float radius = init_radius;
	int num_features = 3;
	float time_delay = (float) iters / log(init_radius);


	// initialize network weights
	float **net_weights = initialize_weights(net_dim_x, net_dim_y, num_features);
	print_weights_debug(net_weights, net_dim_x, net_dim_y, num_features);
	save_weights("../weights/init_sofm_weights.dat", net_weights, net_dim_x, net_dim_y, num_features);

	// begin training process

	for (i = 0; i < iters; i++)
	{
		printf("iter %d\n", i);
		// get training example
		idx = rand() % size;
		x = color_data[idx];

		// find best matching unit
		bmu_idx = find_bmu(net_weights, x, net_dim_x, net_dim_y, num_features);

		// get new radius and new learning rate
		radius = decay_radius(init_radius, i, time_delay);
		lr = decay_lr(init_lr, i, iters);
		// if ((i + 1) % 100 == 0) printf("%5.4f\t%5.4f\n", radius, lr);

		// calculate distance to the BMU, if within radius, update it
		for (ii = 0; ii < net_dim_x; ii++)
		{
			for (j = 0; j < net_dim_y; j++)
			{
				w = net_weights[(ii * net_dim_x) + j];
				bmu_dist = calc_dist_from_bmu(ii, j, bmu_idx);

				// update weight if within range
				if (bmu_dist <= radius)
				{
					influence = calc_influence(bmu_dist, radius);
					update_weight_vec(w, x, lr, influence, num_features);
				}
			}
		}
	}








	save_weights("../weights/final_sofm_weights.dat", net_weights, net_dim_x, net_dim_y, num_features);




	// free memory
	for (i = 0; i < net_dim_x; i++)
	{
		free(net_weights[i]);
	}
	free(net_weights);

	for (i = 0; i < size; i++)
	{
		free(color_data[i]);
	}
	free(color_data);


	return 0;
}


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






