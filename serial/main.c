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

#include "../host/src/sofm.h"


// High-resolution timer.
double getCurrentTimestamp() {
#ifdef _WIN32 // Windows
  // Use the high-resolution performance counter.

  static LARGE_INTEGER ticks_per_second = {};
  if(ticks_per_second.QuadPart == 0) {
    // First call - get the frequency.
    QueryPerformanceFrequency(&ticks_per_second);
  }

  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);

  double seconds = double(counter.QuadPart) / double(ticks_per_second.QuadPart);
  return seconds;
#else         // Linux
  timespec a;
  clock_gettime(CLOCK_MONOTONIC, &a);
  return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
#endif
}


int main (int argc, char **argv)
{
	// helper variables
	int i, ii, j;
	time_t t;

	float *x = NULL; // training example pointer
	int idx = -1; // index for random sample
	struct coords bmu_idx; // location of BMU
	float bmu_dist = 0.0;
	float influence = 0.0;
	float *w = NULL;

	srand((unsigned) time(&t));

	// define network dimensions
	int net_dim_x = 40;
	int net_dim_y = 40;

	// define hyperparameters
	int iters = 10000;
	float init_lr = 0.1;
	float lr = 0.1;
	int max_dim = max(net_dim_x, net_dim_y);
	float init_radius = (float) max_dim / 2.0;
	float radius = init_radius;
	int num_features = 36;
	float time_delay = (float) iters / log(init_radius);

	// load data
	int n_samples = 8157;
	// float *data = load_rand_colors(size, 1000);
	float *data = load_data_file("./data/myc_targets_v2_train_data.dat", num_features, n_samples);


	// initialize network weights
	float *net_weights = initialize_weights(net_dim_x, net_dim_y, num_features);
	//print_weights_debug(net_weights, net_dim_x, net_dim_y, num_features);
	save_weights((char *)"./weights/init_sofm_weights.dat", net_weights, net_dim_x, net_dim_y, num_features);

	// begin training process
	printf("beginning training...\n");

	double start_time = getCurrentTimestamp();

	for (i = 0; i < iters; i++)
	{
		if (i % 1000 == 0)
			printf("iter %d\n", i);
		
		// get training example
		idx = rand() % n_samples;
		x = &(data[idx * COLOR_D]);

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
				w = &(net_weights[ii * net_dim_x * num_features + j * num_features]);
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

	double end_time = getCurrentTimestamp();

	printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

	save_weights((char *)"./weights/final_sofm_weights.dat", net_weights, net_dim_x, net_dim_y, num_features);

	free(net_weights);

	free(data);

	return 0;
}

