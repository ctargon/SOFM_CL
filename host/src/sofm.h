//
// prototypes for sofm.c
//


#include <stdio.h>
#include <stdlib.h>
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


// data/weight related functions
float *initialize_weights(int dim_x, int dim_y, int n_features);
void save_weights(char *fname, float *w, int x, int y, int n);
void print_weights_debug(float *w, int x, int y, int n);
float *load_rand_colors(int size);

// sofm related functions
struct coords find_bmu(float *weights, float *x, int dim_x, int dim_y, int n_features);
float decay_radius(float init_r, int iter, float time_delay);
float decay_lr(float init_lr, int iter, int total_iters);
float calc_dist_from_bmu(int i, int j, struct coords c);
float calc_influence(float bmu_dist, float radius);
void update_weight_vec(float *w, float *x, float lr, float inf, int n_features);


