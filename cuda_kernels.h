#ifndef CUDA_KERNELS
#define CUDA_KERNELS

void compute_costs(pixel* d_pixels, cost_data d_costs, int w, int h, int current_w);
void compute_M(cost_data d_costs, int *d_M, int w, int h, int current_w);
void find_min(int* d_M, int* d_indices, int* d_indices_ref, int w, int h, int current_w);
void find_seam(int* d_M, int *d_indices, int *d_seam, int w, int h, int current_w);
void remove_seam(pixel *d_pixels, pixel *d_pixels_tmp, int *d_seam, int w, int h, int current_w);
//void update_costs(pixel *d_pixels, int *d_costs, int *d_costs_tmp, int *d_seam, int w, int h, int current_w);

#endif
