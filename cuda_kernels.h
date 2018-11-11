#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include <stdint.h>

void compute_costs(uint32_t* d_pixels, cost_data d_costs, int w, int h, int current_w);
void compute_M(cost_data d_costs, int *d_M, int w, int h, int current_w);
void find_min(int* d_M, int* d_indices, int* d_indices_ref, int w, int h, int current_w);
void find_seam(int* d_M, int *d_indices, int *d_seam, int w, int h, int current_w);
void remove_seam(uint32_t *d_pixels, uint32_t *d_pixels_tmp, int *d_seam, int w, int h, int current_w);
void update_costs(uint32_t *d_pixels, cost_data d_costs, cost_data d_costs_tmp, int *d_seam, int w, int h, int current_w);

#endif
