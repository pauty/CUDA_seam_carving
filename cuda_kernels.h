#ifndef CUDA_KERNELS
#define CUDA_KERNELS

void compute_costs(pixel* d_pixels, unsigned int *d_costs, int w, int h, int current_w);
void compute_M(unsigned int *d_costs, unsigned int *d_M, int w, int h, int current_w);
void find_min(unsigned int* d_M, int* d_indices, int* d_indices_ref, int w, int h, int current_w);
void find_seam(unsigned int* d_M, int *d_indices, int *d_seam, int w, int h, int current_w);
void remove_seam(pixel *d_pixels, pixel *d_pixels_tmp, int *d_seam, int w, int h, int current_w);

#endif
