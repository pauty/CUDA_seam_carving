#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include "seam_carver.h"

void compute_costs(seam_carver sc);
void compute_M(seam_carver sc);
void find_min_index(seam_carver sc, short use_stream);
void find_seam(seam_carver sc);
void remove_seam(seam_carver sc);
void update_costs(seam_carver sc);

void approx_setup(seam_carver sc);
void approx_M(seam_carver sc);

#endif
