#ifndef SEAM_CARVER
#define SEAM_CARVER

#include <cuda_runtime.h>

#include "cost_data.h"

typedef enum{
    STANDARD,
    UPDATE
}carver_mode;

typedef struct seam_carver{
    int w, h;
    carver_mode mode;
    
    uchar4 *h_pixels;
    uchar4 *d_pixels;
    uchar4 *d_pixels_swap;
    
    cost_data d_costs;
    cost_data d_costs_swap;
    
    int *d_M;

    int *d_indices_ref;
    int *d_indices;
    
    int *d_seam;
    
    int* reduce_row; //M row to consider for reduce
    
    int current_w;
    
    unsigned char* output;
}seam_carver;

void seam_carver_init(seam_carver *sc, carver_mode mode, unsigned char* img, int w, int h);
void seam_carver_resize(seam_carver *sc, int seams_to_remove);
void seam_carver_free(seam_carver *sc);

#endif
