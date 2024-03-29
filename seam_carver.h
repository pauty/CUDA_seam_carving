#ifndef SEAM_CARVER
#define SEAM_CARVER

#include <cuda_runtime.h>

#include "cost_data.h"

typedef enum{
    SEAM_CARVER_STANDARD_MODE,
    SEAM_CARVER_UPDATE_MODE,
    SEAM_CARVER_APPROX_MODE
}seam_carver_mode;

typedef struct seam_carver{
    int w, h;
    seam_carver_mode mode;
    
    uchar4 *h_pixels;
    uchar4 *d_pixels;
    uchar4 *d_pixels_swap;
    
    cost_data d_costs;
    cost_data d_costs_swap;
    
    int *d_M; //sum map in approx mode
    
    //used in approx
    int *d_index_map;
    //int *h_index_map;
    int *d_offset_map;

    int *d_indices_ref;
    int *d_indices;
    
    int *d_seam;
    //int *h_seam; //used in approx
    
    int* reduce_row; //M row to consider for reduce
    int min_index;
    
    int current_w;
    
    unsigned char* output;
    
    //used in approx
    //cudaStream_t kernel_stream;
    //cudaStream_t copy_stream;
}seam_carver;

void seam_carver_init(seam_carver *sc, seam_carver_mode mode, unsigned char* img, int w, int h);
void seam_carver_resize(seam_carver *sc, int seams_to_remove);
void seam_carver_destroy(seam_carver *sc);

#endif
