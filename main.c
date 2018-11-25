// System includes
#include <stdio.h>
//#include <assert.h>
#include <stdlib.h>
//#include <string.h>

// CUDA runtime
//#include <cuda_runtime.h>

/*
// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
*/

#include "seam_carver.h"

//#define STBI_ONLY_BMP
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main(int argc, char **argv) {
    
    unsigned char* imgv;
    long seams_to_remove;
    char *check;
    int success;
    int w, h, ncomp;

    if(argc < 3){
        printf("usage: %s namefile seams_to_remove\n", argv[0]);
        return 1;
    }
    
    seams_to_remove = strtol(argv[2], &check, 10);  //10 specifies base-10
    if (check == argv[2]){   //if no characters were converted pointers are equal
        printf("ERROR: can't convert string to number, exiting.\n");
        return 1;
    }
    
    imgv = stbi_load(argv[1], &w, &h, &ncomp, 0);
    if(imgv == NULL){
        printf("ERROR: can't load image \"%s\" (maybe the file does not exist?), exiting.\n", argv[1]);
        return 1;
    }
    if(ncomp != 3){
        printf("ERROR: image does not have 3 components (RGB), exiting.\n");
        return 1;
    }
    
    if(seams_to_remove < 0 || seams_to_remove >= w){
        printf("ERROR: number of seams to remove is invalid, exiting.\n");
        return 1;
    }
    
    printf("image loaded. Resizing...\n");
    seam_carver sc;
    seam_carver_init(&sc, APPROX, imgv, w, h);
    seam_carver_resize(&sc, (int)seams_to_remove);
    printf("image resized. Saving new image...\n");

    
    success = stbi_write_bmp("img2.bmp", sc.current_w, sc.h, 3, sc.output);
    printf("success : %d \n", success);
    
    seam_carver_free(&sc);
    
    return 0;
}
