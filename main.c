// System includes
#include <stdio.h>
//#include <assert.h>
#include <stdlib.h>
#include <string.h>

// CUDA runtime
//#include <cuda_runtime.h>

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
    int w, h, ncomp;
    seam_carver_mode mode = SEAM_CARVER_STANDARD_MODE;
    int success;
    
    if(argc < 3){
        printf("usage: %s namefile seams_to_remove [options]\nvalid options:\n-u\tupdate costs instead of recomputing them\n-a\tapproximate computation\n", argv[0]);
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
    
    if(argc >= 4){
        if(strcmp(argv[3],"-u") == 0){
            mode = SEAM_CARVER_UPDATE_MODE;
            printf("update mode selected.\n");
        }
        else if(strcmp(argv[3],"-a") == 0){
            mode = SEAM_CARVER_APPROX_MODE;
            printf("approximation mode selected.\n");
        }
        else{    
            printf("an invalid option was specified and will be ignored. Valid options are: -u, -a.\n");
        }
    }
    
    printf("image loaded. Resizing...\n");
    seam_carver sc;
    seam_carver_init(&sc, mode, imgv, w, h);
    seam_carver_resize(&sc, (int)seams_to_remove);
    printf("image resized. Saving new image...\n");
   
    success = stbi_write_bmp("img2.bmp", sc.current_w, sc.h, 3, sc.output);
    printf("success : %d \n", success);
    
    seam_carver_destroy(&sc);
    
    return 0;
}
