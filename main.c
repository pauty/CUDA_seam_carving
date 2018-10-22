// System includes
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

/*
// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
*/

#include "image.h"


#include "cuda_kernels.h"
//#define STBI_ONLY_BMP
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

pixel *build_pixels(const unsigned char *imgv, int w, int h){
    pixel *pixels = (pixel*)malloc(w*h*sizeof(pixel));
    int i, j;
    for(i = 0; i < h; i++){
        for(j = 0; j < w; j++){
            pixels[i*w + j].r = imgv[i*3*w + 3*j];
            pixels[i*w + j].g = imgv[i*3*w + 3*j + 1];
            pixels[i*w + j].b = imgv[i*3*w + 3*j + 2];
            //pixels[i*w + j].a = (unsigned char)255;
            //printf("%d %d %d; ",pixels[i*w + j].r,pixels[i*w + j].g,pixels[i*w + j].b);
        }
        //printf("endrow\n");
    }
    //printf("END------\n\n");
    return pixels;
}

int main(int argc, char **argv) {
    int w = 0, h = 0, ncomp = 0;
    unsigned char *imgv = NULL;
    pixel *pixels = NULL;
    pixel *d_pixels;
    unsigned int *d_costs;
    unsigned int *costs;
    unsigned int *M;
    unsigned int *d_M;
   
    int *indices_ref;
    int *d_indices_ref;
    int *d_indices;
     
    imgv = stbi_load("imgs/coast.bmp", &w, &h, &ncomp, 0);
    if(ncomp != 3)
        printf("ERROR -- image does not have 3 components (RGB)\n");
    pixels = build_pixels(imgv, w, h);
    free(imgv);

    cudaMalloc((void**)&d_pixels, w*h*sizeof(pixel)); 
    cudaMalloc((void**)&d_costs, 3*w*h*sizeof(unsigned int)); 
    cudaMalloc((void**)&d_M, w*h*sizeof(unsigned int)); 
    cudaMalloc((void**)&d_indices, w*sizeof(int)); 
    cudaMalloc((void**)&d_indices_ref, w*sizeof(int)); 
    
    cudaMemcpy(d_pixels, pixels, w*h*sizeof(pixel), cudaMemcpyHostToDevice);    
    
    //call the kernel to calculate all costs (done once at the start)
    compute_costs(d_pixels, d_costs, w, h, w);
    
    M = (unsigned int*)malloc(w*h*sizeof(unsigned int)); //TO REMOVE
    
    indices_ref = (int*)malloc(w*sizeof(int));
    for(int i = 0; i < w; i++)
        indices_ref[i] = i;
        
    cudaMemcpy(d_indices_ref, indices_ref, w*sizeof(int), cudaMemcpyHostToDevice);
    
    //here start the loop
    
    //call the kernel to compute comulative map
    compute_M(d_costs, d_M, w, h, w);
    
    //set the reference index array
    cudaMemcpy(d_indices, d_indices_ref, w*sizeof(int), cudaMemcpyDeviceToDevice);
    
    //call the kernel to find min
    find_min(d_M, d_indices ,w, h, w);
    
    //kernel to find the seam
    
    //remove seam
    
    //update costs matrix near removed seam
    
    //end loop - decrease current w
    
    cudaMemcpy(M, d_M, w*h*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(indices_ref, d_indices, w*sizeof(int), cudaMemcpyDeviceToHost);
    
    /*
    int i;
    for(i = 40; i < w*h; i = i+w){
        if(i % w == 0)
            printf("\n");
        printf("%d ",costs[i]);
    }*/
    int i;
    for(i = w-1; i < w*h; i = i+w){
        if(i % w == 0)
            printf("\n");
        printf("%d ",M[i]);
    }
    
    printf("\n\n min: %d \n", indices_ref[0]);
    
    cudaFree(d_pixels);
    cudaFree(d_costs);
    free(M);
    free(pixels);

}
