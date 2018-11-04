// System includes
#include <stdio.h>
//#include <assert.h>
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
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

pixel *build_pixels(const unsigned char *imgv, int w, int h){
    pixel *pixels = (pixel*)malloc(w*h*sizeof(pixel));
    int i, j;
    for(i = 0; i < h; i++){
        for(j = 0; j < w; j++){
            pixels[i*w + j].r = imgv[i*3*w + 3*j];
            pixels[i*w + j].g = imgv[i*3*w + 3*j + 1];
            pixels[i*w + j].b = imgv[i*3*w + 3*j + 2]; 
        }
    }
    return pixels;
}

unsigned char *flatten_pixels(pixel *pixels, int w, int h, int new_w){
    unsigned char *flattened = (unsigned char*)malloc(3*new_w*h*sizeof(unsigned char));
    int i, j;
    for(i = 0; i < h; i++){
        for(j = 0; j < new_w; j++){           
            flattened[3*i*new_w + 3*j] = pixels[i*w + j].r;
            flattened[3*i*new_w + 3*j + 1] = pixels[i*w + j].g;
            flattened[3*i*new_w + 3*j + 2] = pixels[i*w + j].b;
        }
    }
    return flattened;
}

int main(int argc, char **argv) {
    int w = 0, h = 0, ncomp = 0;
    unsigned char *imgv = NULL;
    pixel *pixels = NULL;
    pixel *d_pixels;
    pixel *d_pixels_tmp;
    int *d_costs;
    int *d_costs_tmp;
    int *costs;
    int *M;
    int *d_M;
   
    int *indices_ref;
    int *d_indices_ref;
    int *d_indices;
    
    int *d_seam;
    int *seam;
    
    int i;
     
    imgv = stbi_load("imgs/beach.bmp", &w, &h, &ncomp, 0);
    if(ncomp != 3)
        printf("ERROR -- image does not have 3 components (RGB)\n");
    pixels = build_pixels(imgv, w, h);
    free(imgv);

    cudaMalloc((void**)&d_pixels, w*h*sizeof(pixel)); 
    cudaMalloc((void**)&d_pixels_tmp, w*h*sizeof(pixel)); 
    cudaMalloc((void**)&d_costs, 3*w*h*sizeof(int)); 
    cudaMalloc((void**)&d_costs_tmp, 3*w*h*sizeof(int));  /////////////////////
    cudaMalloc((void**)&d_M, w*h*sizeof(int)); 

    //alloc on device for indices
    cudaMalloc((void**)&d_indices, w*sizeof(int)); 
    cudaMalloc((void**)&d_indices_ref, w*sizeof(int)); 
    cudaMalloc((void**)&d_seam, h*sizeof(int)); 
    
    //copy image pixels from host to device 
    cudaMemcpy(d_pixels, pixels, w*h*sizeof(pixel), cudaMemcpyHostToDevice);    
       
    //M = (int*)malloc(w*h*sizeof(int)); //TO REMOVE
    //seam = (int*)malloc(h*sizeof(int)); //TO REMOVE 
    
    //call the kernel to calculate all costs 
    //compute_costs(d_pixels, d_costs, w, h, w);
    
    int current_w = w;
    int num_iterations = 0;
    while(num_iterations < 1500){
        
        
        //call the kernel to calculate all costs 
        compute_costs(d_pixels, d_costs, w, h, current_w);
        
        
        //call the kernel to compute comulative map
        compute_M(d_costs, d_M, w, h, current_w);
        
        //cudaMemcpy(M, d_M, w*h*sizeof(int), cudaMemcpyDeviceToHost);
        
        /*
        for(i = (row)*w; i < (row+1)*w; i++)
            printf("%d \n", M[i]);
        getchar();
        */
        
        //only on the first iteration, initialize indices reference (in parallel with kernel execution)
        if(num_iterations == 0){
            indices_ref = (int*)malloc(w*sizeof(int));
            for(int i = 0; i < w; i++)
                indices_ref[i] = i;
            //wait for previous kernel to finish, copy reference to device   
            cudaMemcpy(d_indices_ref, indices_ref, w*sizeof(int), cudaMemcpyHostToDevice);
        }
               
        //call the kernel to find min index in the last row of M
        find_min(d_M, d_indices, d_indices_ref, w, h, current_w);
        
        
        //call the kernel to find the seam
        find_seam(d_M, d_indices, d_seam, w, h, current_w);
        
        //cudaMemcpy(seam, d_seam, h*sizeof(pixel), cudaMemcpyDeviceToHost);
        
        
        /*
        for(i = 0; i < h; i++)
            printf("%d \n", seam[i]);
        getchar();*/
        
        
        //call the kernel to remove seam
        remove_seam(d_pixels, d_pixels_tmp, d_seam, w, h, current_w);
        
        //update_costs(d_pixels, d_costs, d_costs_tmp, d_seam, w, h, current_w);
      
        //decrease current w
        current_w = current_w - 1;
        num_iterations = num_iterations + 1;
    }
    
    //copy new pixel values back to the host
    cudaMemcpy(pixels, d_pixels, w*h*sizeof(pixel), cudaMemcpyDeviceToHost);
    
    /*
    int i;
    for(i = 40; i < w*h; i = i+w){
        if(i % w == 0)
            printf("\n");
        printf("%d ",costs[i]);
    }*/
    
    /*
    for(i = 260; i < w*h; i = i+w){
        printf("[ %d %d % d ] \n",pixels[i].r, pixels[i].g, pixels[i].b);
    }*/
    
    /*
    printf("\n\n -------- \n");
    
    for(i = 0; i < h; i++){
        printf("%d ", seam[i]);
    }*/
    
    unsigned char *output = flatten_pixels(pixels, w, h, current_w);
    int success = stbi_write_bmp("img2.bmp", current_w, h, 3, output);
    
    printf("success : %d \n",success);
    
    cudaFree(d_pixels);
    cudaFree(d_pixels_tmp);
    cudaFree(d_costs);
    cudaFree(d_M); 
    cudaFree(d_indices); 
    cudaFree(d_indices_ref); 
    cudaFree(d_seam);
    //free(M);
    free(pixels);
    free(indices_ref);
    free(output);

}
