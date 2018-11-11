// System includes
#include <stdio.h>
//#include <assert.h>
#include <stdlib.h>
//#include <string.h>

// CUDA runtime
#include <cuda_runtime.h>

/*
// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
*/

#include "image.h"
#include "cost_data.h"

#include "cuda_kernels.h"
//#define STBI_ONLY_BMP
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//#define UPDATE_COSTS

const uint32_t H_BITMASK = 0x000000FF;  

uint32_t *build_pixels(const unsigned char *imgv, int w, int h){
    uint32_t *pixels = (uint32_t*)malloc(w*h*sizeof(uint32_t));
    int i, j;
    uint32_t pix;
    for(i = 0; i < h; i++){
        for(j = 0; j < w; j++){
            pix = 0;
            pix = (uint32_t)imgv[i*3*w + 3*j];
            pix = (pix<<8) + (uint32_t)imgv[i*3*w + 3*j + 1];
            pix = (pix<<8) + (uint32_t)imgv[i*3*w + 3*j + 2]; 
            pixels[i*w + j] = pix;
            /*
            printf("%u \n",pix);
            printf("original pixel: %d, %d, %d \n",(int)imgv[i*3*w + 3*j],(int)imgv[i*3*w + 3*j + 1], (int)imgv[i*3*w + 3*j + 2]);
            printf("unpacked pixel: %d, %d, %d \n",(int)(pix & H_BITMASK),(int)((pix>>8) & H_BITMASK),(int)((pix>>16) & H_BITMASK));
            getchar();
            */
            
        }
    }
    return pixels;
}

unsigned char *flatten_pixels(uint32_t *pixels, int w, int h, int new_w){
    unsigned char *flattened = (unsigned char*)malloc(3*new_w*h*sizeof(unsigned char));
    int i, j;
    uint32_t pix;
    for(i = 0; i < h; i++){
        for(j = 0; j < new_w; j++){ 
            pix = pixels[i*w + j];
            /*
            flattened[3*i*new_w + 3*j] = (unsigned char)(pix & H_BITMASK);
            flattened[3*i*new_w + 3*j + 1] = (unsigned char)((pix>>8) & H_BITMASK);
            flattened[3*i*new_w + 3*j + 2] = (unsigned char)((pix>>16) & H_BITMASK);
            */
            flattened[3*i*new_w + 3*j + 2] = (unsigned char)(pix & H_BITMASK);
            flattened[3*i*new_w + 3*j + 1] = (unsigned char)((pix>>8) & H_BITMASK);
            flattened[3*i*new_w + 3*j] = (unsigned char)((pix>>16) & H_BITMASK);
        }
    }
    return flattened;
}

int main(int argc, char **argv) {
    int w = 0, h = 0, ncomp = 0;
    unsigned char *imgv = NULL;
    uint32_t *pixels = NULL;
    uint32_t *d_pixels = NULL;
    uint32_t *d_pixels_tmp = NULL;
    uint32_t *pixels_swap = NULL;
    cost_data d_costs;
    int *d_M = NULL;
    
    int *indices_ref = NULL;
    int *d_indices_ref = NULL;
    int *d_indices = NULL;
    
    int *d_seam = NULL;
    int *seam = NULL; //debug
    
    int current_w, num_iterations;
    int i;
    long seams_to_remove;
    char *check;
    unsigned char *output;
    int success;

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
    
    pixels = build_pixels(imgv, w, h);
    free(imgv);

    cudaMalloc((void**)&d_pixels, w*h*sizeof(uint32_t)); 
    cudaMalloc((void**)&d_pixels_tmp, w*h*sizeof(uint32_t)); 
    
    cudaMalloc((void**)&(d_costs.left), w*h*sizeof(int)); 
    cudaMalloc((void**)&(d_costs.up), w*h*sizeof(int)); 
    cudaMalloc((void**)&(d_costs.right), w*h*sizeof(int)); 
    
    #ifdef UPDATE_COSTS
    cost_data d_costs_tmp;
    cost_data costs_swap;
    cudaMalloc((void**)&(d_costs_tmp.left), w*h*sizeof(int)); 
    cudaMalloc((void**)&(d_costs_tmp.up), w*h*sizeof(int)); 
    cudaMalloc((void**)&(d_costs_tmp.right), w*h*sizeof(int));
    #endif
    
    cudaMalloc((void**)&d_M, w*h*sizeof(int)); 

    //alloc on device for indices
    cudaMalloc((void**)&d_indices, w*sizeof(int)); 
    cudaMalloc((void**)&d_indices_ref, w*sizeof(int)); 
    cudaMalloc((void**)&d_seam, h*sizeof(int)); 
    
    //copy image pixels from host to device 
    cudaMemcpy(d_pixels, pixels, w*h*sizeof(uint32_t), cudaMemcpyHostToDevice);    
       
    //M = (int*)malloc(w*h*sizeof(int)); //TO REMOVE
    //seam = (int*)malloc(h*sizeof(int)); //TO REMOVE 
    
    #ifdef UPDATE_COSTS
    //call the kernel to calculate all costs (only once)
    compute_costs(d_pixels, d_costs, w, h, w);
    #endif
    
    current_w = w;
    num_iterations = 0;
    while(num_iterations < seams_to_remove){
        
        #ifndef UPDATE_COSTS
        //call the kernel to calculate all costs 
        compute_costs(d_pixels, d_costs, w, h, current_w);
        #endif
        
        
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
            for(i = 0; i < w; i++)
                indices_ref[i] = i;
            //wait for previous kernel to finish, copy reference to device   
            cudaMemcpy(d_indices_ref, indices_ref, w*sizeof(int), cudaMemcpyHostToDevice);
        }
               
        //call the kernel to find min index in the last row of M
        find_min(d_M, d_indices, d_indices_ref, w, h, current_w);
        
        
        //call the kernel to find the seam
        find_seam(d_M, d_indices, d_seam, w, h, current_w);
        
        //cudaMemcpy(seam, d_seam, h*sizeof(int), cudaMemcpyDeviceToHost); 
     
        /*
        for(i = 0; i < h; i++)
            printf("%d \n", seam[i]);
        getchar();
        */
        
        
        //call the kernel to remove seam
        remove_seam(d_pixels, d_pixels_tmp, d_seam, w, h, current_w);
        //swap pixels
        pixels_swap = d_pixels;
        d_pixels = d_pixels_tmp;
        d_pixels_tmp = pixels_swap;
        
        #ifdef UPDATE_COSTS 
        update_costs(d_pixels, d_costs, d_costs_tmp, d_seam, w, h, current_w);
        //swap costs
        costs_swap = d_costs;
        d_costs = d_costs_tmp;
        d_costs_tmp = costs_swap;
        #endif
        
        //decrease current w
        current_w = current_w - 1;
        num_iterations = num_iterations + 1;
    }
    
    //copy new pixel values back to the host
    cudaMemcpy(pixels, d_pixels, w*h*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
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
    
    output = flatten_pixels(pixels, w, h, current_w);
    success = stbi_write_bmp("img2.bmp", current_w, h, 3, output);
    
    printf("success : %d \n",success);
    
    cudaFree(d_pixels);
    cudaFree(d_pixels_tmp);
    cudaFree(d_costs.left);
    cudaFree(d_costs.up);
    cudaFree(d_costs.right);
    #ifdef UPDATE_COSTS
    cudaFree(d_costs_tmp.left);
    cudaFree(d_costs_tmp.up);
    cudaFree(d_costs_tmp.right);
    #endif
    cudaFree(d_M); 
    cudaFree(d_indices); 
    cudaFree(d_indices_ref); 
    cudaFree(d_seam);
    //free(M);
    free(pixels);
    free(indices_ref);
    free(output);
    if(seam != NULL)
        free(seam);


}
