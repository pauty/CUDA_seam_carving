#include <stdlib.h>
#include <stdio.h>

#include "seam_carver.h"
#include "cuda_kernels.h"

uchar4 *build_pixels(const unsigned char *imgv, int w, int h){
    uchar4 *pixels = (uchar4*)malloc(w*h*sizeof(uchar4));
    int i, j;
    uchar4 pix;
    for(i = 0; i < h; i++){
        for(j = 0; j < w; j++){
            pix.x = imgv[i*3*w + 3*j];
            pix.y = imgv[i*3*w + 3*j + 1];
            pix.z = imgv[i*3*w + 3*j + 2]; 
            pixels[i*w + j] = pix;            
        }
    }
    return pixels;
}

unsigned char *flatten_pixels(uchar4 *pixels, int w, int h, int new_w){
    unsigned char *flattened = (unsigned char*)malloc(3*new_w*h*sizeof(unsigned char));
    int i, j;
    uchar4 pix;
    for(i = 0; i < h; i++){
        for(j = 0; j < new_w; j++){ 
            pix = pixels[i*w + j];
            flattened[3*i*new_w + 3*j] = pix.x;
            flattened[3*i*new_w + 3*j + 1] = pix.y;
            flattened[3*i*new_w + 3*j + 2] = pix.z;
        }
    }
    return flattened;
}

void seam_carver_init(seam_carver *sc, seam_carver_mode mode, unsigned char* imgv, int w, int h){
    sc->w = w;
    sc->h = h;
    sc->current_w = w;
    sc->mode = mode;
    
    sc->h_pixels = build_pixels(imgv, w, h);
   
    cudaMalloc((void**)&sc->d_pixels, w*h*sizeof(uchar4)); 
    cudaMalloc((void**)&sc->d_pixels_swap, w*h*sizeof(uchar4)); 
    
    if(sc->mode != SEAM_CARVER_APPROX_MODE){
        cudaMalloc((void**)&(sc->d_costs.left), w*h*sizeof(short)); 
        cudaMalloc((void**)&(sc->d_costs.up), w*h*sizeof(short)); 
        cudaMalloc((void**)&(sc->d_costs.right), w*h*sizeof(short)); 
    }
    //printf("%lu \n",sizeof(short));
    
    if(sc->mode == SEAM_CARVER_UPDATE_MODE){
        cudaMalloc((void**)&(sc->d_costs_swap.left), w*h*sizeof(short)); 
        cudaMalloc((void**)&(sc->d_costs_swap.up), w*h*sizeof(short)); 
        cudaMalloc((void**)&(sc->d_costs_swap.right), w*h*sizeof(short));
    }
    
    cudaMalloc((void**)&sc->d_M, w*h*sizeof(int)); 
    
    if(sc->mode == SEAM_CARVER_APPROX_MODE)
        sc->reduce_row = &(sc->d_M[0]); //first row
    else
        sc->reduce_row = &(sc->d_M[w*(h-1)]); //last row
        
    if(sc->mode == SEAM_CARVER_APPROX_MODE){
        cudaMalloc((void**)&sc->d_index_map, w*h*sizeof(int));
        cudaMalloc((void**)&sc->d_offset_map, w*h*sizeof(int));
        //cudaMallocHost((void**)&sc->h_index_map, w*h*sizeof(int));
        //cudaMallocHost((void**)&sc->h_seam, h*sizeof(int));
        //cudaStreamCreate(&sc->kernel_stream);
        //cudaStreamCreate(&sc->copy_stream);
    }
    
    //alloc on device for indices
    cudaMalloc((void**)&sc->d_indices, w*sizeof(int)); 
    cudaMalloc((void**)&sc->d_indices_ref, w*sizeof(int)); 
    cudaMalloc((void**)&sc->d_seam, h*sizeof(int));    
}

void seam_carver_resize(seam_carver *sc, int seams_to_remove){
    uchar4* pixels_tmp;
    cost_data costs_tmp;
    int num_iterations;
    
    //copy image pixels from host to device 
    cudaMemcpy(sc->d_pixels, sc->h_pixels, sc->w*sc->h*sizeof(uchar4), cudaMemcpyHostToDevice);       
    int* indices = (int*)malloc(sc->w*sizeof(int));
    for(int i = 0; i < sc->w; i++){
        indices[i] = i;
    }     
    //copy indices reference to device
    cudaMemcpy(sc->d_indices_ref, indices, sc->w*sizeof(int), cudaMemcpyHostToDevice);   
    
    if(sc->mode == SEAM_CARVER_UPDATE_MODE){
        compute_costs(*sc);
    }
    
    num_iterations = 0;
    while(num_iterations < seams_to_remove){
        
        if(sc->mode == SEAM_CARVER_STANDARD_MODE){
            compute_costs(*sc);
        }
        
        if(sc->mode != SEAM_CARVER_APPROX_MODE){
            compute_M(*sc);
            find_min_index(*sc); 
            find_seam(*sc);
        }
        else{
            approx_setup(*sc);       
            approx_M(*sc);
            find_min_index(*sc);
            approx_seam(*sc);               
        }
        
        remove_seam(*sc);   
        //swap pixels
        pixels_tmp = sc->d_pixels;
        sc->d_pixels = sc->d_pixels_swap;
        sc->d_pixels_swap = pixels_tmp;
        
        if(sc->mode == SEAM_CARVER_UPDATE_MODE){
            update_costs(*sc);
            //swap costs
            costs_tmp = sc->d_costs;
            sc->d_costs = sc->d_costs_swap;
            sc->d_costs_swap = costs_tmp;
        }
             
        sc->current_w = sc->current_w - 1;
        num_iterations = num_iterations + 1;
    }
    
    cudaMemcpy(sc->h_pixels, sc->d_pixels, sc->w*sc->h*sizeof(uchar4), cudaMemcpyDeviceToHost);
    sc->output = flatten_pixels(sc->h_pixels, sc->w, sc->h, sc->current_w); 
    free(indices);
}

void seam_carver_destroy(seam_carver *sc){
    cudaFree(sc->d_pixels);
    cudaFree(sc->d_pixels_swap);
    if(sc->mode != SEAM_CARVER_APPROX_MODE){
        cudaFree(sc->d_costs.left);
        cudaFree(sc->d_costs.up);
        cudaFree(sc->d_costs.right);
    }
    if(sc->mode == SEAM_CARVER_UPDATE_MODE){
        cudaFree(sc->d_costs_swap.left);
        cudaFree(sc->d_costs_swap.up);
        cudaFree(sc->d_costs_swap.right);
    }
    cudaFree(sc->d_M); 
    cudaFree(sc->d_indices); 
    cudaFree(sc->d_indices_ref); 
    cudaFree(sc->d_seam);
    if(sc->mode == SEAM_CARVER_APPROX_MODE){
        cudaFree(sc->d_index_map);
        cudaFree(sc->d_offset_map);
    }
    free(sc->h_pixels);
    free(sc->output);   
}
