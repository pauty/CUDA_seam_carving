//#include <cuda_runtime.h>

extern "C"{
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>

#include "image.h"
#include "cost_data.h"
}

__constant__ pixel BORDER_PIXEL = {.r = 0, .g = 0, .b = 0};
__constant__ uint32_t D_BITMASK = 0x000000FF;


__device__ pixel pixel_from_int(uint32_t intpix){
    pixel pix;
    pix.b = (int)((intpix & D_BITMASK));
    pix.g = (int)((intpix>>8) & D_BITMASK);
    pix.r = (int)((intpix>>16) & D_BITMASK);
    return pix;
}   

//#define COMPUTE_COSTS_FULL

//#define COMPUTE_M_SINGLE
//#define COMPUTE_M_ITERATE

const int BLOCKSIZE_X = 32;
const int BLOCKSIZE_Y = 8;

#ifndef COMPUTE_COSTS_FULL

__global__ void compute_costs_kernel(uint32_t *d_pixels, cost_data d_costs, int w, int h, int current_w){
    //first row, first coloumn and last coloumn of shared memory are reserved for halo...
    __shared__ pixel pix_cache[BLOCKSIZE_Y][BLOCKSIZE_X];
    //...and the global index in the image is computed accordingly to this 
    int row = blockIdx.y*(BLOCKSIZE_Y-1) + threadIdx.y -1 ; 
    int coloumn = blockIdx.x*(BLOCKSIZE_X-2) + threadIdx.x -1; 
    int ix = row*w + coloumn;
    int cache_row = threadIdx.y;
    int cache_coloumn = threadIdx.x;
    short active = 0;
     
    if(row < h && coloumn <= current_w){
        //only threads with row in [-1,h-1] and coloumn in [-1,current_w] are actually active
        active = 1;
        //if access to the image is out of bounds, set RGB values to 0
        //otherwise load pixel from global memory
        if(row < 0 || coloumn < 0 || coloumn == current_w){
            pix_cache[cache_row][cache_coloumn] = BORDER_PIXEL;
        }
        else{
            pix_cache[cache_row][cache_coloumn] = pixel_from_int(d_pixels[ix]);
        }
    }
    
    //wait until each thread has initialized its portion of shared memory
    __syncthreads();
    
    //all the threads that are NOT in halo positions can now compute costs, with fast access to shared memory
    if(active && cache_row != 0 && cache_coloumn != 0 
       && cache_coloumn != BLOCKSIZE_X-1 && coloumn < current_w){
        int rdiff, gdiff, bdiff;
        int p_r, p_g, p_b;
        pixel pix1, pix2, pix3;
        
        pix1 = pix_cache[cache_row][cache_coloumn+1];
        pix2 = pix_cache[cache_row][cache_coloumn-1];
        pix3 = pix_cache[cache_row-1][cache_coloumn];
        
        //compute partials
        p_r = abs(pix1.r - pix2.r);
        p_g = abs(pix1.g - pix2.g);
        p_b = abs(pix1.b - pix2.b);
        
        //compute left cost       
        rdiff = p_r + abs(pix3.r - pix2.r);
        gdiff = p_g + abs(pix3.g - pix2.g);
        bdiff = p_b + abs(pix3.b - pix2.b);
        d_costs.left[ix] = rdiff + gdiff + bdiff;
        
        //compute up cost
        d_costs.up[ix] = p_r + p_g + p_b;
        
        //compute right cost
        rdiff = p_r + abs(pix3.r - pix1.r);
        gdiff = p_g + abs(pix3.g - pix1.g);
        bdiff = p_b + abs(pix3.b - pix1.b);
        d_costs.right[ix] = rdiff + gdiff + bdiff;         
    }    
}

#else

__global__ void compute_costs_kernel(uint32_t* d_pixels, cost_data d_costs, int w, int h, int current_w){
    __shared__ pixel pix_cache[BLOCKSIZE_Y+1][BLOCKSIZE_X+2];
    int row = blockIdx.y*BLOCKSIZE_Y + threadIdx.y; 
    int coloumn = blockIdx.x*BLOCKSIZE_X + threadIdx.x; 
    int ix = row*w + coloumn;
    int cache_row = threadIdx.y + 1;
    int cache_coloumn = threadIdx.x + 1;
    short active = 0;
     
    if(row < h && coloumn < current_w){
        active = 1;
        if(threadIdx.x == 0){
            if(coloumn == 0)
                pix_cache[cache_row][0] = BORDER_PIXEL;
            else
                pix_cache[cache_row][0] = pixel_from_int(d_pixels[ix-1]);//pix_cache[cache_row][0] = d_pixels[ix-1];
        }
        if(threadIdx.x == BLOCKSIZE_X-1 || coloumn == current_w-1){
            if(coloumn == current_w-1)
                pix_cache[cache_row][cache_coloumn+1] = BORDER_PIXEL;
            else
                pix_cache[cache_row][BLOCKSIZE_X+1] = pixel_from_int(d_pixels[ix+1]);//pix_cache[cache_row][BLOCKSIZE_X+1] = d_pixels[ix+1];
        }
        if(threadIdx.y == 0){
            if(row == 0)
                pix_cache[0][cache_coloumn] = BORDER_PIXEL;  
            else
                pix_cache[0][cache_coloumn] = pixel_from_int(d_pixels[ix-w]);//pix_cache[0][cache_coloumn] = d_pixels[ix-w];            
        } 
        pix_cache[cache_row][cache_coloumn] = pixel_from_int(d_pixels[ix]);  
        //pix_cache[cache_row][cache_coloumn] = d_pixels[ix];        
    }
    
    //wait until each thread has initialized its portion of shared memory
    __syncthreads();
    
    //all the threads that are NOT in halo positions can now compute costs, with fast access to shared memory
    if(active){
        int rdiff, gdiff, bdiff;
        int p_r, p_g, p_b;
        pixel pix1, pix2, pix3;
        
        pix1 = pix_cache[cache_row][cache_coloumn+1];
        pix2 = pix_cache[cache_row][cache_coloumn-1];
        pix3 = pix_cache[cache_row-1][cache_coloumn];
        
        //compute partials
        p_r = abs(pix1.r - pix2.r);
        p_g = abs(pix1.g - pix2.g);
        p_b = abs(pix1.b - pix2.b);
        
        //compute left cost       
        rdiff = p_r + abs(pix3.r - pix2.r);
        gdiff = p_g + abs(pix3.g - pix2.g);
        bdiff = p_b + abs(pix3.b - pix2.b);
        d_costs.left[ix] = rdiff + gdiff + bdiff;
        
        //compute up cost
        d_costs.up[ix] = p_r + p_g + p_b;
        
        //compute right cost
        rdiff = p_r + abs(pix3.r - pix1.r);
        gdiff = p_g + abs(pix3.g - pix1.g);
        bdiff = p_b + abs(pix3.b - pix1.b);
        d_costs.right[ix] = rdiff + gdiff + bdiff; 
     }
}

#endif

#if !defined(COMPUTE_M_SINGLE) && !defined(COMPUTE_M_ITERATE)

const int WIDEBLOCKSIZE = 128; //must be divisible by 2

__global__ void compute_M_kernel_step1(cost_data d_costs, int* d_M, int w, int h, int current_w, int base_row){
    __shared__ int m_cache[2*WIDEBLOCKSIZE];
    int row;
    int coloumn = blockIdx.x*WIDEBLOCKSIZE + threadIdx.x; 
    int cache_coloumn = threadIdx.x; 
    short is_first;
    short is_last;
    int right, up, left;
    
    is_first = blockIdx.x == 0;
    is_last = blockIdx.x == gridDim.x-1;
    
    if(coloumn < current_w){
        if(base_row == 0){
            left = min(d_costs.left[coloumn], min(d_costs.up[coloumn], d_costs.right[coloumn]));
            m_cache[cache_coloumn] = left;
            d_M[coloumn] = left; 
        }
        else{
            m_cache[cache_coloumn] = d_M[base_row*w + coloumn];    
        }
    }
    //wait until shared memory load is complete
    __syncthreads();
    
    int shift = 0; 
    int ix;
    int inc = 0;
    for(row = base_row+1; row < base_row + WIDEBLOCKSIZE/2 && row < h; row++){
        inc++;
        if((is_first || inc - 1 < threadIdx.x) && (is_last || threadIdx.x < WIDEBLOCKSIZE - inc) && coloumn < current_w){
            ix = row*w + coloumn;
            
            //with left
            if(coloumn > 0)
                left = m_cache[cache_coloumn - 1 + shift] + d_costs.left[ix]; 
            else 
                left = INT_MAX;
            //with up
            up = m_cache[cache_coloumn + shift] + d_costs.up[ix];
            //with right
            if(coloumn < current_w-1)
                right = m_cache[cache_coloumn + 1 + shift] + d_costs.right[ix];
            else
                right = INT_MAX;
                
            left = min(left, min(up, right));           
            d_M[ix] = left;
            //swap read/write shared memory
            shift = WIDEBLOCKSIZE - shift;
            m_cache[cache_coloumn + shift] = left;
        }   
        //wait until every thread has written shared memory
        __syncthreads();                
    }
}


__global__ void compute_M_kernel_step2(cost_data d_costs, int* d_M, int w, int h, int current_w, int base_row){
    //__shared__ int m_cache[WIDEBLOCKSIZE];
    int row;
    int coloumn = blockIdx.x*WIDEBLOCKSIZE + threadIdx.x + WIDEBLOCKSIZE/2; 
    // cache_coloumn = threadIdx.x; 
    int right, up, left;
    right = INT_MAX;
   
    //if(coloumn < current_w)
    //   m_cache[cache_coloumn] = d_costs[base_row*w + coloumn];
        
    //wait until shared memory load is complete
    //__syncthreads();
    
    int ix, prev_ix;
    int inc = 0;
    for(row = base_row+1; row < base_row + WIDEBLOCKSIZE/2 && row < h; row++){
        inc++;
        if((WIDEBLOCKSIZE/2 - inc -1 < threadIdx.x) && (threadIdx.x < WIDEBLOCKSIZE/2 + inc) && coloumn < current_w){
            ix = row*w + coloumn;
            prev_ix = ix - w; //(row-1)*w + coloumn
            //with left
            left = d_M[prev_ix - 1] + d_costs.left[ix]; 
            //with up
            up = d_M[prev_ix] + d_costs.up[ix];
            //with right
            if(coloumn < current_w-1)
                right = d_M[prev_ix + 1] + d_costs.right[ix];
            else
                right = INT_MAX;
            left = min(left, min(up, right));               
            d_M[ix] = left;
        }
        __syncthreads();
    }
}

__global__ void compute_M_kernel_small(cost_data d_costs, int* d_M, int w, int h, int current_w){
    extern __shared__ int m_cache[];
    int coloumn = threadIdx.x;
    int row, ix;
    int left, up, right;
    
    //first row
    left = min(d_costs.left[coloumn], min(d_costs.up[coloumn], d_costs.right[coloumn]));
    d_M[coloumn] = left; 
    m_cache[coloumn] = left;
    
    __syncthreads(); 
    
    int shift = 0;
    
    //other rows
    for(row = 1; row < h; row++){
        if(coloumn < current_w){
            ix = row*w + coloumn;    
            //with left
            if(coloumn > 0)
                left = m_cache[coloumn - 1 + shift] + d_costs.left[ix]; 
            else
                left = INT_MAX;
            //with up
            up = m_cache[coloumn + shift] + d_costs.up[ix];
            //with right
            if(coloumn < current_w-1)
                right = m_cache[coloumn + 1 + shift] + d_costs.right[ix];
            else
                right = INT_MAX;

            left = min(left, min(up, right));            
            d_M[ix] = left;
            //swap read/write shared memory
            shift = current_w - shift;   
            m_cache[coloumn + shift] = left;
        }
        __syncthreads();    
    }     
}

#endif

#ifdef COMPUTE_M_SINGLE

__global__ void compute_M_kernel_single(cost_data d_costs, int* d_M, int w, int h, int current_w, int n_elem){
    extern __shared__ int m_cache[];
    int tid = threadIdx.x*n_elem;
    int i, row, coloumn, ix;
    int left, up, right;
    
    //first row
    for(i = 0; i < n_elem && tid + i < current_w; i++){
        coloumn = tid + i;
        left = min(d_costs.left[coloumn], min(d_costs.up[coloumn], d_costs.right[coloumn]));
        d_M[coloumn] = left; 
        m_cache[coloumn] = left;
    }
    
    __syncthreads(); 
    
    int shift = 0;
    
    //other rows
    for(row = 1; row < h; row++){
        #pragma unroll
        for(i = 0; i < n_elem && tid + i < current_w; i++){
            coloumn = tid + i;
            ix = row*w + coloumn;
            
            //with left
            if(coloumn > 0){
                left = m_cache[coloumn - 1 + shift] + d_costs.left[ix]; 
            }
            else
                left = INT_MAX;
            //with up
            up = m_cache[coloumn + shift] + d_costs.up[ix];
            //with right
            if(coloumn < current_w-1){
                right = m_cache[coloumn + 1 + shift] + d_costs.right[ix];
            }
            else
                right = INT_MAX;
  
            left = min(left, min(up, right));
            d_M[ix] = left;
            m_cache[coloumn + (current_w - shift)] = left;
        }    
        //swap read/write shared memory
        shift = current_w - shift;
        __syncthreads();
    }        
}

#else
#ifdef COMPUTE_M_ITERATE

// UNUSED --- compute M one row at a time with multiple kernel calls for global synchronization
const int WIDEBLOCKSIZE = 128;

__global__ void compute_M_kernel_iterate0(cost_data d_costs, int* d_M, int w, int current_w){
    int coloumn = blockIdx.x*WIDEBLOCKSIZE + threadIdx.x; 
    
    if(coloumn < current_w){
        d_M[coloumn] = min(d_costs.left[coloumn], min(d_costs.up[coloumn], d_costs.right[coloumn]));
    }
    
}

__global__ void compute_M_kernel_iterate1(cost_data d_costs, int* d_M, int w, int current_w, int row){
    int coloumn = blockIdx.x*WIDEBLOCKSIZE + threadIdx.x; 
    int ix = row*w + coloumn;
    int prev_ix = ix - w;
    int left, up, right;
    
    if(coloumn < current_w){
        if(coloumn > 0)
            left = d_M[prev_ix - 1] + d_costs.left[ix]; 
        else
            left = INT_MAX;
            
        //with up
        up = d_M[prev_ix] + d_costs.up[ix];
        
        //with right
        if(coloumn < current_w-1)
            right = d_M[prev_ix + 1] + d_costs.right[ix];
        else
            right = INT_MAX;
                       
        d_M[ix] = min(left, min(up, right));  
   } 
}


#endif

#endif

const int REDUCEBLOCKSIZE = 128;
const int REDUCE_ELEMENTS_PER_THREAD = 8;

__global__ void min_reduce(int* d_values, int* d_indices, int N){
    __shared__ int val_cache[REDUCEBLOCKSIZE];
    __shared__ int ix_cache[REDUCEBLOCKSIZE];
    int tid = threadIdx.x;
    int coloumn = blockIdx.x*REDUCEBLOCKSIZE*REDUCE_ELEMENTS_PER_THREAD + REDUCE_ELEMENTS_PER_THREAD*threadIdx.x; 
    int min_v = INT_MAX;
    int min_i = 0;
    int new_i, new_v;
    int i;
    for(i = 0; i < REDUCE_ELEMENTS_PER_THREAD && coloumn + i < N; i++){
            new_i = d_indices[coloumn + i];
            new_v  = d_values[new_i];
            if(new_v < min_v){
                min_i = new_i;
                min_v = new_v;
            }             
    }
    val_cache[tid] = min_v;
    ix_cache[tid] = min_i;
    
    __syncthreads();
    
    for(i = REDUCEBLOCKSIZE/2; i > 0; i = i/2){
        if(tid < i){
            if(val_cache[tid + i] < val_cache[tid] || (val_cache[tid + i] == val_cache[tid] && ix_cache[tid + i] < ix_cache[tid])){
                val_cache[tid] = val_cache[tid + i];
                ix_cache[tid] = ix_cache[tid + i];
            }
        }
        __syncthreads();
    }
    
    if(tid == 0)
        d_indices[blockIdx.x] = ix_cache[0];    
}

__global__ void find_seam_kernel(int *d_M, int *d_indices, int *d_seam, int w, int h, int current_w){    
    int row, mid;
    int min_index = d_indices[0];
    
    d_seam[h-1] = min_index; 
    for(row = h-2; row >= 0; row--){
        mid = min_index;
        if(mid != 0){
            if(d_M[row*w + mid - 1] < d_M[row*w + min_index])
                min_index = mid - 1;
        }
        if(mid != current_w){
            if(d_M[row*w + mid + 1] < d_M[row*w + min_index])
                min_index = mid + 1;
        }
        d_seam[row] = min_index;
    }
}

__global__ void remove_seam_kernel(uint32_t *d_pixels, uint32_t *d_pixels_tmp, int *d_seam, int w, int h, int current_w){
    int row = blockIdx.y*BLOCKSIZE_Y + threadIdx.y;
    int coloumn = blockIdx.x*BLOCKSIZE_X + threadIdx.x;
    int seam_c = d_seam[row];
    int ix = row*w + coloumn;
    uint32_t pix;

    if(row < h && coloumn < current_w-1){
        if(coloumn >= seam_c)
            pix = d_pixels[ix + 1];
        else
            pix = d_pixels[ix];
            
        d_pixels_tmp[ix] = pix;
    }
}

__global__ void update_costs_kernel(uint32_t *d_pixels, cost_data d_costs, cost_data d_costs_tmp, int *d_seam, int w, int h, int current_w){
    int row = blockIdx.y*BLOCKSIZE_Y + threadIdx.y;
    int coloumn = blockIdx.x*BLOCKSIZE_X + threadIdx.x;
    int seam_c = d_seam[row];
    int ix = row*w + coloumn;
    
    if(row < h && coloumn < current_w-1){
        if(coloumn >= seam_c-2 && coloumn <= seam_c+1){
            //update costs near removed seam
            pixel pix1, pix2, pix3;
            int p_r, p_g, p_b;
            int rdiff, gdiff, bdiff;          
            
            if(coloumn == current_w-2) 
                pix1 = BORDER_PIXEL;
            else
                pix1 = pixel_from_int(d_pixels[ix + 1]);
            if(coloumn == 0)
                pix2 = BORDER_PIXEL;
            else
                pix2 = pixel_from_int(d_pixels[ix - 1]);
            if(row == 0)
                pix3 = BORDER_PIXEL;
            else
                pix3 = pixel_from_int(d_pixels[ix - w]);
                
            //compute partials
            p_r = abs(pix1.r - pix2.r);
            p_g = abs(pix1.g - pix2.g);
            p_b = abs(pix1.b - pix2.b);
            
            //compute left cost       
            rdiff = p_r + abs(pix3.r - pix2.r);
            gdiff = p_g + abs(pix3.g - pix2.g);
            bdiff = p_b + abs(pix3.b - pix2.b);
            d_costs_tmp.left[ix] = rdiff + gdiff + bdiff;
            
            //compute up cost
            d_costs_tmp.up[ix] = p_r + p_g + p_b;
            
            //compute right cost
            rdiff = p_r + abs(pix3.r - pix1.r);
            gdiff = p_g + abs(pix3.g - pix1.g);
            bdiff = p_b + abs(pix3.b - pix1.b);
            d_costs_tmp.right[ix] = rdiff + gdiff + bdiff;             
        }
        else if(coloumn > seam_c+1){
            //shift costs to the left
            d_costs_tmp.left[ix] = d_costs.left[ix + 1];
            d_costs_tmp.up[ix] = d_costs.up[ix + 1];
            d_costs_tmp.right[ix] = d_costs.right[ix + 1];
        }
        //else if(coloumn < seam_c-2){
        else{
            //copy remaining costs
            d_costs_tmp.left[ix] = d_costs.left[ix];
            d_costs_tmp.up[ix] = d_costs.up[ix];
            d_costs_tmp.right[ix] = d_costs.right[ix];
        }
    }
}


/* ############### wrappers #################### */

extern "C"{

int next_pow2(int n){
    int res = 1;
    while(res < n)
        res = res*2;
    return res;
}

#ifndef COMPUTE_COSTS_FULL

void compute_costs(uint32_t *d_pixels, cost_data d_costs, int w, int h, int current_w){
    dim3 threads_per_block(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 num_blocks;
    num_blocks.x = (int)((current_w-1)/(threads_per_block.x-2)) + 1;
    num_blocks.y = (int)((h-1)/(threads_per_block.y-1)) + 1;    
    compute_costs_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_costs, w, h, current_w);
}

#else

void compute_costs(uint32_t *d_pixels, cost_data d_costs, int w, int h, int current_w){
    dim3 threads_per_block(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 num_blocks;
    num_blocks.x = (int)((current_w-1)/(threads_per_block.x)) + 1;
    num_blocks.y = (int)((h-1)/(threads_per_block.y)) + 1;    
    compute_costs_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_costs, w, h, current_w);
}

#endif

#if !defined(COMPUTE_M_SINGLE) && !defined(COMPUTE_M_ITERATE)

void compute_M(cost_data d_costs, int *d_M, int w, int h, int current_w){   
    if(current_w <= 1024){
        dim3 threads_per_block(current_w, 1);   
        dim3 num_blocks(1,1);
        compute_M_kernel_small<<<num_blocks, threads_per_block, 2*current_w*sizeof(int)>>>(d_costs, d_M, w, h, current_w);
    }
    else{
        dim3 threads_per_block(WIDEBLOCKSIZE, 1);
        
        dim3 num_blocks;
        num_blocks.x = (int)((current_w-1)/(threads_per_block.x)) + 1;
        num_blocks.y = 1;
        
        dim3 num_blocks2;
        num_blocks2.x = (int)((current_w-WIDEBLOCKSIZE-1)/(threads_per_block.x)) + 1; 
        num_blocks2.y = 1;  
        
        int num_iterations;
        num_iterations = (int)((h-1)/(WIDEBLOCKSIZE/2 - 1)) + 1;
            
        int i;
        int base_row = 0;
        for(i = 0; i < num_iterations; i++){
            compute_M_kernel_step1<<<num_blocks, threads_per_block>>>(d_costs, d_M, w, h, current_w, base_row);
            compute_M_kernel_step2<<<num_blocks2, threads_per_block>>>(d_costs, d_M, w, h, current_w, base_row);
            base_row = base_row + (WIDEBLOCKSIZE/2) - 1;    
        }
    }
}

#endif

#ifdef COMPUTE_M_SINGLE

//compute M in a single block kernel
void compute_M(cost_data d_costs, int *d_M, int w, int h, int current_w){
    dim3 threads_per_block(min(1024, next_pow2(current_w)), 1);   
    dim3 num_blocks(1,1);
    int num_el = (int)((current_w-1)/threads_per_block.x) + 1;
    compute_M_kernel_single<<<num_blocks, threads_per_block, 2*current_w*sizeof(int)>>>(d_costs, d_M, w, h, current_w, num_el);
}

#else
#ifdef COMPUTE_M_ITERATE

void compute_M(cost_data d_costs, int *d_M, int w, int h, int current_w){
    dim3 threads_per_block(WIDEBLOCKSIZE, 1);   
    dim3 num_blocks;
    num_blocks.x = (int)((current_w-1)/threads_per_block.x) + 1;
    num_blocks.y = 1;
    compute_M_kernel_iterate0<<<num_blocks, threads_per_block>>>(d_costs, d_M, w, current_w);
    for(int row = 1; row < h; row++){
        compute_M_kernel_iterate1<<<num_blocks, threads_per_block>>>(d_costs, d_M, w, current_w, row);
    }
}

#endif
#endif

void find_min(int *d_M, int *d_indices, int *d_indices_ref, int w, int h, int current_w){
    //set the reference index array
    cudaMemcpy(d_indices, d_indices_ref, current_w*sizeof(int), cudaMemcpyDeviceToDevice);
    
    dim3 threads_per_block(REDUCEBLOCKSIZE, 1);   

    dim3 num_blocks;
    num_blocks.y = 1; 
    int reduce_num_elements = current_w;
    int *last_M_row = &(d_M[w*(h-1)]);
    do{
        num_blocks.x = (int)((reduce_num_elements-1)/(threads_per_block.x*REDUCE_ELEMENTS_PER_THREAD)) + 1;
        min_reduce<<<num_blocks, threads_per_block>>>(last_M_row, d_indices, reduce_num_elements); 
        reduce_num_elements = num_blocks.x;          
    }while(num_blocks.x > 1);    
}

void find_seam(int* d_M, int *d_indices, int *d_seam, int w, int h, int current_w){
    find_seam_kernel<<<1, 1>>>(d_M, d_indices, d_seam, w, h, current_w);
}

void remove_seam(uint32_t *d_pixels, uint32_t *d_pixels_tmp, int *d_seam, int w, int h, int current_w){
    dim3 threads_per_block(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 num_blocks;
    num_blocks.x = (int)((current_w-1)/(threads_per_block.x)) + 1;
    num_blocks.y = (int)((h-1)/(threads_per_block.y)) + 1;    
    remove_seam_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_pixels_tmp, d_seam, w, h, current_w);
}

//UNUSED
void update_costs(uint32_t *d_pixels, cost_data d_costs, cost_data d_costs_tmp, int *d_seam, int w, int h, int current_w){
    dim3 threads_per_block(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 num_blocks;
    num_blocks.x = (int)((current_w-1)/(threads_per_block.x)) + 1;
    num_blocks.y = (int)((h-1)/(threads_per_block.y)) + 1;    
    update_costs_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_costs, d_costs_tmp, d_seam, w, h, current_w);
}


}
