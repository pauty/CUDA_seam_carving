//#include <cuda_runtime.h>

extern "C"{

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>

#include "cuda_kernels.h"
#include "image.h"
#include "cost_data.h"

}

//#define COMPUTE_COSTS_FULL

//#define COMPUTE_M_SINGLE
//#define COMPUTE_M_ITERATE

const int COSTS_BLOCKSIZE_X = 32;
const int COSTS_BLOCKSIZE_Y = 8;

const int COMPUTE_M_BLOCKSIZE_X = 128; //must be divisible by 2

const int REDUCE_BLOCKSIZE_X = 128;
const int REDUCE_ELEMENTS_PER_THREAD = 8;

const int REMOVE_BLOCKSIZE_X = 32;
const int REMOVE_BLOCKSIZE_Y = 8;

const int UPDATE_BLOCKSIZE_X = 32;
const int UPDATE_BLOCKSIZE_Y = 8;

const int APPROX_SETUP_BLOCKSIZE_X = 32;
const int APPROX_SETUP_BLOCKSIZE_Y = 8;

const int APPROX_M_BLOCKSIZE_X = 128;


__constant__ pixel BORDER_PIXEL = {.r = 0, .g = 0, .b = 0};


__device__ pixel pixel_from_uchar4(uchar4 uc4){
    pixel pix;
    pix.r = (int)uc4.x;
    pix.g = (int)uc4.y;
    pix.b = (int)uc4.z;
    return pix;
}   

__device__ void pointer_swap(void **p1, void **p2){
    void *tmp;
    tmp = *p1;
    *p1 = *p2;
    *p2 = tmp; 
}

__global__ void compute_costs_kernel(uchar4 *d_pixels, cost_data d_costs, int w, int h, int current_w){
    //first row, first coloumn and last coloumn of shared memory are reserved for halo...
    __shared__ pixel pix_cache[COSTS_BLOCKSIZE_Y][COSTS_BLOCKSIZE_X];
    //...and the global index in the image is computed accordingly to this 
    int row = blockIdx.y*(COSTS_BLOCKSIZE_Y-1) + threadIdx.y -1 ; 
    int coloumn = blockIdx.x*(COSTS_BLOCKSIZE_X-2) + threadIdx.x -1; 
    int ix = row*w + coloumn;
    int cache_row = threadIdx.y;
    int cache_coloumn = threadIdx.x;
    short active = 0;
     
    if(row >= 0 && row < h && coloumn >= 0 && coloumn < current_w){
        active = 1;
        pix_cache[cache_row][cache_coloumn] = pixel_from_uchar4(d_pixels[ix]);
    }
    
    //wait until each thread has initialized its portion of shared memory
    __syncthreads();
    
    //all the threads that are NOT in halo positions can now compute costs, with fast access to shared memory
    if(active && cache_row != 0 && cache_coloumn != 0 && cache_coloumn != COSTS_BLOCKSIZE_X-1){
        int rdiff, gdiff, bdiff;
        int p_r, p_g, p_b;
        pixel pix1, pix2, pix3;
        
        if(coloumn < current_w-1)
            pix1 = pix_cache[cache_row][cache_coloumn+1];
        else 
            pix1 = BORDER_PIXEL;
        if(coloumn > 0)
            pix2 = pix_cache[cache_row][cache_coloumn-1];
        else
            pix2 = BORDER_PIXEL;
        if(row > 0)
            pix3 = pix_cache[cache_row-1][cache_coloumn];
        else
            pix3 = BORDER_PIXEL;
        
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

__global__ void compute_costs_full_kernel(uchar4* d_pixels, cost_data d_costs, int w, int h, int current_w){
    __shared__ pixel pix_cache[COSTS_BLOCKSIZE_Y+1][COSTS_BLOCKSIZE_X+2];
    int row = blockIdx.y*COSTS_BLOCKSIZE_Y + threadIdx.y; 
    int coloumn = blockIdx.x*COSTS_BLOCKSIZE_X + threadIdx.x; 
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
                pix_cache[cache_row][0] = pixel_from_uchar4(d_pixels[ix-1]);
        }
        if(threadIdx.x == COSTS_BLOCKSIZE_X-1 || coloumn == current_w-1){
            if(coloumn == current_w-1)
                pix_cache[cache_row][cache_coloumn+1] = BORDER_PIXEL;
            else
                pix_cache[cache_row][COSTS_BLOCKSIZE_X+1] = pixel_from_uchar4(d_pixels[ix+1]);
        }
        if(threadIdx.y == 0){
            if(row == 0)
                pix_cache[0][cache_coloumn] = BORDER_PIXEL;  
            else
                pix_cache[0][cache_coloumn] = pixel_from_uchar4(d_pixels[ix-w]);          
        } 
        pix_cache[cache_row][cache_coloumn] = pixel_from_uchar4(d_pixels[ix]);  
    }
    
    __syncthreads();
    
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

__global__ void compute_M_kernel_step1(cost_data d_costs, int* d_M, int w, int h, int current_w, int base_row){
    __shared__ int cache[2*COMPUTE_M_BLOCKSIZE_X];
    int *m_cache = cache;
    int *m_cache_swap = &(cache[COMPUTE_M_BLOCKSIZE_X]);
    int row;
    int coloumn = blockIdx.x*COMPUTE_M_BLOCKSIZE_X + threadIdx.x; 
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
    __syncthreads();
    
    int ix;
    int inc = 0;
    for(row = base_row+1; row < base_row + COMPUTE_M_BLOCKSIZE_X/2 && row < h; row++){
        inc++;
        if((is_first || inc - 1 < threadIdx.x) && (is_last || threadIdx.x < COMPUTE_M_BLOCKSIZE_X - inc) && coloumn < current_w){
            ix = row*w + coloumn;
            
            //with left
            if(coloumn > 0)
                left = m_cache[cache_coloumn - 1] + d_costs.left[ix]; 
            else 
                left = INT_MAX;
            //with up
            up = m_cache[cache_coloumn] + d_costs.up[ix];
            //with right
            if(coloumn < current_w-1)
                right = m_cache[cache_coloumn + 1] + d_costs.right[ix];
            else
                right = INT_MAX;
                
            left = min(left, min(up, right));           
            d_M[ix] = left;
            //swap read/write shared memory
            pointer_swap((void**)&m_cache, (void**)&m_cache_swap);
            m_cache[cache_coloumn] = left;
        }   
        //wait until every thread has written shared memory
        __syncthreads();                
    }
}


__global__ void compute_M_kernel_step2(cost_data d_costs, int* d_M, int w, int h, int current_w, int base_row){
    //__shared__ int m_cache[COMPUTE_M_BLOCKSIZE_X];
    int row;
    int coloumn = blockIdx.x*COMPUTE_M_BLOCKSIZE_X + threadIdx.x + COMPUTE_M_BLOCKSIZE_X/2; 
    // cache_coloumn = threadIdx.x; 
    int right, up, left;
    right = INT_MAX;
   
    //if(coloumn < current_w)
    //   m_cache[cache_coloumn] = d_costs[base_row*w + coloumn];
        
    //wait until shared memory load is complete
    //__syncthreads();
    
    int ix, prev_ix;
    int inc = 0;
    for(row = base_row+1; row < base_row + COMPUTE_M_BLOCKSIZE_X/2 && row < h; row++){
        inc++;
        if((COMPUTE_M_BLOCKSIZE_X/2 - inc -1 < threadIdx.x) && (threadIdx.x < COMPUTE_M_BLOCKSIZE_X/2 + inc) && coloumn < current_w){
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
    extern __shared__ int cache[];
    int *m_cache = cache;
    int *m_cache_swap = &(cache[current_w]);
    int coloumn = threadIdx.x;
    int row, ix;
    int left, up, right;
    
    //first row
    left = min(d_costs.left[coloumn], min(d_costs.up[coloumn], d_costs.right[coloumn]));
    d_M[coloumn] = left; 
    m_cache[coloumn] = left;
    
    __syncthreads(); 
  
    //other rows
    for(row = 1; row < h; row++){
        if(coloumn < current_w){
            ix = row*w + coloumn;    
            //with left
            if(coloumn > 0)
                left = m_cache[coloumn - 1] + d_costs.left[ix]; 
            else
                left = INT_MAX;
            //with up
            up = m_cache[coloumn] + d_costs.up[ix];
            //with right
            if(coloumn < current_w-1)
                right = m_cache[coloumn + 1] + d_costs.right[ix];
            else
                right = INT_MAX;

            left = min(left, min(up, right));            
            d_M[ix] = left;
            //swap read/write shared memory
            pointer_swap((void**)&m_cache, (void**)&m_cache_swap); 
            m_cache[coloumn] = left;
        }
        __syncthreads();    
    }     
}

__global__ void compute_M_kernel_single(cost_data d_costs, int* d_M, int w, int h, int current_w, int n_elem){
    extern __shared__ int cache[];
    int *m_cache = cache;
    int *m_cache_swap = &(cache[current_w]);
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
    
    //other rows
    for(row = 1; row < h; row++){
        #pragma unroll
        for(i = 0; i < n_elem && tid + i < current_w; i++){
            coloumn = tid + i;
            ix = row*w + coloumn;
            
            //with left
            if(coloumn > 0){
                left = m_cache[coloumn - 1] + d_costs.left[ix]; 
            }
            else
                left = INT_MAX;
            //with up
            up = m_cache[coloumn] + d_costs.up[ix];
            //with right
            if(coloumn < current_w-1){
                right = m_cache[coloumn + 1] + d_costs.right[ix];
            }
            else
                right = INT_MAX;
  
            left = min(left, min(up, right));
            d_M[ix] = left;
            m_cache_swap[coloumn] = left;
        }    
        //swap read/write shared memory
        pointer_swap((void**)&m_cache, (void**)&m_cache_swap);
        __syncthreads();
    }        
}

//compute M one row at a time with multiple kernel calls for global synchronization
__global__ void compute_M_kernel_iterate0(cost_data d_costs, int* d_M, int w, int current_w){
    int coloumn = blockIdx.x*COMPUTE_M_BLOCKSIZE_X + threadIdx.x; 
    
    if(coloumn < current_w){
        d_M[coloumn] = min(d_costs.left[coloumn], min(d_costs.up[coloumn], d_costs.right[coloumn]));
    }
    
}

__global__ void compute_M_kernel_iterate1(cost_data d_costs, int* d_M, int w, int current_w, int row){
    int coloumn = blockIdx.x*COMPUTE_M_BLOCKSIZE_X + threadIdx.x; 
    int ix = row*w + coloumn;
    int prev_ix = ix - w;
    int left, up, right;
    
    if(coloumn < current_w){
        //with left
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

__global__ void min_reduce(int* d_values, int* d_indices, int size){
    __shared__ int val_cache[REDUCE_BLOCKSIZE_X];
    __shared__ int ix_cache[REDUCE_BLOCKSIZE_X];
    int tid = threadIdx.x;
    int coloumn = blockIdx.x*REDUCE_BLOCKSIZE_X*REDUCE_ELEMENTS_PER_THREAD + REDUCE_ELEMENTS_PER_THREAD*threadIdx.x; 
    int min_v = INT_MAX;
    int min_i = 0;
    int new_i, new_v;
    
    for(int i = 0; i < REDUCE_ELEMENTS_PER_THREAD && coloumn + i < size; i++){
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
    
    for(int i = REDUCE_BLOCKSIZE_X/2; i > 0; i = i/2){
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

__global__ void remove_seam_kernel(uchar4 *d_pixels, uchar4 *d_pixels_swap, int *d_seam, int w, int h, int current_w){
    int row = blockIdx.y*REMOVE_BLOCKSIZE_Y + threadIdx.y;
    int coloumn = blockIdx.x*REMOVE_BLOCKSIZE_X + threadIdx.x;
    int seam_c = d_seam[row];
    int ix = row*w + coloumn;
    uchar4 pix;

    if(row < h && coloumn < current_w-1){
        if(coloumn >= seam_c)
            pix = d_pixels[ix + 1];
        else
            pix = d_pixels[ix];
            
        d_pixels_swap[ix] = pix;
    }
}

__global__ void update_costs_kernel(uchar4 *d_pixels, cost_data d_costs, cost_data d_costs_swap, int *d_seam, int w, int h, int current_w){
    int row = blockIdx.y*UPDATE_BLOCKSIZE_Y + threadIdx.y;
    int coloumn = blockIdx.x*UPDATE_BLOCKSIZE_X + threadIdx.x;
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
                pix1 = pixel_from_uchar4(d_pixels[ix + 1]);
            if(coloumn == 0)
                pix2 = BORDER_PIXEL;
            else
                pix2 = pixel_from_uchar4(d_pixels[ix - 1]);
            if(row == 0)
                pix3 = BORDER_PIXEL;
            else
                pix3 = pixel_from_uchar4(d_pixels[ix - w]);
                
            //compute partials
            p_r = abs(pix1.r - pix2.r);
            p_g = abs(pix1.g - pix2.g);
            p_b = abs(pix1.b - pix2.b);
            
            //compute left cost       
            rdiff = p_r + abs(pix3.r - pix2.r);
            gdiff = p_g + abs(pix3.g - pix2.g);
            bdiff = p_b + abs(pix3.b - pix2.b);
            d_costs_swap.left[ix] = rdiff + gdiff + bdiff;
            
            //compute up cost
            d_costs_swap.up[ix] = p_r + p_g + p_b;
            
            //compute right cost
            rdiff = p_r + abs(pix3.r - pix1.r);
            gdiff = p_g + abs(pix3.g - pix1.g);
            bdiff = p_b + abs(pix3.b - pix1.b);
            d_costs_swap.right[ix] = rdiff + gdiff + bdiff;             
        }
        else if(coloumn > seam_c+1){
            //shift costs to the left
            d_costs_swap.left[ix] = d_costs.left[ix + 1];
            d_costs_swap.up[ix] = d_costs.up[ix + 1];
            d_costs_swap.right[ix] = d_costs.right[ix + 1];
        }
        else{
            //copy remaining costs
            d_costs_swap.left[ix] = d_costs.left[ix];
            d_costs_swap.up[ix] = d_costs.up[ix];
            d_costs_swap.right[ix] = d_costs.right[ix];
        }
    }
}


__global__ void approx_setup_kernel(cost_data d_costs, int *d_index_map, int *d_offset_map, int *d_M, int w, int h, int current_w){
    int row = blockIdx.y*APPROX_SETUP_BLOCKSIZE_Y + threadIdx.y;
    int coloumn = blockIdx.x*APPROX_SETUP_BLOCKSIZE_X + threadIdx.x;
    int ix = row*w + coloumn;
    
    if(row < h-1 && coloumn < current_w){
        int min_cost = INT_MAX;
        int map_ix;
        int cost, ixx;
       
        if(coloumn > 0){
            ixx = ix + w - 1;
            min_cost = d_costs.right[ixx];
            map_ix = ixx;
        }
        
        ixx = ix + w;
        cost = d_costs.up[ixx];
        if(cost < min_cost){
            min_cost = cost;
            map_ix = ixx;
        }
        
        if(coloumn < current_w-1){
            ixx = ix + w + 1;
            cost = d_costs.left[ixx];
            if(cost < min_cost){
                min_cost = cost;
                map_ix = ixx;
            }
        }
        
        d_index_map[ix] = map_ix;
        d_offset_map[ix] = map_ix;
        d_M[ix] = min_cost;           
    }
}

__global__ void approx_M_kernel(int *d_offset_map, int *d_M, int w, int h, int current_w, int step){
        int row = blockIdx.y*2*step;
        int next_row = row + step;
        int coloumn = blockIdx.x*APPROX_M_BLOCKSIZE_X + threadIdx.x;
        int ix = row*w + coloumn;
        //int next_ix = next_row*w + coloumn;
        
        if(next_row < h-1 && coloumn < current_w){
            int offset;
            offset = d_offset_map[ix];
            d_M[ix] = d_M[ix] + d_M[offset];
            d_offset_map[ix] = d_offset_map[offset];
            
            //if(row == h-2 || next_row == h-2)
             //   printf("ok");
        }
}

__global__ void approx_seam_kernel(int *d_index_map, int *d_indices, int *d_seam, int w, int h){
    int ix;
    ix = d_indices[0]; //min index
    for(int i = 0; i < h; i++){
            d_seam[i] = ix - i*w;
            ix = d_index_map[ix];
    }
}

/*################################################*/









/* ############### wrappers #################### */

extern "C"{

int next_pow2(int n){
    int res = 1;
    while(res < n)
        res = res*2;
    return res;
}

void compute_costs(seam_carver sc){
    #ifndef COMPUTE_COSTS_FULL
    
    dim3 threads_per_block(COSTS_BLOCKSIZE_X, COSTS_BLOCKSIZE_Y);
    dim3 num_blocks;
    num_blocks.x = (int)((sc.current_w-1)/(threads_per_block.x-2)) + 1;
    num_blocks.y = (int)((sc.h-1)/(threads_per_block.y-1)) + 1;    
    compute_costs_kernel<<<num_blocks, threads_per_block>>>(sc.d_pixels, sc.d_costs, sc.w, sc.h, sc.current_w);
    
    #else
    
    dim3 threads_per_block(COSTS_BLOCKSIZE_X, COSTS_BLOCKSIZE_Y);
    dim3 num_blocks;
    num_blocks.x = (int)((sc.current_w-1)/(threads_per_block.x)) + 1;
    num_blocks.y = (int)((sc.h-1)/(threads_per_block.y)) + 1;    
    compute_costs_full_kernel<<<num_blocks, threads_per_block>>>(sc.d_pixels, sc.d_costs, sc.w, sc.h, sc.current_w);
    
    #endif
}

void compute_M(seam_carver sc){ 
    #if !defined(COMPUTE_M_SINGLE) && !defined(COMPUTE_M_ITERATE)  
    
    if(sc.current_w <= 1024){
        dim3 threads_per_block(sc.current_w, 1);   
        dim3 num_blocks(1,1);
        compute_M_kernel_small<<<num_blocks, threads_per_block, 2*sc.current_w*sizeof(int)>>>(sc.d_costs, sc.d_M, sc.w, sc.h, sc.current_w);
    }
    else{
        dim3 threads_per_block(COMPUTE_M_BLOCKSIZE_X, 1);
        
        dim3 num_blocks;
        num_blocks.x = (int)((sc.current_w-1)/(threads_per_block.x)) + 1;
        num_blocks.y = 1;
        
        dim3 num_blocks2;
        num_blocks2.x = (int)((sc.current_w-COMPUTE_M_BLOCKSIZE_X-1)/(threads_per_block.x)) + 1; 
        num_blocks2.y = 1;  
        
        int num_iterations;
        num_iterations = (int)((sc.h-1)/(COMPUTE_M_BLOCKSIZE_X/2 - 1)) + 1;
            
        int i;
        int base_row = 0;
        for(i = 0; i < num_iterations; i++){
            compute_M_kernel_step1<<<num_blocks, threads_per_block>>>(sc.d_costs, sc.d_M, sc.w, sc.h, sc.current_w, base_row);
            compute_M_kernel_step2<<<num_blocks2, threads_per_block>>>(sc.d_costs, sc.d_M, sc.w, sc.h, sc.current_w, base_row);
            base_row = base_row + (COMPUTE_M_BLOCKSIZE_X/2) - 1;    
        }
    }
    
    #endif
    #ifdef COMPUTE_M_SINGLE    
    
    dim3 threads_per_block(min(1024, next_pow2(sc.current_w)), 1);   
    dim3 num_blocks(1,1);
    int num_el = (int)((sc.current_w-1)/threads_per_block.x) + 1;
    compute_M_kernel_single<<<num_blocks, threads_per_block, 2*sc.current_w*sizeof(int)>>>(sc.d_costs, sc.d_M, sc.w, sc.h, sc.current_w, num_el);
    
    #else
    #ifdef COMPUTE_M_ITERATE
    
    dim3 threads_per_block(COMPUTE_M_BLOCKSIZE_X, 1);   
    dim3 num_blocks;
    num_blocks.x = (int)((sc.current_w-1)/threads_per_block.x) + 1;
    num_blocks.y = 1;
    compute_M_kernel_iterate0<<<num_blocks, threads_per_block>>>(sc.d_costs, sc.d_M, sc.w, sc.current_w);
    for(int row = 1; row < h; row++){
        compute_M_kernel_iterate1<<<num_blocks, threads_per_block>>>(sc.d_costs, sc.d_M, sc.w, sc.current_w, row);
    }
    
    #endif
    #endif
}

void find_min_index(seam_carver sc){
    //set the reference index array
    cudaMemcpy(sc.d_indices, sc.d_indices_ref, sc.current_w*sizeof(int), cudaMemcpyDeviceToDevice);
    
    dim3 threads_per_block(REDUCE_BLOCKSIZE_X, 1);   
    dim3 num_blocks;
    num_blocks.y = 1; 
    int reduce_num_elements = sc.current_w;
    do{
        num_blocks.x = (int)((reduce_num_elements-1)/(threads_per_block.x*REDUCE_ELEMENTS_PER_THREAD)) + 1;
        min_reduce<<<num_blocks, threads_per_block>>>(sc.reduce_row, sc.d_indices, reduce_num_elements); 
        reduce_num_elements = num_blocks.x;          
    }while(num_blocks.x > 1);    
}

void find_seam(seam_carver sc){
    find_seam_kernel<<<1, 1>>>(sc.d_M, sc.d_indices, sc.d_seam, sc.w, sc.h, sc.current_w);
}

void remove_seam(seam_carver sc){
    dim3 threads_per_block(REMOVE_BLOCKSIZE_X, REMOVE_BLOCKSIZE_Y);
    dim3 num_blocks;
    num_blocks.x = (int)((sc.current_w-1)/(threads_per_block.x)) + 1;
    num_blocks.y = (int)((sc.h-1)/(threads_per_block.y)) + 1;    
    remove_seam_kernel<<<num_blocks, threads_per_block>>>(sc.d_pixels, sc.d_pixels_swap, sc.d_seam, sc.w, sc.h, sc.current_w);
}

void update_costs(seam_carver sc){
    dim3 threads_per_block(UPDATE_BLOCKSIZE_X, UPDATE_BLOCKSIZE_Y);
    dim3 num_blocks;
    num_blocks.x = (int)((sc.current_w-1)/(threads_per_block.x)) + 1;
    num_blocks.y = (int)((sc.h-1)/(threads_per_block.y)) + 1;    
    update_costs_kernel<<<num_blocks, threads_per_block>>>(sc.d_pixels, sc.d_costs, sc.d_costs_swap, sc.d_seam, sc.w, sc.h, sc.current_w);
}

void approx_setup(seam_carver sc){
    dim3 threads_per_block(APPROX_SETUP_BLOCKSIZE_X, APPROX_SETUP_BLOCKSIZE_Y);
    dim3 num_blocks;
    num_blocks.x = (int)((sc.current_w-1)/(threads_per_block.x)) + 1;
    num_blocks.y = (int)((sc.h-2)/(threads_per_block.y)) + 1;    
    approx_setup_kernel<<<num_blocks, threads_per_block>>>(sc.d_costs, sc.d_index_map, sc.d_offset_map, sc.d_M, sc.w, sc.h, sc.current_w);
}

void approx_M(seam_carver sc){
    dim3 threads_per_block(APPROX_M_BLOCKSIZE_X, 1);
    dim3 num_blocks;
    num_blocks.x = (int)((sc.current_w-1)/(threads_per_block.x)) + 1;
    num_blocks.y = (int)((sc.h)/2);  
    int step = 1;
    while(num_blocks.y > 0){
        approx_M_kernel<<<num_blocks, threads_per_block>>>(sc.d_offset_map, sc.d_M, sc.w, sc.h, sc.current_w, step);
        num_blocks.y = (int)num_blocks.y/2;
        step = step*2;
    }
}

void approx_seam(seam_carver sc){
    approx_seam_kernel<<<1, 1>>>(sc.d_index_map, sc.d_indices, sc.d_seam, sc.w, sc.h);
}

}
