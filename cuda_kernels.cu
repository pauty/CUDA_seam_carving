//#include <cuda_runtime.h>

extern "C"{
#include <stdio.h>
#include <stdlib.h>
//#include <math.h>
#include <limits.h>
#include "image.h"
}

__constant__ pixel BORDER_PIXEL = {.r = 0, .g = 0, .b = 0, .a = 0};

//#define COMPUTE_M_SINGLE
//#define COMPUTE_COSTS_FULL

const int BLOCKSIZE = 16;

#ifndef COMPUTE_COSTS_FULL

__global__ void compute_costs_kernel(pixel* d_pixels, int* d_costs, int w, int h, int current_w){
    //first row, first coloumn and last coloumn of shared memory are reserved for halo...
    __shared__ pixel pix_cache[BLOCKSIZE][BLOCKSIZE];
    //...and the global index in the image is computed accordingly to this 
    int row = blockIdx.y*(BLOCKSIZE-1) + threadIdx.y -1 ; 
    int coloumn = blockIdx.x*(BLOCKSIZE-2) + threadIdx.x -1; 
    int ix = row*w + coloumn;
    int wh = w*h;
    int cache_r = threadIdx.y;
    int cache_c = threadIdx.x;
    short active = 0;
     
    if(row < h && coloumn <= current_w){
        //only threads with row in [-1,h-1] and coloumn in [-1,current_w] are actually active
        active = 1;
        //if access to the image is out of bounds, set RGB values to 0
        //otherwise load pixel from global memory
        if(row < 0 || coloumn < 0 || coloumn == current_w){
            pix_cache[cache_r][cache_c] = BORDER_PIXEL;
        }
        else
            pix_cache[cache_r][cache_c] = d_pixels[ix];
    }
    
    //wait until each thread has initialized its portion of shared memory
    __syncthreads();
    
    //all the threads that are NOT in halo positions can now compute costs, with fast access to shared memory
    if(active && cache_r != 0 && cache_c != 0 
        && cache_c != BLOCKSIZE-1 && coloumn < current_w){
        int rdiff, gdiff, bdiff;
        int p_r, p_g, p_b;
        
        p_r = abs(pix_cache[cache_r][cache_c+1].r - pix_cache[cache_r][cache_c-1].r);
        p_g = abs(pix_cache[cache_r][cache_c+1].g - pix_cache[cache_r][cache_c-1].g);
        p_b = abs(pix_cache[cache_r][cache_c+1].b - pix_cache[cache_r][cache_c-1].b);
        //calc left
        rdiff = p_r + abs(pix_cache[cache_r-1][cache_c].r - pix_cache[cache_r][cache_c-1].r);
        gdiff = p_g + abs(pix_cache[cache_r-1][cache_c].g - pix_cache[cache_r][cache_c-1].g);
        bdiff = p_b + abs(pix_cache[cache_r-1][cache_c].b - pix_cache[cache_r][cache_c-1].b);
        d_costs[ix] = rdiff + gdiff + bdiff;
        //calc up
        d_costs[ix + wh] = p_r + p_g + p_b;
         //calc right
        rdiff = p_r + abs(pix_cache[cache_r-1][cache_c].r - pix_cache[cache_r][cache_c+1].r);
        gdiff = p_g + abs(pix_cache[cache_r-1][cache_c].g - pix_cache[cache_r][cache_c+1].g);
        bdiff = p_b + abs(pix_cache[cache_r-1][cache_c].b - pix_cache[cache_r][cache_c+1].b);
        d_costs[ix + 2*wh] = rdiff + gdiff + bdiff;
        
        /*
        //calc left
        rdiff = abs(pix_cache[cache_r][cache_c+1].r - pix_cache[cache_r][cache_c-1].r) +
                abs(pix_cache[cache_r-1][cache_c].r - pix_cache[cache_r][cache_c-1].r);
        gdiff = abs(pix_cache[cache_r][cache_c+1].g - pix_cache[cache_r][cache_c-1].g) +
                abs(pix_cache[cache_r-1][cache_c].g - pix_cache[cache_r][cache_c-1].g);
        bdiff = abs(pix_cache[cache_r][cache_c+1].b - pix_cache[cache_r][cache_c-1].b) +
                abs(pix_cache[cache_r-1][cache_c].b - pix_cache[cache_r][cache_c-1].b);
        d_costs[ix] = rdiff + gdiff + bdiff;
        //calc up
        rdiff = abs(pix_cache[cache_r][cache_c+1].r - pix_cache[cache_r][cache_c-1].r);
        gdiff = abs(pix_cache[cache_r][cache_c+1].g - pix_cache[cache_r][cache_c-1].g);
        bdiff = abs(pix_cache[cache_r][cache_c+1].b - pix_cache[cache_r][cache_c-1].b);
        d_costs[ix + wh] = rdiff + gdiff + bdiff;
        //calc right
        rdiff = abs(pix_cache[cache_r][cache_c+1].r - pix_cache[cache_r][cache_c-1].r) +
                abs(pix_cache[cache_r-1][cache_c].r - pix_cache[cache_r][cache_c+1].r);
        gdiff = abs(pix_cache[cache_r][cache_c+1].g - pix_cache[cache_r][cache_c-1].g) +
                abs(pix_cache[cache_r-1][cache_c].g - pix_cache[cache_r][cache_c+1].g);
        bdiff = abs(pix_cache[cache_r][cache_c+1].b - pix_cache[cache_r][cache_c-1].b) +
                abs(pix_cache[cache_r-1][cache_c].b - pix_cache[cache_r][cache_c+1].b);
        d_costs[ix + 2*wh] = rdiff + gdiff + bdiff;
        */
    }
       
}

#else

__global__ void compute_costs_kernel(pixel* d_pixels, int* d_costs, int w, int h, int current_w){
    __shared__ pixel pix_cache[BLOCKSIZE+1][BLOCKSIZE+2];
    int row = blockIdx.y*BLOCKSIZE + threadIdx.y; 
    int coloumn = blockIdx.x*BLOCKSIZE + threadIdx.x; 
    int ix = row*w + coloumn;
    int wh = w*h;
    int cache_r = threadIdx.y + 1;
    int cache_c = threadIdx.x + 1;
    short active = 0;
     
    if(row < h && coloumn < current_w){
        active = 1;
        if(threadIdx.x == 0){
            if(coloumn == 0)
                pix_cache[cache_r][0] = BORDER_PIXEL;
            else
                pix_cache[cache_r][0] = d_pixels[ix-1];
        }
        if(threadIdx.x == BLOCKSIZE-1 || coloumn == current_w-1){
            if(coloumn == current_w-1)
                pix_cache[cache_r][cache_c+1] = BORDER_PIXEL;
            else
                pix_cache[cache_r][BLOCKSIZE+1] = d_pixels[ix+1];
        }
        if(threadIdx.y == 0){
            if(row == 0)
                pix_cache[0][cache_c] = BORDER_PIXEL;  
            else
                pix_cache[0][cache_c] = d_pixels[ix-w];            
        }
        
        pix_cache[cache_r][cache_c] = d_pixels[ix];
        
    }
    
    //wait until each thread has initialized its portion of shared memory
    __syncthreads();
    
    //all the threads that are NOT in halo positions can now compute costs, with fast access to shared memory
    if(active){
        int rdiff, gdiff, bdiff;
        int p_r, p_g, p_b;
        
        p_r = abs(pix_cache[cache_r][cache_c+1].r - pix_cache[cache_r][cache_c-1].r);
        p_g = abs(pix_cache[cache_r][cache_c+1].g - pix_cache[cache_r][cache_c-1].g);
        p_b = abs(pix_cache[cache_r][cache_c+1].b - pix_cache[cache_r][cache_c-1].b);
        //calc left
        rdiff = p_r + abs(pix_cache[cache_r-1][cache_c].r - pix_cache[cache_r][cache_c-1].r);
        gdiff = p_g + abs(pix_cache[cache_r-1][cache_c].g - pix_cache[cache_r][cache_c-1].g);
        bdiff = p_b + abs(pix_cache[cache_r-1][cache_c].b - pix_cache[cache_r][cache_c-1].b);
        d_costs[ix] = rdiff + gdiff + bdiff;
        //calc up
        d_costs[ix + wh] = p_r + p_g + p_b;
         //calc right
        rdiff = p_r + abs(pix_cache[cache_r-1][cache_c].r - pix_cache[cache_r][cache_c+1].r);
        gdiff = p_g + abs(pix_cache[cache_r-1][cache_c].g - pix_cache[cache_r][cache_c+1].g);
        bdiff = p_b + abs(pix_cache[cache_r-1][cache_c].b - pix_cache[cache_r][cache_c+1].b);
        d_costs[ix + 2*wh] = rdiff + gdiff + bdiff;
     }
}

#endif


#ifndef COMPUTE_M_SINGLE

const int WIDEBLOCKSIZE = 128; //must be divisible by 2

__global__ void compute_M_kernel_step1(int *d_costs, int* d_M, int w, int h, int current_w, int base_row){
    __shared__ int m_cache[WIDEBLOCKSIZE];
    int row;
    int coloumn = blockIdx.x*WIDEBLOCKSIZE + threadIdx.x; 
    int cache_coloumn = threadIdx.x; 
    int wh = w*h;
    short is_first;
    short is_last;
    int right, up, left;
    
    is_first = blockIdx.x == 0;
    is_last = blockIdx.x == gridDim.x-1;
    
    if(base_row == 0 && coloumn < current_w){
        left = min(d_costs[coloumn], min(d_costs[coloumn + wh], d_costs[coloumn + 2*wh]));
        m_cache[cache_coloumn] = left;
        d_M[coloumn] = left; 
    }
    else{
        m_cache[cache_coloumn] = d_M[base_row*w + coloumn];    
    }
    //wait until shared memory load is complete
    __syncthreads();
    
    int ix;
    int inc = 0;
    for(row = base_row+1; row < base_row + WIDEBLOCKSIZE/2 && row < h; row++){
        inc++;
        if((is_first || inc - 1 < threadIdx.x) && (is_last || threadIdx.x < WIDEBLOCKSIZE - inc) && coloumn < current_w){
            ix = row*w + coloumn;
            
            //with left
            if(coloumn > 0)
                left = m_cache[cache_coloumn - 1] + d_costs[ix]; 
            else 
                left = INT_MAX;
            //with up
            up = m_cache[cache_coloumn] + d_costs[ix + wh];
            //with right
            if(coloumn < current_w-1)
                right = m_cache[cache_coloumn + 1] + d_costs[ix + 2*wh];
            else
                right = INT_MAX;
                
            left = min(left, min(up, right));
                
            /* INEFFICIENT
            //up cost -- all thread can compute it
            up = m_cache[cache_coloumn] + d_costs[ix + wh];
            if(coloumn == 0){
                right = m_cache[cache_coloumn + 1] + d_costs[ix + 2*wh];
                left = min(up, right);
            }
            else if(coloumn == current_w-1){
                left = m_cache[cache_coloumn - 1] + d_costs[ix]; 
                left = min(left, up);
            }
            else{
                right = m_cache[cache_coloumn + 1] + d_costs[ix + 2*wh];
                left = m_cache[cache_coloumn - 1] + d_costs[ix]; 
                left = min(left, min(up, right));
            }*/
            
            d_M[ix] = left;

        }   
        //wait until every thread has read shared memory
        __syncthreads();
        m_cache[cache_coloumn] = left;
        //wait until every thread has written shared memory
        __syncthreads();          
    }
}


__global__ void compute_M_kernel_step2(int *d_costs, int* d_M, int w, int h, int current_w, int base_row){
    //__shared__ int m_cache[WIDEBLOCKSIZE];
    int row;
    int coloumn = blockIdx.x*WIDEBLOCKSIZE + threadIdx.x + WIDEBLOCKSIZE/2; 
    int wh = w*h;
    // cache_coloumn = threadIdx.x; 
    int right, up, left;
   
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
            left = d_M[prev_ix - 1] + d_costs[ix]; 
            //with up
            up = d_M[prev_ix] + d_costs[ix + wh];
            //with right
            if(coloumn < current_w-1){
                right = d_M[prev_ix + 1] + d_costs[ix + 2*wh];
            }
            else{
                right = INT_MAX;
            }
            left = min(left, min(up, right));
                      
            /*
            if(coloumn < current_w-1){
                right = d_M[prev_ix + 1] + d_costs[ix + 2*wh];
                left = min(left, min(up, right)); 
            }
            else{
                left = min(left, up);
            }
            */
                
            d_M[ix] = left;
        }
        __syncthreads();
    }
}

__global__ void compute_M_kernel_small(int *d_costs, int* d_M, int w, int h, int current_w){
    extern __shared__ int m_cache[];
    int coloumn = threadIdx.x;
    int row, ix;
    int wh = w*h;
    int left, up, right;
    
    //first row
    left = min(d_costs[coloumn], min(d_costs[coloumn + wh], d_costs[coloumn + 2*wh]));
    d_M[coloumn] = left; 
    m_cache[coloumn] = left;
    
    __syncthreads(); 
    
    //other rows
    for(row = 1; row < h; row++){
        if(coloumn < current_w){
            ix = row*w + coloumn;    
            //with left
            if(coloumn > 0){
                left = m_cache[coloumn - 1] + d_costs[ix]; 
            }
            else
                left = INT_MAX;
            //with up
            up = m_cache[coloumn] + d_costs[ix + wh];
            //with right
            if(coloumn < current_w-1){
                right = m_cache[coloumn + 1] + d_costs[ix + 2*wh];
            }
            else
                right = INT_MAX;

            left = min(left, min(up, right));
            
            
            /* INEFFICIENT 
            up = m_cache[coloumn] + d_costs[ix + wh];
            if(coloumn == 0){
                right = m_cache[coloumn + 1] + d_costs[ix + 2*wh];
                left = min(up, right);
            }
            else if(coloumn == current_w-1){
                left = m_cache[coloumn - 1] + d_costs[ix];  
                left = min(left, up);
            }
            else{
                right = m_cache[coloumn + 1] + d_costs[ix + 2*wh];
                left = m_cache[coloumn - 1] + d_costs[ix];  
                left = min(left, min(up, right));
            }*/
            
            d_M[ix] = left;
        }
        //everyone has read
        __syncthreads();
        if(coloumn < current_w)
            m_cache[coloumn] = left;
        //everyone has written
        __syncthreads();       
    }
        
}

#else

__global__ void compute_M_kernel_single(int *d_costs, int* d_M, int w, int h, int current_w, int n_elem){
    extern __shared__ int m_cache[];
    int tid = threadIdx.x*n_elem;
    int i, row, coloumn, ix;
    int wh = w*h;
    int left, up, right;
    
    //first row
    for(i = 0; i < n_elem && tid + i < current_w; i++){
        coloumn = tid + i;
        left = min(d_costs[coloumn], min(d_costs[coloumn + wh], d_costs[coloumn + 2*wh]));
        d_M[coloumn] = left; 
        m_cache[coloumn] = left;
    }
    
    __syncthreads(); 
    
    short shift = 0;
    
    //other rows
    for(row = 1; row < h; row++){
        for(i = 0; i < n_elem && tid + i < current_w; i++){
            coloumn = tid + i;
            ix = row*w + coloumn;
            
            //with left
            if(coloumn > 0){
                left = m_cache[coloumn - 1 + shift*current_w] + d_costs[ix]; 
            }
            else
                left = INT_MAX;
            //with up
            up = m_cache[coloumn + shift*current_w] + d_costs[ix + wh];
            //with right
            if(coloumn < current_w-1){
                right = m_cache[coloumn + 1 + shift*current_w] + d_costs[ix + 2*wh];
            }
            else
                right = INT_MAX;
  
            left = min(left, min(up, right));
                      
            /* INEFFICIENT 
            up = m_cache[coloumn] + d_costs[ix + wh];
            if(coloumn == 0){
                right = m_cache[coloumn + 1] + d_costs[ix + 2*wh];
                left = min(up, right);
            }
            else if(coloumn == current_w-1){
                left = m_cache[coloumn - 1] + d_costs[ix];  
                left = min(left, up);
            }
            else{
                right = m_cache[coloumn + 1] + d_costs[ix + 2*wh];
                left = m_cache[coloumn - 1] + d_costs[ix];  
                left = min(left, min(up, right));
            }*/
            
            d_M[ix] = left;
            m_cache[coloumn + (1-shift)*current_w] = left;
        }
        
        __syncthreads();
        //swap read/write shared memory
        shift = 1 - shift;
    }        
}

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

/* SEQUENTIAL MIN SEARCH
__global__ void find_min_kernel(int *d_M, int *d_indices, int w, int h, int current_w){
    int base_ix = w*(h-1);
    int min_i = 0;
    int i;
    for(i = 1; i < current_w; i++){
        if(d_M[base_ix + i] < d_M[base_ix + min_i])
            min_i = i;
    }
    d_indices[0] = min_i;
}
*/

__global__ void find_seam_kernel(int *d_M, int *d_indices, int *d_seam, int w, int h, int current_w){    
    int row, mid;
    int min_index = d_indices[0];
    
    d_seam[h-1] = min_index; 
    for(row = h-2; row >= 0; row--){
        mid = min_index;
        if(mid != 0){
            if(d_M[row*w + mid - 1] < d_M[row*w + min_index] )
                min_index = mid - 1;
        }
        if(mid != current_w){
            if(d_M[row*w + mid + 1] < d_M[row*w + min_index] )
                min_index = mid + 1;
        }
        d_seam[row] = min_index;
    }
}


__global__ void remove_seam_kernel_step1(pixel *d_pixels, pixel *d_pixels_tmp, int *d_seam, int w, int h, int current_w){
    int row = blockIdx.y*BLOCKSIZE + threadIdx.y;
    int coloumn = blockIdx.x*BLOCKSIZE + threadIdx.x;
    int seam_c = d_seam[row];
    int ix = row*w + coloumn;
    if(row < h && coloumn < current_w-1 && coloumn >= seam_c){
        d_pixels_tmp[ix] = d_pixels[ix + 1];
    }
}

__global__ void remove_seam_kernel_step2(pixel *d_pixels, pixel *d_pixels_tmp, int *d_seam, int w, int h, int current_w){
    int row = blockIdx.y*BLOCKSIZE + threadIdx.y;
    int coloumn = blockIdx.x*BLOCKSIZE + threadIdx.x;
    int seam_c = d_seam[row];
    int ix = row*w + coloumn;
    if(row < h && coloumn < current_w-1 && coloumn >= seam_c){
        d_pixels[ix] = d_pixels_tmp[ix];
    }
}

__global__ void update_costs_kernel_step1(pixel *pixels, int *d_costs, int *d_costs_tmp, int *d_seam, int w, int h, int current_w){
    int row = blockIdx.y*BLOCKSIZE + threadIdx.y;
    int coloumn = blockIdx.x*BLOCKSIZE + threadIdx.x;
    int seam_c = d_seam[row];
    int wh = w*h;
    int ix = row*w + coloumn;
    if(row < h && coloumn < current_w-1){
        if(coloumn >= seam_c-2 && coloumn <= seam_c+1){
            pixel pix1, pix2, pix3;
            int p_r, p_g, p_b;
            int rdiff, gdiff, bdiff;          
            
            if(coloumn == current_w-2) 
                pix1 = BORDER_PIXEL;
            else
                pix1 = pixels[ix + 1];
            if(coloumn == 0)
                pix2 = BORDER_PIXEL;
            else
                pix2 = pixels[ix - 1];
            if(row == 0)
                pix3 = BORDER_PIXEL;
            else
                pix3 = pixels[ix - w];
                
            //compute partials
            p_r = abs(pix1.r - pix2.r);
            p_g = abs(pix1.g - pix2.g);
            p_b = abs(pix1.b - pix2.b);
            
            //compute left cost       
            rdiff = p_r + abs(pix3.r - pix2.r);
            gdiff = p_g + abs(pix3.g - pix2.g);
            bdiff = p_b + abs(pix3.b - pix2.b);
            d_costs_tmp[ix] = rdiff + gdiff + bdiff;
            
            //compute up cost
            d_costs_tmp[ix + wh] = p_r + p_g + p_b;
            
            //compute right cost
            rdiff = p_r + abs(pix3.r - pix1.r);
            gdiff = p_g + abs(pix3.g - pix1.g);
            bdiff = p_b + abs(pix3.b - pix1.b);
            d_costs_tmp[ix + 2*wh] = rdiff + gdiff + bdiff;
            
        }
        else if(coloumn > seam_c+1){
            //shift costs to the left
            d_costs_tmp[ix] = d_costs[ix + 1];
            d_costs_tmp[ix + wh] = d_costs[ix + 1 + wh];
            d_costs_tmp[ix + 2*wh] = d_costs[ix + 1 + 2*wh];
        }
    }
}

__global__ void update_costs_kernel_step2(int *d_costs, int *d_costs_tmp, int *d_seam, int w, int h, int current_w){
    int row = blockIdx.y*BLOCKSIZE + threadIdx.y;
    int coloumn = blockIdx.x*BLOCKSIZE + threadIdx.x;
    int seam_c = d_seam[row];
    int wh = w*h;
    int ix = row*w + coloumn;
    if(row < h && coloumn < current_w-1 && coloumn >= seam_c-2){
        d_costs[ix] = d_costs_tmp[ix];
        d_costs[ix + wh] = d_costs_tmp[ix + wh];
        d_costs[ix + 2*wh] = d_costs_tmp[ix + 2*wh];
    }
}

int next_pow2(int n){
    int res = 1;
    while(res < n)
        res = res*2;
    return res;
}


/* ############### wrappers #################### */

extern "C"{

#ifndef COMPUTE_COSTS_FULL

void compute_costs(pixel *d_pixels, int *d_costs, int w, int h, int current_w){
    dim3 threads_per_block(BLOCKSIZE, BLOCKSIZE);
    int nblocks_x, nblocks_y;
    nblocks_x = (int)((current_w-1)/(threads_per_block.x-2)) + 1;
    nblocks_y = (int)((h-1)/(threads_per_block.y-1)) + 1;    
    dim3 num_blocks(nblocks_x, nblocks_y);
    compute_costs_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_costs, w, h, current_w);
}

#else

void compute_costs(pixel *d_pixels, int *d_costs, int w, int h, int current_w){
    dim3 threads_per_block(BLOCKSIZE, BLOCKSIZE);
    int nblocks_x, nblocks_y;
    nblocks_x = (int)((current_w-1)/(threads_per_block.x)) + 1;
    nblocks_y = (int)((h-1)/(threads_per_block.y)) + 1;    
    dim3 num_blocks(nblocks_x, nblocks_y);
    compute_costs_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_costs, w, h, current_w);
}

#endif

#ifndef COMPUTE_M_SINGLE

void compute_M(int *d_costs, int *d_M, int w, int h, int current_w){
    
    
    if(current_w <= 1024){
        dim3 threads_per_block(current_w, 1);   
        dim3 num_blocks(1,1);
        compute_M_kernel_small<<<num_blocks, threads_per_block, current_w*sizeof(int)>>>(d_costs, d_M, w, h, current_w);
    }
    else{
        dim3 threads_per_block(WIDEBLOCKSIZE, 1);
        
        dim3 num_blocks;
        num_blocks.x = (int)((current_w-1)/(threads_per_block.x)) + 1;
        num_blocks.y = 1;
        
        dim3 num_blocks2;
        num_blocks2.x = (int)((current_w-WIDEBLOCKSIZE-1)/(threads_per_block.x)) + 1; 
        num_blocks2.y = 1;  

        //printf("%d \n\n",num_blocks2.x);
        
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

#else

//compute M in a single block kernel
void compute_M(int *d_costs, int *d_M, int w, int h, int current_w){

    dim3 threads_per_block(1024, 1);   
    dim3 num_blocks(1,1);
    
    int num_el = (int)((current_w-1)/threads_per_block.x) + 1;
    
    compute_M_kernel_single<<<num_blocks, threads_per_block, 2*current_w*sizeof(int)>>>(d_costs, d_M, w, h, current_w, num_el);
}

#endif


void find_min(int *d_M, int *d_indices, int *d_indices_ref, int w, int h, int current_w){
    //set the reference index array
    cudaMemcpy(d_indices, d_indices_ref, current_w*sizeof(int), cudaMemcpyDeviceToDevice);
    
    dim3 threads_per_block(REDUCEBLOCKSIZE, 1);   

    dim3 num_blocks;
    num_blocks.y = 1; 
    int reduce_num_elements = current_w;
    do{
        num_blocks.x = (int)((reduce_num_elements-1)/(threads_per_block.x*REDUCE_ELEMENTS_PER_THREAD)) + 1;
        min_reduce<<<num_blocks, threads_per_block>>>(&(d_M[w*(h-1)]), d_indices, reduce_num_elements); 
        reduce_num_elements = num_blocks.x;          
    }while(num_blocks.x > 1);
    
}

void find_seam(int* d_M, int *d_indices, int *d_seam, int w, int h, int current_w){
    find_seam_kernel<<<1, 1>>>(d_M, d_indices, d_seam, w, h, current_w);
}

/*
void find_min(int *d_M, int *d_indices, int *d_indices_ref, int w, int h, int current_w){
    find_min_kernel<<<1,1>>>(d_M, d_indices, w, h, current_w);
}*/


void remove_seam(pixel *d_pixels, pixel *d_pixels_tmp, int *d_seam, int w, int h, int current_w){
    dim3 threads_per_block(BLOCKSIZE, BLOCKSIZE);
    int nblocks_x, nblocks_y;
    nblocks_x = (int)((current_w-1)/(threads_per_block.x)) + 1;
    nblocks_y = (int)((h-1)/(threads_per_block.y)) + 1;    
    dim3 num_blocks(nblocks_x, nblocks_y);
    remove_seam_kernel_step1<<<num_blocks, threads_per_block>>>(d_pixels, d_pixels_tmp, d_seam, w, h, current_w);
    remove_seam_kernel_step2<<<num_blocks, threads_per_block>>>(d_pixels, d_pixels_tmp, d_seam, w, h, current_w);
}

//UNUSED
void update_costs(pixel *d_pixels, int *d_costs, int *d_costs_tmp, int *d_seam, int w, int h, int current_w){
    dim3 threads_per_block(BLOCKSIZE, BLOCKSIZE);
    int nblocks_x, nblocks_y;
    nblocks_x = (int)((current_w-1)/(threads_per_block.x)) + 1;
    nblocks_y = (int)((h-1)/(threads_per_block.y)) + 1;    
    dim3 num_blocks(nblocks_x, nblocks_y);
    update_costs_kernel_step1<<<num_blocks, threads_per_block>>>(d_pixels, d_costs, d_costs_tmp, d_seam, w, h, current_w);
    update_costs_kernel_step2<<<num_blocks, threads_per_block>>>(d_costs, d_costs_tmp, d_seam, w, h, current_w);
}


}
