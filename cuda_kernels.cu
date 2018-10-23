//#include <cuda_runtime.h>

extern "C"{
#include <stdio.h>
//#include <math.h>
#include <limits.h>
#include "image.h"
}

const int BLOCKSIZE = 16;

__global__ void compute_costs_kernel(pixel* d_pixels, unsigned int* d_costs, int w, int h, int current_w){
    //first row, first coloumn and last coloumn of shared memory are reserved for halo...
    __shared__ pixel pix_cache[BLOCKSIZE][BLOCKSIZE];
    //...and the global index in the image is computed accordingly to this 
    int row = blockIdx.y*(BLOCKSIZE-1) + threadIdx.y -1 ; 
    int coloumn = blockIdx.x*(BLOCKSIZE-2) + threadIdx.x -1; 
    int ix = row*w + coloumn;
    unsigned int wh = w*h;
    unsigned int cache_r = threadIdx.y;
    unsigned int cache_c = threadIdx.x;
    short active = 0;
     
    if(row < h && coloumn <= current_w){
        //only threads with row in [-1,h-1] and coloumn in [-1,current_w] are actually active*/
        active = 1;
        //if access to the image is out of bounds, set RGB values to 0
        //otherwise load pixel from global memory
        if(row < 0 || coloumn < 0 || coloumn == current_w){
            pix_cache[cache_r][cache_c].r = 0;
            pix_cache[cache_r][cache_c].g = 0;
            pix_cache[cache_r][cache_c].b = 0;
        }
        else
            pix_cache[cache_r][cache_c] = d_pixels[ix];
    }
    
    //wait until each thread has initialized its portion of shared memory
    __syncthreads();
    
    //all the threads that are NOT in halo positions can now compute costs, with fast access to shared memory
    if(active && cache_r != 0 && cache_c != 0 
        && cache_c != BLOCKSIZE-1 && coloumn < current_w){
        unsigned int rdiff, gdiff, bdiff;
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
    }
       
}

const int WIDEBLOCKSIZE = 128; //must be divisible by 2

__global__ void compute_M_kernel_phase1(unsigned int *d_costs, unsigned int* d_M, int w, int h, int current_w, int base_row){
    __shared__ unsigned int m_cache[WIDEBLOCKSIZE];
    unsigned int row;
    unsigned int coloumn = blockIdx.x*WIDEBLOCKSIZE + threadIdx.x; 
    unsigned int cache_coloumn = threadIdx.x; 
    unsigned int wh = w*h;
    short is_first = 0;
    short is_last = 0;
    unsigned int right, up, left;
    
    if(blockIdx.x == 0)
        is_first = 1;
    else if(blockIdx.x == gridDim.x-1)
        is_last = 1;
    
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
                left = UINT_MAX;
            //with up
            up = m_cache[cache_coloumn] + d_costs[ix + wh];
            //with right
            if(coloumn < current_w-1)
                right = m_cache[cache_coloumn + 1] + d_costs[ix + 2*wh];
            else
                right = UINT_MAX;
                
            left = min(left, min(up, right));
            d_M[ix] = left;
            m_cache[cache_coloumn] = left;            
        }   
        __syncthreads();
    }
}


__global__ void compute_M_kernel_phase2(unsigned int *d_costs, unsigned int* d_M, int w, int h, int current_w, int base_row){
    //__shared__ unsigned int m_cache[WIDEBLOCKSIZE];
    int row;
    int coloumn = blockIdx.x*WIDEBLOCKSIZE + threadIdx.x + WIDEBLOCKSIZE/2; 
    int wh = w*h;
    // cache_coloumn = threadIdx.x; 
    unsigned int right, up, left;
   
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
            else
                right = UINT_MAX;
  
            left = min(left, min(up, right));
            d_M[ix] = left;
        }
        __syncthreads();
    }
}

const int REDUCEBLOCKSIZE = 128;
const int ELEMENTS_PER_THREAD = 8;
__global__ void min_reduce(unsigned int* d_values, int* d_indices, int N){
    __shared__ unsigned int val_cache[REDUCEBLOCKSIZE];
    __shared__ unsigned int ix_cache[REDUCEBLOCKSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int coloumn = blockIdx.x*REDUCEBLOCKSIZE + ELEMENTS_PER_THREAD*threadIdx.x; 
    unsigned int min_v = UINT_MAX;
    unsigned int min_i = 0;
    unsigned int i;
    for(i = 0; i < ELEMENTS_PER_THREAD && coloumn + i < N; i++){
            unsigned int new_i = d_indices[coloumn + i];
            unsigned int new_v  = d_values[new_i];
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
            if(val_cache[tid] > val_cache[tid + i]){
                val_cache[tid] = val_cache[tid + i];
                ix_cache[tid] = ix_cache[tid + i];
            }
        }
        __syncthreads();
    }
    
    if(tid == 0)
        d_indices[blockIdx.x] = ix_cache[0];   
    
}


__global__ void find_seam_kernel(unsigned int* d_M, int *d_indices, int *d_seam, int w, int h, int current_w){    
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


const int SHIFTBLOCKSIZE = 32;

__global__ void remove_seam_kernel(pixel *d_pixel, int *d_seam, int w, int h, int current_w){
    int row = blockIdx.y*SHIFTBLOCKSIZE + threadIdx.y;
    if(row < h){
        int c;
        for(c = d_seam[row]; c < current_w-1; c++)
            d_pixel[row*w + c] = d_pixel[row*w + c + 1];
    }

}


/*
__global__ void update_costs(unsigned int* d_M, unsigned int* d_costs, unsigned int* d_seam, int w, int h, int current_w){
    int row = blockIdx.y*SHIFTBLOCKSIZE + threadIdx.y;
}
*/

/*
 //////SIMPLE ///////////
 
__global__ void compute_M_kernel(unsigned int *d_costs, unsigned int* d_M, int w, int h, int current_w, int row){
    //__shared__ unsigned int m_cache[WIDEBLOCKSIZE];
    int coloumn = blockIdx.x*WIDEBLOCKSIZE + threadIdx.x;   
    
    //__syncthreads();
    
    unsigned int left, up, right;
    if(coloumn < current_w){
        //with left
            if(coloumn > 0)
                left = d_costs[(row-1)*w + coloumn -1] + d_costs[row*w + coloumn]; 
            else 
                left = UINT_MAX;
            //with up
            up = d_costs[(row-1)*w + coloumn] + d_costs[row*w + coloumn + w*h];
            //with right
            if(coloumn < current_w-1)
                right = d_costs[(row-1)*w + coloumn +1] + d_costs[row*w + coloumn + 2*w*h];
            else
                right = UINT_MAX;
                
            d_M[row*w + coloumn] = min(left, min(up, right));         
    } 
}

//only for row 0
__global__ void compute_M_kernel_init(unsigned int *d_costs, int w, int h, int current_w){
    int coloumn = blockIdx.x*WIDEBLOCKSIZE + threadIdx.x;
    if(coloumn < current_w)
        d_M[coloumn] = min(d_costs[coloumn], min(d_costs[coloumn + w*h], d_costs[coloumn + 2*w*h]));

}

///////////////END SIMPLE //////////
*/

/*wrappers */
extern "C"{

void compute_costs(pixel *d_pixels, unsigned int *d_costs, int w, int h, int current_w){
    dim3 threads_per_block(BLOCKSIZE, BLOCKSIZE);
    int nblocks_x, nblocks_y;
    nblocks_x = (int)((current_w-1)/(threads_per_block.x-2)) + 1;
    nblocks_y = (int)((h-1)/(threads_per_block.y-1)) + 1;    
    dim3 num_blocks(nblocks_x, nblocks_y);
    compute_costs_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_costs, w, h, current_w);
}


void compute_M(unsigned int *d_costs, unsigned int *d_M, int w, int h, int current_w){
    dim3 threads_per_block(WIDEBLOCKSIZE, 1);
    
    int nblocks_x;
    nblocks_x = (int)((current_w-1)/(threads_per_block.x)) + 1;   
    
    if(nblocks_x == 1){
        //simple
    }
    else{
        dim3 num_blocks;
        num_blocks.x = nblocks_x;
        num_blocks.y = 1;
        
        dim3 num_blocks2;
        num_blocks2.y = 1;
        nblocks_x = (int)((current_w-WIDEBLOCKSIZE-1)/(threads_per_block.x)) + 1;   
            num_blocks2.x = nblocks_x;

        //printf("%d \n\n",num_blocks2.x);
        
        int num_iterations;
        num_iterations = (int)((h-1)/(WIDEBLOCKSIZE/2)) + 1;
            
        int i;
        int base_row = 0;
        for(i = 0; i < num_iterations; i++){
            compute_M_kernel_phase1<<<num_blocks, threads_per_block>>>(d_costs, d_M, w, h, current_w, base_row);
            compute_M_kernel_phase2<<<num_blocks2, threads_per_block>>>(d_costs, d_M, w, h, current_w, base_row);
            base_row = base_row + (WIDEBLOCKSIZE/2) - 1;    
        }
    }
}

void find_min(unsigned int *d_M, int *d_indices, int *d_indices_ref, int w, int h, int current_w){
    //set the reference index array
    cudaMemcpy(d_indices, d_indices_ref, current_w*sizeof(int), cudaMemcpyDeviceToDevice);
    
    dim3 threads_per_block(REDUCEBLOCKSIZE, 1);   

    dim3 num_blocks;
    num_blocks.y = 1; 
    do{
        num_blocks.x = (int)((current_w-1)/(threads_per_block.x*ELEMENTS_PER_THREAD)) + 1;
        min_reduce<<<num_blocks, threads_per_block>>>(&(d_M[w*(h-1)]), d_indices, current_w); 
        current_w = num_blocks.x;
    }while(num_blocks.x > 1);
}

void find_seam(unsigned int* d_M, int *d_indices, int *d_seam, int w, int h, int current_w){
    find_seam_kernel<<<1, 1>>>(d_M, d_indices, d_seam, w, h, current_w);
}

void remove_seam(pixel *d_pixel, int *d_seam, int w, int h, int current_w){
    dim3 threads_per_block(1, SHIFTBLOCKSIZE);   

    dim3 num_blocks;
    num_blocks.x = 1;
    num_blocks.y = (int)((h-1)/(threads_per_block.y)) + 1;
    remove_seam_kernel<<<num_blocks, threads_per_block>>>(d_pixel, d_seam, w, h, current_w);
}


/*

/////////////SIMPLE ///////////////////////


void compute_M(unsigned int *d_costs, unsigned int *d_M, int w, int h, int current_w){
    dim3 threads_per_block(WIDEBLOCKSIZE, 1);
    int nblocks_x;
    nblocks_x = (int)((current_w-1)/(threads_per_block.x)) + 1;    
    dim3 num_blocks(nblocks_x, 1);
    compute_M_kernel_init<<<num_blocks, threads_per_block>>>(d_costs, d_M, w, h, current_w);
    int row;
    for(row = 1; row < h; row++)
        compute_M_kernel<<<num_blocks, threads_per_block>>>(d_costs, d_M, w, h, current_w, row);
}

*/


}
