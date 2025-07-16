#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"

/**
 *  $M \times N$ (rows × columns), where $M = 4$ rows and $N = 8$ columns.
 *
 *  matd = [1, 2, 3, 4, 5, 6, 7, 8,           // Row 0
            9, 10, 11, 12, 13, 14, 15, 16,    // Row 1
            17, 18, 19, 20, 21, 22, 23, 24,   // Row 2
            25, 26, 27, 28, 29, 30, 31, 32]   // Row 3

    gridDim.x = 4 (so there are 4 blocks, one for each row).
    blockDim.x = 8 (so each block has 8 threads, one for each element in a row).
    The mapping of rows is as follows:

    Row 0 → matd[0] to matd[7]

    Row 1 → matd[8] to matd[15]

    Row 2 → matd[16] to matd[23]

    Row 3 → matd[24] to matd[31].

    Blocks (blockIdx.x):
    Each block is assigned to process a specific row:

    blockIdx.x = 0 → Block 0 processes row 0 (matd[0:8]).
    blockIdx.x = 1 → Block 1 processes row 1 (matd[8:16]).
    blockIdx.x = 2 → Block 2 processes row 2 (matd[16:24]).
    blockIdx.x = 3 → Block 3 processes row 3 (matd[24:32]).
 *
    Block 0:
    threadIdx.x = 0 → Element 0 of Row 0 (matd[0])
    threadIdx.x = 1 → Element 1 of Row 0 (matd[1])
    threadIdx.x = 2 → Element 2 of Row 0 (matd[2])
    ...
    threadIdx.x = 7 → Element 7 of Row 0 (matd[7])


    Block 1:
    threadIdx.x = 0 → Element 0 of Row 1 (matd[8])
    threadIdx.x = 1 → Element 1 of Row 1 (matd[9])
    threadIdx.x = 2 → Element 2 of Row 1 (matd[10])
    ...
    threadIdx.x = 7 → Element 7 of Row 1 (matd[15])

 */

__global__ void softmax_kernel_3(float *matd, float *resd, int M, int N)
{
    /**
     * blockDim.x stays constant for all blocks: $8$ threads per block.
     * blockIdx.x changes for each block in the grid: $0, 1, 2, 3$.
     */
    __shared__ float smem[1024];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    unsigned int warp_size = 32;

    // edge condition (we don't process further)
    if (row >= M)
        return;

    float *input_row = matd + row * N;
    float *output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // load input values from global momoey to register using memory coalescing
    for (int i = tid; i < N; i += blockDim.x)
    {
        float x = input_row[i];
        if (x > local_max)
        {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    // warp level reduction using XOR shuffle ('exchanges' the values in the threads)
    // note: if there are 256 threads in one block (8 warps of 32 threads each)
    // the following for loop reduces the value in all the 8 warps
    // the 8 warps contain the 8 maximum values of the 32 threads that reside in those warps
    // float val = smem[tid];
    float val = local_max;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
    {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    /**
     * when blockDim is greater than 32, we need to do a block level reduction,
     * blockDim.x > warp_size 代表我們有很多個warp，所以我們需要將每個warp的第一個數值(代表max) 放到share memory
     *
     * AFTER warp level reductions since we have the 8 maximum values that needs to be reduced again
     * the global max will be stored in the first warp
     */

    // global mean, block level using shuffling
    if (blockDim.x > warp_size)
    {
        if (tid % warp_size == 0) // if first index of a warp
        {
            // which warp are we at?
            // store the value in its first thread index
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        // first warp will do global reduction only
        // this is possible because we stored the values in the shared memory
        // so the threads in the first warp will read from it and then reduce
        if (tid < warp_size)
        {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : -INFINITY;
            for (int offset = warp_size / 2; offset > 0; offset /= 2)
            {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            if (tid == 0)
                smem[0] = val;
        }
    }
    else
    {
        // this is for when the number of threads in a block are not
        // greater than the warp size, in that case we already reduced
        // so we can store the value
        if (tid == 0)
            smem[0] = val;
    }
    __syncthreads();
}

void swap(char *x, char *y)
{
    char tmp = *x;
    *x = *y;
    *y = tmp;
}

void swap(int *a, int *b)
{
    int t = *a;
}

void reverse(char *first, char *last)
{
    --last;
    while (first < last)
    {
        swap(first, last);
        ++first;
        --last;
    }
}

int main()
{
    char str[] = "hello world";
    reverse(&str[0], &str[11]);
}