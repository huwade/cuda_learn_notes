#include <cuda_runtime.h>
#include <assert.h>
#include "matmul.h"

const int threadDim = 32;

template <const int TILE_SIZE>
__global__ void sgemm_shared_mem_block(const float *A, const float *B, float *C, int A_row, int A_column, int B_column)
{
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[TILE_SIZE * TILE_SIZE];
    __shared__ float Bs[TILE_SIZE * TILE_SIZE];

    // the inner row & col that we're accessing in this thread
    const uint threadCol = threadIdx.x % TILE_SIZE;
    const uint threadRow = threadIdx.x / TILE_SIZE;

    // advance pointers to the starting positions
    A += cRow * TILE_SIZE * A_column;                    // row=cRow, col=0
    B += cCol * TILE_SIZE;                               // row=0, col=cCol
    C += cRow * TILE_SIZE * B_column + cCol * TILE_SIZE; // row=cRow, col=cCol

    float tmp = 0;
    for (int bkIdx = 0; bkIdx < A_column; bkIdx += TILE_SIZE)
    {
        As[threadRow * TILE_SIZE + threadCol] = A[threadRow * A_column + threadCol];
        Bs[threadRow * TILE_SIZE + threadCol] = B[threadRow * B_column + threadCol];
        __syncthreads();

        // block threads in this block until cache is fully populated
        __syncthreads();
        A += TILE_SIZE;
        B += TILE_SIZE * B_column;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < TILE_SIZE; ++dotIdx)
        {
            tmp += As[threadRow * TILE_SIZE + dotIdx] *
                   Bs[dotIdx * TILE_SIZE + threadCol];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    C[threadRow * B_column + threadCol] = tmp;
}

namespace matmul
{

    void MatmulOperator::mat_mul_cuda_shared_mem(const struct matmul_params *params)
    {
        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        assert(A->column == B->row);
        assert(C->column == B->column);
        assert(C->row == A->row);

        float *d_A;
        float *d_B;
        float *d_C;

        // Initailize C
        /*for (int i = 0; i < C->row; i++)
          for (int j = 0; j < C->column; j++)
          C->data_ptr[j + C->column * i] = 0;*/

        // Allocate memory
        cudaMalloc(&d_A, A->column * A->row * sizeof(float));
        cudaMalloc(&d_B, B->column * B->row * sizeof(float));
        cudaMalloc(&d_C, C->column * C->row * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_A, A->data_ptr, A->column * A->row * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B->data_ptr, B->column * B->row * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, C->data_ptr, C->column * C->row * sizeof(float), cudaMemcpyHostToDevice);

        // Make sure we can break the input matrix into blocks
        assert(A->column % threadDim == 0);
        assert(A->row % threadDim == 0);
        assert(B->column % threadDim == 0);
        const dim3 threadsPerBlock(threadDim, threadDim);
        const dim3 numBlocks(C->column / threadsPerBlock.x, C->row / threadsPerBlock.y);

        // Invoke the cuda imp.
        // matrixMul_blockC<<< numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A->row, A->column, B->column);
        sgemm_shared_mem_block<32><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A->row, A->column, B->column);

        // Get the result back
        cudaMemcpy(C->data_ptr, d_C, C->column * C->row * sizeof(float), cudaMemcpyDeviceToHost);
    }
}
