#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256
const size_t N = 8ULL * 1024ULL * 1024ULL; // data size

__global__ void reduce_a(float *gdata, float *out)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    while (idx < N)
    {
        // grid stride loop
        sdata[tid] += gdata[idx];
        idx += gridDim.x * blockDim.x;
    }

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tid < s)
            sdata[tid] += sdata[tid + s];
    }

    if (tid == 0)
        atomicAdd(out, sdata[0]);
}

// ref: https://zhuanlan.zhihu.com/p/572820783 , https://youtu.be/D4l1YMsGNlU?si=px7i_6gUxx4D7CRZ&t=4804
__global__ void reduce_ws(float *gdata, float *out)
{
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float val = 0.0f;

    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;

    while (idx < N)
    {
        val += gdata[idx];
        idx += gridDim.x * blockDim.x;
    }

    for (size_t s = warpSize >> 1; s > 0; s >>= 1)
    {
        val += __shfl_down_sync(mask, val, s);
    }

    if (lane == 0)
        sdata[warpID] = val;

    // put warp result in shared mem
    __syncthreads();

    if (warpID == 0)
    {
        // reload val from shared mem
        val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0;

        for (size_t s = warpSize >> 1; s > 0; s >>= 1)
            val += __shfl_down_sync(mask, val, s);

        if (tid == 0)
            atomicAdd(out, val);
    }
}

int main()
{
    float *h_A, *h_sum, *d_A, *d_sum;
    h_A = new float[N]; // allocate space for data in host memory
    h_sum = new float;
    for (int i = 0; i < N; i++) // initialize matrix in host memory
        h_A[i] = 1.0f;

    cudaMalloc(&d_A, N * sizeof(float)); // allocate device space for A
    cudaMalloc(&d_sum, sizeof(float));   // allocate device space for sum

    // copy matrix A to device:
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));
    // Define the number of threads and blocks
    int blockSize = 640;
    int numBlocks = (N + blockSize - 1) / blockSize;

    reduce_a<<<numBlocks, BLOCK_SIZE>>>(d_A, d_sum);
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if (*h_sum != (float)N)
    {
        printf("reduction w/atomic sum incorrect!\n");
        printf("reduce w/atomic sum is %f\n", *h_sum);
        return -1;
    }
    printf("reduction w/atomic sum correct!\n");

    cudaMemset(d_sum, 0, sizeof(float));
    reduce_ws<<<numBlocks, blockSize>>>(d_A, d_sum);
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    // Wait for GPU to finish before accessing on host

    if (*h_sum != (float)N)
    {
        printf("reduction warp shuffle sum incorrect!\n");

        return -1;
    }
    printf("reduction warp shuffle sum correct!!\n");

    // Clean up
    delete[] h_A;
    delete h_sum;
    cudaFree(d_A);
    cudaFree(d_sum);

    return 0;
}