/**
 * please compile with -arch, when I run this code on A100, compile without arch I got
 * reduction w/atomic sum incorrect! reduce w/atomic sum is 0.000000, but with arch I got the right answer.
 */
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256
const size_t N = 8ULL * 1024ULL * 1024ULL; // data size

// sequential addressing, with atomic
__global__ void reduce_a(float *gdata, float *out)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    while (idx < N)
    {
        // grid stride loop to load data
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
    int lane = threadIdx.x % warpSize;   // which thread am I in the warp. each warp has 32 thread.
    int warpID = threadIdx.x / warpSize; // multiple warp, warp 0, 1, ...

    // instead in share memory, this time we run val, is local variable
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

/**
 * compare reduce_a it is faster.
 */
__global__ void reduce4(float *gdata, float *out)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = threadIdx.x + (blockDim.x * 2) * blockIdx.x;

    // Load data to shared memory
    while (idx < N)
    {
        // grid stride loop to load data, First Add During Load
        sdata[tid] = gdata[idx] + gdata[idx + blockDim.x];
        idx += gridDim.x * blockDim.x * 2;
    }

    __syncthreads();

    // do reduction in shared mem, reduction tree
    for (size_t s = blockDim.x >> 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        out[blockIdx.x] = sdata[0];
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

    // Define the number of threads and blocks
    int threadsPerBlock = 256;                                     // Number of threads per block
    int numBlocks = (N + threadsPerBlock - 1) / (threadsPerBlock); // Number of blocks

    const int blockSize = 640;
    cudaMemset(d_sum, 0, sizeof(float));

    reduce_a<<<blockSize, BLOCK_SIZE>>>(d_A, d_sum);
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if (*h_sum != (float)N)
    {
        printf("reduction w/atomic sum incorrect!\n");
        printf("reduce w/atomic sum is %f\n", *h_sum);
        return -1;
    }
    printf("reduction w/atomic sum correct!\n");

    cudaMemset(d_sum, 0, sizeof(float));
    reduce_ws<<<numBlocks, threadsPerBlock>>>(d_A, d_sum);
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    // Wait for GPU to finish before accessing on host

    if (*h_sum != (float)N)
    {
        printf("reduction warp shuffle sum incorrect!\n");

        return -1;
    }
    printf("reduction warp shuffle sum correct!!\n");

    // Define the number of threads and blocks
    cudaMemset(d_sum, 0, sizeof(float));

    reduce4<<<numBlocks, threadsPerBlock>>>(d_A, d_sum);

    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if (*h_sum != (float)N)
    {
        printf("reduction4  sum incorrect!\n");
        printf("reduce4 sum is %f\n", *h_sum);
        return -1;
    }
    printf("reduction4 sum correct!\n");

    // Clean up
    delete[] h_A;
    delete h_sum;
    cudaFree(d_A);
    cudaFree(d_sum);

    return 0;
}