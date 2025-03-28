#include <cstdio>
#include <cuda_runtime.h>

#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val)
{
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int NUM_THREADS = 256>
__global__ void dot_prod_f32_f32_kernel(float *a, float *b, float *y, int N)
{
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid; // global thread idx
    constexpr size_t NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // Use warp shuffle to sum within the warp
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);

    if (lane == 0)
    {
        reduce_smem[warp] = prod;
    }
    __syncthreads();

    prod = (tid < (NUM_WARPS * WARP_SIZE) && lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;

    if (warp == 0)
    {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
        if (tid == 0)
        {
            atomicAdd(y, prod);
        }
    }
}

const int N = 1024;

int main()
{
    float *h_A, *h_B, *y;
    cudaError_t err;

    // Allocate memory on the device
    err = cudaMallocManaged(&h_A, N * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("Error allocating h_A: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMallocManaged(&h_B, N * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("Error allocating h_B: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMallocManaged(&y, sizeof(float));
    if (err != cudaSuccess)
    {
        printf("Error allocating y: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Initialize arrays
    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }
    *y = 0.0f;

    // Define the number of threads and blocks
    constexpr int NUM_THREADS = 256;
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // Launch the kernel
    dot_prod_f32_f32_kernel<NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS>>>(h_A, h_B, y, N);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for device to finish all operations
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("Device synchronization failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Dot product is %f\n", *y);

    // Free device memory
    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(y);

    return 0;
}