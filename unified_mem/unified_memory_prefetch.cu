#include <iostream>
#include <math.h>

__global__ void add(int n, float *x, float *y)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride)
    {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20;
    float *x, *y;
    int device = -1;

    // allocate unified memory -- accessible from cpu or gpu
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    cudaGetDevice(&device);

    cudaMemPrefetchAsync(x, N * sizeof(float), device, NULL);
    cudaMemPrefetchAsync(y, N * sizeof(float), device, NULL);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaMemPrefetchAsync(y, N * sizeof(float), cudaCpuDeviceId, NULL);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}