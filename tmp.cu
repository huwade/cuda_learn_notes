#include <cstdio>
#include <cuda_runtime.h>

#define BLOCKSIZE 256

#define warpSize 32

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            getchar();
            exit(code);
        }
    }
}

/******************/
/* REDUCE0 KERNEL */
/******************/
/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
template <class T>
__global__ void reduce0(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;                         // Local thread index
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    // --- Loading data to shared memory
    sdata[tid] = (i < N) ? g_idata[i] : 0;

    // --- Before going further, we have to make sure that all the shared memory loads have been completed
    __syncthreads();

    // --- Reduction in shared memory
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        // --- Only the threads with index multiple of 2*s perform additions. Furthermore, modulo arithmetic is slow.
        if ((tid % (2 * s)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        // --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
    //     individual blocks
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

/******************/
/* REDUCE1 KERNEL */
/******************/
/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
template <class T>
__global__ void reduce1(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;                         // Local thread index
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    // --- Loading data to shared memory
    sdata[tid] = (i < N) ? g_idata[i] : 0;

    // --- Before going further, we have to make sure that all the shared memory loads have been completed
    __syncthreads();

    // --- Reduction in shared memory
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        // --- Use contiguous threads leading to non-divergent branch
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s]; /* --- Strided shared memory access */
        }
        // --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
    //     individual blocks
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;                         // Local thread index
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    // --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
    sdata[tid] = (i < N) ? g_idata[i] : 0;

    // --- Before going further, we have to make sure that all the shared memory loads have been completed
    __syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        // --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
    //     individual blocks
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

/******************/
/* REDUCE3 KERNEL */
/******************/
/*
    This version performs the first level of reduction using registers when reading from global memory.
*/
template <class T>
__global__ void reduce3(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;                               // Local thread index
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // Global thread index - Fictitiously double the block dimension

    // --- Performs the first level of reduction in registers when reading from global memory.
    T mySum = (i < N) ? g_idata[i] : 0;
    if (i + blockDim.x < N)
        mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;

    // --- Before going further, we have to make sure that all the shared memory loads have been completed
    __syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        // --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
    //     individual blocks
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

/******************/
/* REDUCE4 KERNEL */
/******************/
/*
    This version uses the warp shuffle operation if available to reduce
    warp synchronization. When shuffle is not available the final warp's
    worth of work is unrolled to reduce looping overhead.

    This kernel assumes that blockSize > 64.
*/
template <class T>
__global__ void reduce4(T *g_idata, T *g_odata, unsigned int N)
{
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;                               // Local thread index
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // Global thread index - Fictitiously double the block dimension

    // --- Performs the first level of reduction in registers when reading from global memory.
    T mySum = (i < N) ? g_idata[i] : 0;
    if (i + blockDim.x < N)
        mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;

    // --- Before going further, we have to make sure that all the shared memory loads have been completed
    __syncthreads();

    // --- Reduction in shared memory. Only half of the threads contribute to reduction.
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        // --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
        __syncthreads();
    }

#if (__CUDA_ARCH__ >= 300)
    // --- Single warp reduction by shuffle operations
    if (tid < 32)
    {
        // --- Last iteration removed from the for loop, but needed for shuffle reduction
        mySum += sdata[tid + 32];
        // --- Reduce final warp using shuffle
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            mySum += __shfl_down(mySum, offset);
        // for (int offset=1; offset < warpSize; offset *= 2) mySum += __shfl_xor(mySum, i);
    }
#else
    // --- Single warp reduction by loop unrolling. Assuming blockDim.x >64
    if (tid < 32)
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
        __syncthreads();
        sdata[tid] = mySum = mySum + sdata[tid + 16];
        __syncthreads();
        sdata[tid] = mySum = mySum + sdata[tid + 8];
        __syncthreads();
        sdata[tid] = mySum = mySum + sdata[tid + 4];
        __syncthreads();
        sdata[tid] = mySum = mySum + sdata[tid + 2];
        __syncthreads();
        sdata[tid] = mySum = mySum + sdata[tid + 1];
        __syncthreads();
    }
#endif

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
    //     individual blocks
    if (tid == 0)
        g_odata[blockIdx.x] = mySum;
}

#define BLOCK_SIZE 256
const size_t N = 8ULL * 1024ULL * 1024ULL; // data size

int main()
{

    // --- Creating events for timing
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *input, *output;
    input = new float[N]; // allocate space for data in host memory
    output = new float;
    for (int i = 0; i < N; i++) // initialize matrix in host memory
        input[i] = 1.0f;

    cudaMallocManaged(&input, N * sizeof(float));
    cudaMallocManaged(&output, N * sizeof(float));
    cudaEventRecord(start, 0);

    int NumThreads = 640;
    int numBlocks = (N + NumThreads - 1) / NumThreads;
    int smemSize = (NumThreads <= 32) ? 2 * NumThreads * sizeof(int) : NumThreads * sizeof(int);

    reduce0<float><<<numBlocks, NumThreads, smemSize>>>(input, output, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("reduce0 - Elapsed time:  %3.3f ms \n", time);
}