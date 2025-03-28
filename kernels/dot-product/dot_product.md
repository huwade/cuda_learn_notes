# include <stdio.h>
# include <stdlib.h>
# include <float.h>
# include <vector>
# include <algorithm>
# include <cuda_runtime.h>
# include <cuda_fp16.h>
# include <cuda_bf16.h>
# include <cuda_fp8.h>

# define WARP_SIZE 32
# define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
# define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
# define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
# define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
# define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// ---------------------------------------FP32---------------------
// Warp Reduce Sum
/**

* Ref : <https://www.youtube.com/watch?v=D4l1YMsGNlU&ab_channel=catblue>
*
* 先從 accumulate 開始介紹，之後做到dot product，多了一個乘法。
*
* 使用cuda的優勢來計算sum all的方法，符合reduce的定義
* 底下的index的操作只是其中一個方法，還有很多針對sub-optimal的方法
* 可以針對不同的index pattern來找尋對應的方法。

* for(size_t s = blockDim.x/2; s > 0; s>>=1)
* {
*      if(tid < s)
*          sdata[tid] += sdata[tid+s];
*
*      __syncthreads();
* }
 */

/*
這是基本簡單版本，主要是介紹atomic，

__global__ void reduce(float *gdata, float*out)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    while(idx < N) // grid stride loop to load data
    {
        sdata[tid] += gdata[idx];
        idx += gridDim.x * blockDim.x;
    }

    for(size_t s = blockDim.x /2 ; s >0; s>>=1)
    {
        __syncthreads();
        if(tid < s) // parallel sweep reductoin
            sdata[tid] += sdata[tid+s];
    }
    if(tid == 0)
        out[blockIdx.x] = sdata[0];

    *********************************
    Notice that, there is a tech called get rid of 2nd kerenl call.
    from
    '''
        if (tid == 0)
           out[blockIdx.x] = sdata[0];
    '''
    to
    '''
        if(tid == 0)
            atomicAdd(out, sdata[0]);
    '''
    ********************************
}
*/

/* 加入了warp shuffle reduciton

__global__ void reduce_ws(float *gdata, float* out)
{
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    float val = 0.0f;

    unsigned mask = 0xffffffff;
    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;

    // grid stride loop to load

    while(idx < N){
        val += gdata[idx];
        idx += gridDim.x*blockDim.x;
    }

    // 1st warp-shuffle reduction

    for(int offset = warpSize/2; offset > 0; offset >>=1)
        val += shfl_down_sync(mask, val, offset);

    if(lane ==0)
        sdata[warpID] = vla;
    __syncthread(); //put warp result in shared mem

    // here after, just warp 0
    if(warpID==0)
    {
        //reload val from shared mem if warp existed
        val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0;

        //final warp-shfulle reduction
        for(int offset = warpSize/2; offset >0; offset >>=1)
            val+= __shfl_down_sync(mask, val, offset);

        if(tid == 0)
            atomicAdd(out, val);

    }

}
*/

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val)
{
# pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    {
        // __shfl_sync() : copy from lane ID(arbitray pattern)
        //__shfl_xor_sync() : copy form calculated lane ID(calculated pattern)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

/**

* Dot product
* 運用上面介紹到的加法，再結合乘法
* grid(N/256), block(256)
* a: Nx1, b: Nx1, y = sum(element_wise_mul(a,b))
*
* Templates can also accept non-type parameters,
* which are values rather than types.
* These parameters must be constant expressions known at compile time.
*
* 這邊開始介紹warp shuffle，影片1:06:42開始介紹。
* onethread in a warp to send a data to another thread in a warp,
* without the need for slow, high-latency shared memory or global memory accesses.
 */

template <const int NUM_THREADS = 256>
__global__ void dot_prod_f32_f32_kernel(float *a, float*b, float *y, int N)
{
    size_t tid{threadIdx.x};
    size_t idx{blockIdx.x* NUM_THREADS + tid}; // global thread idx
    constexpr size_t NUM_WARPS{(NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE};
    __shared__ float reduce_smem[NUM_WARPS];

    float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
    // 第幾個warp
    int warp = tid / WARP_SIZE;

    // 在warp裡面的第幾個位置
    // "Lane" is a term used to identify the position of a thread within its warp.
    // Lanes are numbered from 0 to 31 within each warp.
    // For example, in a warp of 32 threads, the first thread is in lane 0, the second thread is in lane 1, and so on up to lane 31.
    int lane = tid % WARP_SIZE;

    // use warp shuffle
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);

    if (lane == 0)
    {
        reduce_smem[warp] = prod;
    }
    __syncthreads();
    prod = lane < NUM_WARPS ? reduce_smem[lane] : 0.0f;

    if (warp == 0)
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    if (tid == 0)
        atomicAdd(y, prod);
    /**
     * from
     *      if (tid == 0)
     *          out[blockIdx.x] = sdata[0];
     *
     * to
     *      if(tid == 0)
     *          atomicAdd(out, sdata[0]);
     *
     * Is a tech called get rid of 2nd kerenl call
     */
}

template <const int NUM_THREADS = 256 / 4>
__global__ void dot_prod_f32x4_f32_kernel(float *a, float*b, float *y, int N)
{
    int tid = threadIdx.x;
    int idx = (blockIdx.x* NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float prod = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y + reg_a.z * reg_b.z + reg_a.w * reg_b.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
    // warp leaders store the data to shared memory.
    if (lane == 0)
        reduce_smem[warp] = prod;
    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    if (tid == 0)
        atomicAdd(y, prod);
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
    // Initialize matrices
    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }
    *y = 0.0f;

    // Define the number of threads and blocks
    constexpr int NUM_THREADS = 256;
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    dot_prod_f32_f32_kernel<NUM_THREADS><<<NUM_BLOCKS, s>>>(h_A, h_B, y, N);
    cudaDeviceSynchronize();

    printf("Dot product is %f\n", *y);
    return 0;
}
