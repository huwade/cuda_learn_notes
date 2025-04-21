#include <cassert>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char *const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void add_val_in_place(int32_t *data, int32_t val, uint32_t n)
{
    uint32_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    uint32_t const stride{blockDim.x * gridDim.x};

    for (uint32_t i{idx}; i < n; i += stride)
    {
        data[i] += val;
    }
}

void launch_add_val_in_place(int32_t *data, int32_t val, uint32_t n, cudaStream_t stream)
{
    dim3 const threads_pre_block{1024};
    dim3 const blocks_pre_grid{32};

    add_val_in_place<<<blocks_pre_grid, threads_pre_block, 0, stream>>>(data, val, n);
    CHECK_LAST_CUDA_ERROR();
}

bool check_array_value(int32_t const *data, uint32_t n, int32_t val)
{
    for (uint32_t i{0}; i < n; i++)
    {
        if (data[i] != val)
            return false;
    }
    return true;
}

int main()
{
    constexpr uint32_t const n{1000000};
    constexpr int32_t const val_1{1};
    constexpr int32_t const val_2{2};
    constexpr int32_t const val_3{3};

    // create an multi-stream application
    cudaStream_t stream1{0};
    cudaStream_t stream2{0};

    // stream1 is a non-default blocking stream
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));

    std::vector<int32_t> vec(n, 0);
    int32_t *d_data{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, n * sizeof(int32_t)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, vec.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));

    // run a sequence of cuda kernels in order on the same cuda stream
    launch_add_val_in_place(d_data, val_1, n, stream1);

    // the second kernel lanuch is supposed to be run on stream1.
    // however the implemetation has a typo such that the kernel launch
    // is run on the default stream2
    launch_add_val_in_place(d_data, val_2, n, stream2);
    launch_add_val_in_place(d_data, val_3, n, stream1);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaMemcpy(vec.data(), d_data, n * sizeof(int32_t),
                                cudaMemcpyDeviceToHost));

    // Check the correctness of the application.
    // Yet the result will still be correct if the default stream_2
    // is a legacy default stream.
    assert(check_array_value(vec.data(), n, val_1 + val_2 + val_3));

    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
}
/**

在下面的範例中，我使用 cudaStreamCreate 建立了一個非預設阻塞流。對於一系列應該在同一個非預設阻塞 CUDA 流上按順序運行的 CUDA 內核，
我犯了一個錯誤，意外地將其中一個內核的預設流用於其中。
如果預設流是預設舊流，則當在舊流中採取操作（例如核心啟動或 cudaStreamWaitEvent()）時，舊流首先等待所有阻塞流，該操作在舊流中排隊，
然後所有阻塞流都在舊流上等待。因此，即使我犯了錯誤，CUDA 核心仍然按順序運行，並且應用程式的正確性不受影響。
如果預設流是預設的每個執行緒流，則它是非阻塞的並且不會與其他 CUDA 流同步。因此我的錯誤會導致應用程式無法正確運行。
 */