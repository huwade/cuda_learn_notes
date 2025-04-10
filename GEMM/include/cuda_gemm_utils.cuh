#ifndef CUDA_GEMM_UTILS_CUH
#define CUDA_GEMM_UTILS_CUH

#include <cuda_runtime.h>

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_X = 0U,
          size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_from_global_memory_to_shared_memory(
    T const *A, size_t lda, T const *B, size_t ldb,
    T A_thread_block_tile[BLOCK_TILE_SIZE_Y]
                         [BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
    T B_thread_block_tile[BLOCK_TILE_SIZE_K]
                         [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n,
    size_t k)
{
    // Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
                        NUM_THREADS;
         ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               A_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        T val{static_cast<T>(0)};
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx * lda + A_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.
        static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS ==
                      0U);
        // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
        //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
        // {
        //     A_thread_block_tile[A_thread_block_tile_row_idx]
        //                        [A_thread_block_tile_col_idx] = val;
        // }
        A_thread_block_tile[A_thread_block_tile_row_idx]
                           [A_thread_block_tile_col_idx] = val;
    }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
                        NUM_THREADS;
         ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X};
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               B_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        T val{static_cast<T>(0)};
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.
        static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS ==
                      0U);
        // if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
        //     B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
        // {
        //     B_thread_block_tile[B_thread_block_tile_row_idx]
        //                        [B_thread_block_tile_col_idx] = val;
        // }
        B_thread_block_tile[B_thread_block_tile_row_idx]
                           [B_thread_block_tile_col_idx] = val;
    }
}

#endif