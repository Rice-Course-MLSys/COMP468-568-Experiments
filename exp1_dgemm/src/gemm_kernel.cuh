#pragma once

#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32;

inline dim3 make_grid(int m, int n) {
    return dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (m + BLOCK_SIZE - 1) / BLOCK_SIZE,
                1);
}

__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int N,
                                  int K) {
    /* TODO(student): compute row/col indices, accumulate dot product, write to C */
}

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int N,
                                  int K) {
    /* TODO(student): use shared memory tiles of size BLOCK_SIZE x BLOCK_SIZE */
}

inline void launch_naive_gemm(const float* d_a,
                              const float* d_b,
                              float* d_c,
                              int M,
                              int N,
                              int K,
                              cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_grid(M, N);
    /* TODO(student): launch gemm_naive_kernel with provided stream. */
    (void)d_a;
    (void)d_b;
    (void)d_c;
    (void)grid;
    (void)block;
    (void)stream;
}

inline void launch_tiled_gemm(const float* d_a,
                              const float* d_b,
                              float* d_c,
                              int M,
                              int N,
                              int K,
                              cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_grid(M, N);
    /* TODO(student): launch gemm_tiled_kernel and check for errors. */
    (void)d_a;
    (void)d_b;
    (void)d_c;
    (void)grid;
    (void)block;
    (void)stream;
}

