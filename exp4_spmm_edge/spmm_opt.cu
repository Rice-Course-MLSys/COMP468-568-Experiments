
// spmm_opt.cu — Two-Step GNN: SDDMM + SpMM Optimized (STUDENT SKELETON)
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

using float_t = float;

extern void load_csr_from_edgelist(const std::string& filename,
                                   int& M, int& K,
                                   std::vector<int>& row_ptr,
                                   std::vector<int>& col_idx,
                                   std::vector<float>& vals);

extern void sddmm_cpu(int M, int D,
                       const std::vector<int>& row_ptr,
                       const std::vector<int>& col_idx,
                       const std::vector<float>& E,
                       std::vector<float>& vals_out);

extern void spmm_cpu(int M, int K, int N,
                     const std::vector<int>& row_ptr,
                     const std::vector<int>& col_idx,
                     const std::vector<float>& vals,
                     const std::vector<float>& B,
                     std::vector<float>& C);

extern float max_abs_err(const std::vector<float>& A, const std::vector<float>& B);

/*
=================================================================
 SDDMM OPTIMIZED KERNEL — STUDENT TODO
 One warp per row.
 Each lane in the warp handles a subset of the row's nonzeros.
 For each assigned edge (i, col_idx[p]), compute the full
 dot product dot(E[i,:], E[col,:]) over D dimensions.
 No row_indices array needed — the warp knows its row.
=================================================================
*/
__global__ void sddmm_csr_warp_kernel(
    int M, int D,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float_t* __restrict__ d_E,       // M x D embedding matrix
    float_t* __restrict__ d_vals)          // nnz: output edge weights
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = global_tid / 32;
    int lane = threadIdx.x % 32;

    if (warp >= M) return;
    int row = warp;

    // TODO student: fetch start and end from d_row_ptr
    // TODO student: for each nonzero p = start + lane, start + lane + 32, ...
    //   - read column j from d_col_idx[p]
    //   - compute dot(E[row,:], E[j,:]) over D dimensions
    //   - write result to d_vals[p]
}

/*
=================================================================
 SpMM OPTIMIZED KERNEL — STUDENT TODO
 One warp per row. Lanes split across output columns.
=================================================================
*/
__global__ void spmm_csr_warp_kernel(
    int M, int N,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float_t* __restrict__ d_vals,
    const float_t* __restrict__ d_B,
    float_t* __restrict__ d_C)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = global_tid / 32;
    int lane = threadIdx.x % 32;

    if (warp >= M) return;
    int row = warp;

    // TODO student: fetch start, end
    // TODO student: for j = lane; j < N; j += 32 ...
    // TODO student: loop over nonzeros
    // TODO student: accumulate
}

int main() {
    int M, K;
    const int D = 64;  // embedding dimension
    std::vector<int> row_ptr, col_idx;
    std::vector<float> vals;

    load_csr_from_edgelist("graph_edges.txt", M, K, row_ptr, col_idx, vals);
    int nnz = row_ptr.back();
    assert(M == K && "Adjacency matrix must be square");

    std::cout << "Loaded graph for optimized kernels: M=" << M
              << " nnz=" << nnz << " D=" << D << "\n";

    // --- Generate random embedding E (M x D) ---
    std::vector<float> E((size_t)M * D);
    srand(42);
    for (size_t i = 0; i < E.size(); i++) E[i] = float(rand()) / RAND_MAX;

    // === CPU Reference ===
    std::vector<float> vals_ref;
    sddmm_cpu(M, D, row_ptr, col_idx, E, vals_ref);

    std::vector<float> C_ref;
    spmm_cpu(M, M, D, row_ptr, col_idx, vals_ref, E, C_ref);

    // === GPU Setup ===
    int *d_row_ptr, *d_col_idx;
    float *d_vals, *d_E, *d_C;
    cudaMalloc(&d_row_ptr, (M+1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_vals, nnz * sizeof(float));
    cudaMalloc(&d_E, (size_t)M * D * sizeof(float));
    cudaMalloc(&d_C, (size_t)M * D * sizeof(float));

    cudaMemcpy(d_row_ptr, row_ptr.data(), (M+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E.data(), (size_t)M * D * sizeof(float), cudaMemcpyHostToDevice);

    long long total_threads = (long long)M * 32;
    int block = 256;
    int grid = (total_threads + block - 1) / block;
    std::cout << "Launching warp kernels: Grid=" << grid << ", Block=" << block << "\n";

    // === Step 1: SDDMM on GPU ===
    sddmm_csr_warp_kernel<<<grid, block>>>(M, D, d_row_ptr, d_col_idx, d_E, d_vals);
    cudaDeviceSynchronize();

    // Validate SDDMM
    std::vector<float> vals_gpu(nnz);
    cudaMemcpy(vals_gpu.data(), d_vals, nnz * sizeof(float), cudaMemcpyDeviceToHost);
    float sddmm_err = max_abs_err(vals_ref, vals_gpu);
    std::cout << "SDDMM max error = " << sddmm_err << "\n";
    if (sddmm_err < 1e-5)
        std::cout << "SDDMM PASSED\n";
    else
        std::cout << "SDDMM FAILED\n";

    // === Step 2: SpMM on GPU (uses SDDMM output d_vals) ===
    spmm_csr_warp_kernel<<<grid, block>>>(M, D, d_row_ptr, d_col_idx, d_vals, d_E, d_C);
    cudaDeviceSynchronize();

    // Validate SpMM
    std::vector<float> C_gpu((size_t)M * D);
    cudaMemcpy(C_gpu.data(), d_C, (size_t)M * D * sizeof(float), cudaMemcpyDeviceToHost);
    float spmm_err = max_abs_err(C_ref, C_gpu);
    std::cout << "SpMM  max error = " << spmm_err << "\n";
    if (spmm_err < 1e-4)
        std::cout << "SpMM  PASSED\n";
    else
        std::cout << "SpMM  FAILED\n";

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_vals);
    cudaFree(d_E);
    cudaFree(d_C);
    return 0;
}
