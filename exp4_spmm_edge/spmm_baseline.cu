
// spmm_baseline.cu — Two-Step GNN: SDDMM + SpMM (STUDENT SKELETON)
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

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

using float_t = float;

/*
===============================================================
 SDDMM BASELINE KERNEL — STUDENT TODO
 One thread per nonzero edge.
 row_indices[p] gives the source row for edge p.
===============================================================
*/
__global__ void sddmm_csr_baseline_kernel(
    int nnz, int D,
    const int* __restrict__ d_row_indices, // nnz: row index for each edge
    const int* __restrict__ d_col_idx,     // nnz: column index for each edge
    const float_t* __restrict__ d_E,       // M x D embedding matrix
    float_t* __restrict__ d_vals)          // nnz: output edge weights
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz) return;

    // TODO student: read source row i from d_row_indices[p]
    // TODO student: read destination column j from d_col_idx[p]
    // TODO student: compute dot product of E[i,:] and E[j,:] over D dimensions
    // TODO student: write result to d_vals[p]
}

/*
===============================================================
 SpMM BASELINE KERNEL — STUDENT TODO
 One thread per row of the output matrix.
 Computes C[row,:] = sum over nonzeros in row of val * E[col,:]
===============================================================
*/
__global__ void spmm_csr_row_kernel(
    int M, int N,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float_t* __restrict__ d_vals,
    const float_t* __restrict__ d_B,
    float_t* __restrict__ d_C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // TODO student: init output row
    // TODO student: load start, end
    // TODO student: loop over p in row nnz
    // TODO student: accumulate into d_C[row*N + j]
}

int main(int argc, char** argv) {
    int M, K;
    const int D = 64;  // embedding dimension

    std::vector<int> row_ptr, col_idx;
    std::vector<float> vals;

    load_csr_from_edgelist("graph_edges.txt", M, K, row_ptr, col_idx, vals);
    int nnz = row_ptr.back();
    assert(M == K && "Adjacency matrix must be square");

    std::cout << "Loaded graph: M=" << M << " nnz=" << nnz << " D=" << D << "\n";

    // --- Generate random embedding E (M x D) ---
    std::vector<float> E((size_t)M * D);
    srand(42);
    for (size_t i = 0; i < E.size(); i++) E[i] = float(rand()) / RAND_MAX;

    // --- Build row_indices array (for baseline SDDMM kernel) ---
    std::vector<int> row_indices(nnz);
    for (int i = 0; i < M; i++)
        for (int p = row_ptr[i]; p < row_ptr[i+1]; p++)
            row_indices[p] = i;

    // === CPU Reference ===
    // Step 1: SDDMM — compute edge weights
    std::vector<float> vals_ref;
    sddmm_cpu(M, D, row_ptr, col_idx, E, vals_ref);

    // Step 2: SpMM — C = A_weighted * E
    std::vector<float> C_ref;
    spmm_cpu(M, M, D, row_ptr, col_idx, vals_ref, E, C_ref);

    // === GPU Setup ===
    int *d_row_ptr, *d_col_idx, *d_row_indices;
    float *d_vals, *d_E, *d_C;
    cudaMalloc(&d_row_ptr, (M+1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_row_indices, nnz * sizeof(int));
    cudaMalloc(&d_vals, nnz * sizeof(float));
    cudaMalloc(&d_E, (size_t)M * D * sizeof(float));
    cudaMalloc(&d_C, (size_t)M * D * sizeof(float));

    cudaMemcpy(d_row_ptr, row_ptr.data(), (M+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, row_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E.data(), (size_t)M * D * sizeof(float), cudaMemcpyHostToDevice);

    // === Step 1: SDDMM on GPU ===
    {
        int block = 256;
        int grid = (nnz + block - 1) / block;
        sddmm_csr_baseline_kernel<<<grid, block>>>(nnz, D, d_row_indices, d_col_idx, d_E, d_vals);
        cudaDeviceSynchronize();
    }

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
    {
        int block = 256;
        int grid = (M + block - 1) / block;
        spmm_csr_row_kernel<<<grid, block>>>(M, D, d_row_ptr, d_col_idx, d_vals, d_E, d_C);
        cudaDeviceSynchronize();
    }

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
    cudaFree(d_row_indices);
    cudaFree(d_vals);
    cudaFree(d_E);
    cudaFree(d_C);
    return 0;
}
