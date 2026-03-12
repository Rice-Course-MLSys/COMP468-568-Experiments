
// spmm_ref.cpp
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>
#include <utility>

using float_t = float;

/*
============================================================
 Load matrix in CSR from edge-list text file:
 Each line:   u v
 representing A[u, v] = 1
============================================================
*/
void load_csr_from_edgelist(
    const std::string& filename,
    int& M, int& K,
    std::vector<int>& row_ptr,
    std::vector<int>& col_idx,
    std::vector<float>& vals)
{
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open edge file: " << filename << std::endl;
        exit(1);
    }

    std::vector<std::pair<int,int>> edges;
    edges.reserve(1000000);

    int u, v;
    int max_u = 0, max_v = 0;
    while (fin >> u >> v) {
        edges.emplace_back(u, v);
        if (u > max_u) max_u = u;
        if (v > max_v) max_v = v;
    }
    fin.close();

    M = max_u + 1;
    K = max_v + 1;

    row_ptr.assign(M + 1, 0);

    // Count nnz per row
    for (auto &e : edges)
        row_ptr[e.first + 1]++;

    // Prefix sum
    for (int i = 0; i < M; i++)
        row_ptr[i + 1] += row_ptr[i];

    int nnz = row_ptr.back();
    col_idx.assign(nnz, 0);
    vals.assign(nnz, 1.0f);    // All weights = 1

    // Temporary pointer to fill each row
    std::vector<int> fill(row_ptr.begin(), row_ptr.end());

    for (auto &e : edges) {
        int pos = fill[e.first]++;
        col_idx[pos] = e.second;
    }
}

// CPU reference: SDDMM — for each edge (i, j), compute dot(E[i,:], E[j,:])
void sddmm_cpu(int M, int D,
               const std::vector<int>& row_ptr,
               const std::vector<int>& col_idx,
               const std::vector<float>& E,        // M x D, row-major
               std::vector<float>& vals_out)        // nnz
{
    int nnz = row_ptr[M];
    vals_out.resize(nnz);
    for (int i = 0; i < M; i++) {
        for (int p = row_ptr[i]; p < row_ptr[i+1]; ++p) {
            int j = col_idx[p];
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += E[(size_t)i * D + d] * E[(size_t)j * D + d];
            }
            vals_out[p] = dot;
        }
    }
}

// CPU reference: C = A (CSR) * B (dense)
void spmm_cpu(int M, int K, int N,
              const std::vector<int>& row_ptr,
              const std::vector<int>& col_idx,
              const std::vector<float_t>& vals,
              const std::vector<float_t>& B, // K x N, row-major
              std::vector<float_t>& C) {    // M x N, row-major
    C.assign((size_t)M*N, 0.0f);
    for (int i=0;i<M;i++){
        for (int p = row_ptr[i]; p < row_ptr[i+1]; ++p){
            int k = col_idx[p];
            float_t v = vals[p];
            const float_t* Brow = &B[(size_t)k * N];
            float_t* Crow = &C[(size_t)i * N];
            for (int j=0;j<N;j++){
                Crow[j] += v * Brow[j];
            }
        }
    }
}

// small utility: max abs difference
float_t max_abs_err(const std::vector<float_t>& A, const std::vector<float_t>& B){
    assert(A.size()==B.size());
    float_t mx = 0;
    for (size_t i=0;i<A.size();++i){
        float_t d = std::abs(A[i]-B[i]);
        if (d>mx) mx=d;
    }
    return mx;
}
