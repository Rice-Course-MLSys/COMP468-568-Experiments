
# Two-Step GNN Computation: SDDMM + SpMM

This experiment implements the two core sparse operations in Graph Neural Networks (GNNs):

1. **SDDMM (Sampled Dense-Dense Matrix Multiplication)**: For each edge (i, j) in the graph, compute the dot product of node embeddings `dot(E[i,:], E[j,:])` to produce edge weights. This mirrors the attention score computation in Graph Attention Networks (GAT).

2. **SpMM (Sparse Matrix-Matrix Multiplication)**: Multiply the weighted adjacency matrix by the embedding matrix `C = A_weighted × E` to produce new node embeddings. This is the message aggregation step.

Together, these form a complete GNN message-passing layer:
```
E (M×D embedding) + A (M×M adjacency, CSR)
        ↓ SDDMM
A_weighted (M×M, vals = dot products)
        ↓ SpMM  (A_weighted × E)
C (M×D new embeddings)
```

## Input Format: `graph_edges.txt`
Each line:
```
u v
```
means there is an edge from node u to node v (A[u, v] = 1).

## Build
```
make
```

## Run
```
./spmm_baseline
./spmm_opt
```

## Tasks

### Task 1: SDDMM Baseline Kernel (`spmm_baseline.cu`)
- Implement `sddmm_csr_baseline_kernel`: one thread per nonzero edge.
- Each thread reads its source row from `row_indices[p]` and column from `col_idx[p]`.
- Computes the dot product `dot(E[row,:], E[col,:])` over D dimensions.
- Writes the result to `d_vals[p]`.

### Task 2: SpMM Baseline Kernel (`spmm_baseline.cu`)
- Implement `spmm_csr_row_kernel`: one thread per row.
- Each thread iterates over the nonzeros in its row, accumulating `val * E[col,:]` into `C[row,:]`.

### Task 3: SDDMM Optimized Kernel (`spmm_opt.cu`)
- Implement `sddmm_csr_warp_kernel`: one warp (32 threads) per row.
- Lanes split the nonzero edges within the row — each lane computes the full dot product for its assigned edges.
- No `row_indices` array needed since the warp already knows its row.

### Task 4: SpMM Optimized Kernel (`spmm_opt.cu`)
- Implement `spmm_csr_warp_kernel`: one warp per row.
- Lanes split across output columns (lane handles columns `lane, lane+32, lane+64, ...`).
- Each lane accumulates contributions from all nonzeros in the row.

### Validation
- SDDMM: verified against CPU reference (max error < 1e-5).
- SpMM: verified against CPU reference (max error < 1e-4, due to accumulated floating-point error).

## Deliverables
- Completed CUDA source files with all TODOs resolved.
- A brief report (1-2 pages) including:
  - Explanation of your implementation approach for each kernel.
  - Clear plots showing performance comparison (e.g., speedup of warp kernels vs baseline).
  - Discussion of acceleration results observed.

## Rubric (20 pts)
- **Correctness (10)** — All four kernels produce results matching the CPU reference within tolerance.
- **Performance (5)** — The optimized (warp-based) kernels demonstrate a relative improvement over the baseline (naive) versions. No hard roofline threshold is required — just show a measurable speedup.
- **Report (5)** — Clear explanation of implementation, plots showing performance comparison between baseline and optimized kernels, and discussion of results.

## Academic Integrity
You may discuss ideas with classmates, but all code and report content must be your own.
