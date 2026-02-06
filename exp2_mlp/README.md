## cuBLAS-Powered MLP Mini-Lab

This experiment targets senior undergraduates who already know CUDA fundamentals and linear algebra. Students will build a mini inference engine for fully-connected neural networks (MLPs) by composing cuBLAS GEMM calls and lightweight CUDA activation kernels. The starter harness handles argument parsing, reproducible tensor seeding, and timing, while leaving critical orchestration steps and kernels for students to complete.

### Learning Goals
- Translate MLP layer math into batched GEMM calls using cuBLAS.
- Implement activation and bias-add CUDA kernels that interleave with GEMMs.
- Profile end-to-end throughput, reason about tensor layouts, and validate numerics.
- Compare naive layer-by-layer scheduling against a fused alternative.

### Directory Layout
- `src/main.cu` – command-line interface, memory management, inference loop scaffolding.
- `src/mlp_layers.cuh` – helper structs, cuBLAS wrappers, and CUDA activation stubs (TODOs inside).
- `scripts/measure.sh` – template for sweeping batch sizes / layer widths.
- `Makefile` – builds `bin/dmlp` with CUDA 12+ and links cuBLAS.
- `data/` – create this directory to store CSV timing logs.

### Getting Started
1. Install CUDA 12+; verify `nvcc --version` and `nvidia-smi`.
2. Adjust `ARCH` in `Makefile` to match your GPU's compute capability (e.g., `sm_90`).
3. Build and smoke-test the harness before editing:
   ```bash
   make
   ./bin/dmlp --layers 1024,2048,1024 --batch 128 --impl baseline
   ```
4. Read every TODO inside `src/main.cu` and `src/mlp_layers.cuh`.

### Experiment Tasks
1. **Pre-Lab Math**
   - Derive the FLOP count for one forward pass: `\sum_i 2 * B * n_i * n_{i+1}`.
   - Sketch data movement (bytes) for activations/weights under row-major layout.
2. **Host Orchestration (`src/main.cu`)**
   - Complete the TODOs for allocating device buffers, copying weights/activations, and invoking layer helpers.
   - Implement timing around the forward pass and collect GFLOP/s.
   - Implement the CPU reference (single-threaded) for correctness checking on small batches.
3. **Layer Implementations (`src/mlp_layers.cuh`)**
   - Fill out the `GemmLayer` helper that wraps `cublasSgemm` with column/row major handling.
   - Implement the ReLU / GELU activation kernels and their launch helpers.
   - Add bias-add kernel logic (1 bias vector per layer) and integrate it between GEMM and activation.
4. **Performance Study**
   - Use `scripts/measure.sh` to sweep batch sizes {64,128,256,512} and layer widths {512,1024,2048}.
   - Compare the provided "baseline" schedule (GEMM + bias + activation per layer) with an "activation_fused" version where you inline bias/activation in a single kernel.
   - Plot latency & throughput vs. batch size, and discuss occupancy / memory bandwidth limits.
5. **Stretch Goals (optional)**
   - Add mixed-precision (FP16 accumulation to FP32) via cuBLAS Lt.
   - Implement residual connections or dropout to explore other memory patterns.
   - Integrate TensorRT or PyTorch outputs for reference comparison.

### Deliverables
- Completed CUDA source with TODOs resolved and clearly documented code.
- Measurement logs (CSV) covering all requested sweeps.
- A ≤5-page PDF report summarizing methodology, performance analysis, and key takeaways.

### Rubric (20 pts)
- Correctness (6) – GPU inference matches CPU baseline within 1e-3 absolute error.
- Performance (6) – Achieves ≥70% of theoretical GEMM throughput for largest batch/width.
- Analysis (4) – Insightful discussion on tiling, tensor layout, and cuBLAS configuration.
- Presentation (4) – Clear code organization, plots, and report narrative.

### Suggested Timeline
| Day | Milestone |
|-----|-----------|
| 1   | Build harness, finish pre-lab derivations |
| 2   | Implement cuBLAS layer wrapper + bias kernels |
| 3   | Implement activation kernels + verification path |
| 4   | Run sweeps, collect data, iterate on tuning |
| 5   | Draft report, polish code/comments |

### Make Targets
```bash
make        # build bin/dmlp
make clean  # remove bin/
```

### Academic Integrity
Discuss high-level strategies with classmates, but write your own code/report. Cite any external resources (papers, forums, blog posts) that informed your implementation.
