# CUDA GPU Programming Fundamentals

Reference guide for understanding and optimizing CUDA programs included in this project.

## CUDA Programming Basics

### What is CUDA?

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that allows:
- Executing code on NVIDIA GPUs
- Massive parallelism (thousands of threads)
- High performance computing (TFLOP/s to PFLOP/s)
- Direct C/C++ programming

### GPU vs CPU Architecture

| Aspect | CPU | GPU |
|--------|-----|-----|
| Cores | 8-64 | 1000s-10000s |
| Memory | Large (GB) | Medium (GB) |
| Latency | Low | Higher |
| Throughput | Moderate | Very High |
| Power | High | Variable |
| Best for | Sequential | Data parallel |

## CUDA Execution Model

### Threads, Blocks, and Grids

**Threads**: Individual processing units
- Run same code (SIMD)
- Have local registers
- Can synchronize within block

**Blocks**: Groups of threads
- Share memory (shared memory)
- Can synchronize
- Execute independently
- Max 1024 threads per block (typically)

**Grids**: Collections of blocks
- Execute same kernel
- Can span multiple GPUs
- No synchronization between blocks

### Memory Hierarchy

```
GPU Memory Hierarchy (fastest → slowest):
├── Registers (per thread)     - Very fast, limited
├── Shared Memory (per block)  - Fast, limited (48-96 KB)
├── L1/L2 Cache               - Automatic caching
├── Global Memory             - Large (10-80 GB), slower
└── Host (CPU) Memory         - Largest, slowest for GPU
```

## CUDA Programs in This Project

### 1. Vector Addition - Memory Coalescing Example

**File**: `src/vector_add.cu`

**What it does**: Computes $C[i] = A[i] + B[i]$ for all elements

**Key Concepts**:

```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate index
    
    if (idx < n) {
        C[idx] = A[idx] + B[idx];  // One thread per element
    }
}
```

**Thread Index Calculation**:
$$\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

Where:
- `blockIdx.x`: Block index in grid
- `blockDim.x`: Threads per block (256)
- `threadIdx.x`: Thread index within block

**Memory Access Pattern** (Coalesced):
```
Thread 0 accesses A[0], B[0], C[0]
Thread 1 accesses A[1], B[1], C[1]
Thread 2 accesses A[2], B[2], C[2]
...
```
→ Sequential memory addresses = coalesced = efficient!

**Kernel Launch**:
```cuda
int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Ceiling division
vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, size);
```

**Execution Flow**:
1. Allocate GPU memory
2. Copy data to GPU
3. Launch kernel (gridSize blocks of 256 threads each)
4. Each thread computes one addition
5. Copy results back
6. Verify on CPU

### 2. Matrix Multiplication - Shared Memory Optimization

**File**: `src/matrix_multiply.cu`

**What it does**: Computes $C = A \times B$ using tiled algorithm

**Naive Approach Problem**:
- For $1024 \times 1024$ matrices
- Each element requires 1024 multiplications
- = 1,073,741,824 global memory accesses
- Global memory: ~200-300 GB/s
- This is slow!

**Tiled Approach Solution**:

Use shared memory to cache data:

```cuda
__shared__ float tile_A[TILE_SIZE][TILE_SIZE];  // 32x32 on-chip cache
__shared__ float tile_B[TILE_SIZE][TILE_SIZE];  // 32x32 on-chip cache

// Load data into shared memory
tile_A[threadIdx.y][threadIdx.x] = A[row * size + col];
__syncthreads();  // All threads in block wait

// Compute using cached data (much faster!)
sum += tile_A[...] * tile_B[...];
```

**Why Tiling Works**:
- Shared memory: ~10 TB/s (vs ~300 GB/s global)
- Reduces global memory traffic
- Increases cache hit rate
- Better GPU utilization

**Algorithm**:
1. Divide matrices into $32 \times 32$ tiles
2. Loop over tiles in K dimension
3. For each tile:
   - Load into shared memory
   - Compute partial products
   - Accumulate results
4. Write final result

**Performance**:
- Bandwidth: 1TB/s+ (vs 300GB/s naive)
- Computation: GFLOP/s scales with GPU

### 3. GPU Information Utility

**File**: `src/gpu_info.cu`

**What it does**: Query and display GPU capabilities

**Key Information Retrieved**:

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device_id);

// Compute capability (architecture version)
prop.major, prop.minor  // e.g., 7.0 = Volta, 8.0 = Ampere

// Memory
prop.totalGlobalMem     // Total GPU memory
prop.sharedMemPerBlock  // Per-block cache

// Threading
prop.maxThreadsPerBlock // Max threads per block
prop.maxThreadsDim      // Max threads per dimension

// Performance
prop.clockRate          // GPU clock speed
prop.multiProcessorCount  // Number of SMs
```

## Performance Concepts

### Memory Bandwidth

**Formula**:
$$\text{Throughput (GB/s)} = \frac{\text{Data Transferred (GB)}}{\text{Time (s)}}$$

**For Vector Addition**:
- 3 arrays × 4 bytes × 1M elements = 12 MB
- If kernel takes 1 ms: 12,000 MB/s = 12 GB/s

### FLOP/s (Floating Point Operations per Second)

**For Matrix Multiplication**:
$$\text{FLOP/s} = \frac{2 \times N^3}{\text{Time (s)}}$$

Where $N \times N$ is matrix size

**For 1024×1024 matrix in 100 ms**:
$$\text{GFLOP/s} = \frac{2 \times 1024^3}{0.1} \approx 2.1 \text{ TFLOP/s}$$

### Achieved vs Theoretical Peak

**Tesla V100**:
- Peak: ~7 TFLOP/s (FP32)
- Typical achieved: 60-80% = 4-5.5 TFLOP/s

**Factors affecting percentage**:
- Memory bandwidth utilization
- Shared memory efficiency
- Register pressure
- Instruction dependencies

## Optimization Techniques

### 1. Coalesced Memory Access

**Good Pattern** (Accesses sequential memory):
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
data[idx]  // Thread 0→0, Thread 1→1, ...
```
→ Memory access: 0,1,2,3,... (COALESCED)

**Bad Pattern** (Accesses scattered memory):
```cuda
data[idx * 32]  // Thread 0→0, Thread 1→32, Thread 2→64, ...
```
→ Memory access: 0,32,64,... (SCATTERED)

### 2. Shared Memory Optimization

**Avoid Bank Conflicts**:
```cuda
// Bank conflict: all threads access same bank
__shared__ float data[32];
float x = data[threadIdx.x];  // All threads: same bank

// No conflict: threads access different banks
float x = data[threadIdx.x];  // Thread 0→bank 0, Thread 1→bank 1
```

### 3. Thread Block Size

**Typical Sizes**:
- 128-256 threads: Good balance
- 512 threads: For memory-heavy kernels
- 1024 threads: Maximum, may reduce registers per thread

**Selection**: Start with 256, adjust based on profiling

### 4. Occupancy

**Occupancy** = (Actual threads) / (Max possible threads)

Higher occupancy → Better latency hiding

**Factors**:
- Registers per thread
- Shared memory per block
- Max threads per SM

## Hands-On Optimization Workflow

### Step 1: Measure Baseline

```bash
# Compile and run
make build
./build/vector_add 100000000
# Record: execution time, throughput
```

### Step 2: Identify Bottleneck

```bash
# Profile with NVIDIA tools
nsys profile ./build/vector_add 100000000

# Check results:
# - GPU time vs CPU time
# - Memory utilization
# - Kernel efficiency
```

### Step 3: Optimize

**Common optimizations**:
1. Increase block size (if register pressure allows)
2. Use shared memory to cache data
3. Ensure coalesced memory access
4. Minimize synchronization

### Step 4: Measure Improvement

```bash
make clean && make build
./build/vector_add 100000000
# Compare with baseline
```

## CUDA Error Handling

**Best Practice**:
```cuda
cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    return 1;
}
```

**Common Errors**:
- `invalid device ordinal` - GPU not found
- `out of memory` - GPU memory exceeded
- `invalid argument` - Wrong parameters
- `device not found` - No CUDA GPU

## Further Reading

### NVIDIA Documentation
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)

### Optimization
- [NVIDIA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Profiling Guide](https://docs.nvidia.com/cuda/profiling-guide/)

### Resources
- [NVIDIA DevBlog](https://developer.nvidia.com/blog/)
- CUDA by Example textbook
- GPU Gems series

## Quick Reference: Common CUDA Functions

| Function | Purpose |
|----------|---------|
| `cudaMalloc()` | Allocate GPU memory |
| `cudaFree()` | Deallocate GPU memory |
| `cudaMemcpy()` | Copy data host ↔ device |
| `cudaMemset()` | Set GPU memory value |
| `cudaDeviceReset()` | Reset GPU state |
| `cudaEventCreate()` | Create event timer |
| `cudaEventRecord()` | Record timestamp |
| `cudaEventElapsedTime()` | Get elapsed time |

---

**See Also**: README.md, src/vector_add.cu, src/matrix_multiply.cu
