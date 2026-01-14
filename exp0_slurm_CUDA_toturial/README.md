# CUDA + Slurm Cluster Experiment

Comprehensive tutorial and experimental framework for compiling and running CUDA programs on Slurm-based HPC clusters.

## Project Overview

This project provides a complete setup for developing, compiling, and executing CUDA applications on high-performance computing (HPC) clusters managed by Slurm. It includes:

- **CUDA Source Programs**: Vector addition, matrix multiplication with optimizations, and GPU information utility
- **Slurm Job Scripts**: Pre-configured batch scripts for various testing scenarios
- **Build System**: Makefile for easy compilation and deployment
- **Documentation**: Complete guides for setup and execution

## Project Structure

```
Slurm_toturial/
├── src/
│   ├── vector_add.cu              # CUDA vector addition program
│   ├── matrix_multiply.cu         # CUDA matrix multiplication with tiling
│   └── gpu_info.cu                # GPU information utility
├── scripts/
│   ├── job_gpu_info.sh            # Slurm job for GPU info
│   ├── job_vector_add.sh          # Slurm job for vector addition
│   ├── job_matrix_multiply.sh     # Slurm job for matrix multiplication
│   ├── job_all_tests.sh           # Comprehensive test job
│   ├── job_multi_gpu.sh           # Multi-GPU test job
│   └── launch_interactive.sh      # Interactive session launcher
├── build/                         # Compiled executables (created at build time)
├── logs/                          # Job output and error logs (created at runtime)
├── docs/                          # Documentation files
│   ├── SETUP_GUIDE.md             # General setup and installation
│   ├── EXECUTION_GUIDE.md         # Running jobs on generic Slurm clusters
│   ├── GPU_PROGRAMMING_GUIDE.md   # CUDA concepts and optimization
│   └── NOTS_SETUP.md              # Rice NOTS cluster-specific guide
├── Makefile                       # Build configuration
└── README.md                      # This file

## Prerequisites

### System Requirements
- Linux-based HPC cluster with Slurm workload manager
- NVIDIA GPU(s) with CUDA compute capability 3.0 or higher
- CUDA Toolkit 10.0 or newer installed on cluster nodes
- GCC/G++ compiler compatible with CUDA

### For Rice NOTS Cluster Users
This project has been adapted for the Rice NOTS cluster. See `docs/NOTS_SETUP.md` for NOTS-specific configuration including:
- Partition names: debug, commons, long, scavenge
- GPU types: volta (V100), ampere (A40), lovelace (L40S), h100, h200
- Required account specification: `--account=commons`
- Memory specification format: `--mem-per-cpu`

### Required Modules
Load these modules on your cluster:
```bash
module load cuda                # NOTS automatically handles versioning
module load gcc                 # C++ compiler
module load slurm               # Slurm (usually pre-loaded)
```

### Local Environment
- Bash shell for executing scripts
- Basic UNIX utilities (mkdir, grep, etc.)

## Quick Start

### 1. Compile Programs

```bash
# Compile all programs
make compile

# Compile individual programs
make $(BUILD_DIR)/vector_add
make $(BUILD_DIR)/matrix_multiply
make $(BUILD_DIR)/gpu_info
```

### 2. View Build Options

```bash
make help
```

### 3. Submit Jobs to Slurm

```bash
# Submit GPU information job
make submit_gpu

# Submit vector addition job
make submit_vector

# Submit matrix multiplication job
make submit_matrix

# Submit comprehensive test
make submit_all

# Submit multi-GPU test
make submit_multi_gpu
```

### 4. Monitor Jobs

```bash
# Check job status
make status

# View recent logs
make view_logs

# Cancel all your jobs
make cancel_all
```

## Building and Compilation

### Prerequisites for Compilation

Before compiling, ensure:
1. NVIDIA CUDA Toolkit is installed
2. Correct GPU architecture is specified in Makefile
3. Build directory will be created automatically

### GPU Architecture Selection

Edit `CUDA_ARCH` in the Makefile for your GPU:

| GPU | Architecture | Code |
|-----|-------------|------|
| Tesla V100 | Volta | `-arch=sm_70` |
| Tesla A100 | Ampere | `-arch=sm_80` |
| RTX 3090 | Ampere | `-arch=sm_86` |
| RTX 4090 | Ada | `-arch=sm_89` |
| Tesla K40 | Kepler | `-arch=sm_35` |

### Compilation Commands

```bash
# Full build with all optimizations
make compile

# Clean and rebuild
make clean && make compile

# Quick local test (without Slurm)
make run_vector
make run_matrix
make run_gpu_info
```

## Slurm Job Submission

### Basic Job Submission

```bash
# Submit and get job ID
sbatch scripts/job_vector_add.sh

# View job details
squeue -j <JOB_ID>

# Cancel job
scancel <JOB_ID>
```

### Job Script Parameters

Key Slurm directives in job scripts:

- `--job-name`: Name for the job
- `--output`: Standard output file (logs/job_name_%j.out)
- `--error`: Standard error file (logs/job_name_%j.err)
- `--nodes`: Number of compute nodes (1 for single-node)
- `--ntasks`: Number of tasks
- `--cpus-per-task`: CPUs per task
- `--gpus`: Number of GPUs requested
- `--mem`: Memory per node
- `--time`: Maximum job runtime (HH:MM:SS)
- `--partition`: Queue name (adjust for your cluster)

### Example Job Configurations

#### Single GPU, Single Task
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
```

#### Multiple GPUs
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --time=00:20:00
```

#### High Memory Requirements
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
```

## CUDA Programs

### 1. GPU Information (`gpu_info.cu`)

Displays comprehensive GPU hardware information.

**Features**:
- Device count
- GPU name and compute capability
- Memory configuration
- Performance specifications
- Threading capabilities

**Execution**:
```bash
./build/gpu_info
```

**Output includes**:
- GPU model name
- Compute capability (architecture)
- Total global memory
- Shared memory per block
- Max threads per block
- Clock rates
- Cache information

### 2. Vector Addition (`vector_add.cu`)

Performs element-wise addition of two large floating-point vectors on GPU.

**Features**:
- Configurable vector size
- Memory performance metrics
- Verification of results
- Timing using CUDA events

**Execution**:
```bash
# Default size (1M elements)
./build/vector_add

# Custom size
./build/vector_add 100000000
```

**Performance Output**:
- Kernel execution time (ms)
- Throughput (GB/s)
- Verification status

**Algorithm**:
- Each thread adds one pair of elements
- Block size: 256 threads
- Memory pattern: Coalesced memory access

### 3. Matrix Multiplication (`matrix_multiply.cu`)

Multiplies two square matrices using tiled shared memory optimization.

**Features**:
- Tiled matrix multiplication algorithm
- Automatic size alignment to tile boundaries
- Shared memory optimization
- Sample-based verification
- Performance metrics in GFLOP/s

**Execution**:
```bash
# 512x512 matrix
./build/matrix_multiply 512

# 1024x1024 matrix
./build/matrix_multiply 1024

# 2048x2048 matrix
./build/matrix_multiply 2048
```

**Algorithm Details**:
- Tile size: 32×32 elements
- Shared memory for caching
- Optimized memory access patterns
- Computes: $C = A \times B$ where $A, B, C$ are $n \times n$ matrices

**Performance Output**:
- Kernel execution time (ms)
- Total FLOP count
- Sustained performance (GFLOP/s)

## Advanced Usage

### Running on Multi-GPU Systems

```bash
# Specify GPU device
CUDA_VISIBLE_DEVICES=0 ./build/vector_add 100000000

# Run on GPU 1
CUDA_VISIBLE_DEVICES=1 ./build/vector_add 100000000
```

### Profiling with NVIDIA Tools

```bash
# Profile kernel execution
nsys profile ./build/vector_add 50000000

# Check utilization
nvidia-smi dmon -s pucvmet
```

### Custom Compilation Flags

Edit the Makefile to add flags:

```makefile
# Add debug info
CFLAGS := -O3 -std=c++11 -g

# Add profiling
CFLAGS := -O3 -std=c++11 -lineinfo

# Verbose warnings
CFLAGS := -O3 -std=c++11 -Wall -Wextra
```

## Troubleshooting

### Issue: "No CUDA-capable device detected"

**Solution**: 
```bash
# Check GPU availability
nvidia-smi

# Run with CPU emulation (limited, for testing)
./build/gpu_info
```

### Issue: "CUDA out of memory"

**Solution**:
- Reduce vector/matrix size in job script
- Request more GPU memory via `--mem-gpu` in Slurm script
- Use smaller test cases first

### Issue: Job queuing for long time

**Solution**:
```bash
# Check queue status
sinfo

# Submit to faster queue
sbatch --partition=gpu_short scripts/job_vector_add.sh

# Check resource availability
sinfo -N
```

### Issue: Compilation errors

**Solution**:
```bash
# Update CUDA arch in Makefile
make clean && make compile

# Check CUDA installation
which nvcc
nvcc --version
```

### Issue: Permission denied on scripts

**Solution**:
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Re-submit
sbatch scripts/job_vector_add.sh
```

## Monitoring and Debugging

### Real-time Job Monitoring

```bash
# Watch job status
watch -n 2 'squeue -u $USER'

# Detailed job info
scontrol show job <JOB_ID>
```

### View Job Output

```bash
# Standard output
cat logs/vector_add_<JOB_ID>.out

# Error output
cat logs/vector_add_<JOB_ID>.err

# Real-time tail
tail -f logs/vector_add_*.out
```

### System Information During Job

```bash
# GPU utilization
nvidia-smi

# Memory usage
free -h

# CPU usage
top
```

## Performance Optimization

### Memory Optimization

1. **Coalesced memory access**: Sequential threads access sequential memory
2. **Shared memory usage**: Cache frequently accessed data
3. **Memory alignment**: Ensure proper alignment for vector operations

### Kernel Optimization

1. **Block size tuning**: Test 128, 256, 512 threads per block
2. **Grid size**: Ensure sufficient blocks to utilize all SMs
3. **Shared memory**: Minimize bank conflicts

### Compilation Optimization

```makefile
# Maximum optimization
CFLAGS := -O3 --use_fast_math

# Additional optimizations
CFLAGS := -O3 --generate-line-info --source-in-ptx
```

## Common Slurm Commands Reference

```bash
# Submit job
sbatch script.sh

# Check job status
squeue -u <username>
squeue -j <job_id>

# Detailed job info
sinfo -N
sinfo -p <partition>

# Job statistics
sacct -j <job_id>
sacct --format=JobID,Start,End,Elapsed,State,ExitCode

# Cancel job
scancel <job_id>
scancel -u <username>           # Cancel all user jobs
scancel --state=PENDING -u <u>  # Cancel pending jobs

# Resource limits
slurm_load
slurm_limits
```

## Environment Module Management

```bash
# List available modules
module avail

# Load CUDA module
module load cuda

# Check loaded modules
module list

# Unload module
module unload cuda

# Purge all modules
module purge
```

## Batch Job Array Execution

Submit multiple jobs with different parameters:

```bash
# Create array job script
cat > job_array.sh << 'EOF'
#!/bin/bash
#SBATCH --array=0-3
#SBATCH --gpus=1

SIZES=(1000000 10000000 100000000 1000000000)
./build/vector_add ${SIZES[$SLURM_ARRAY_TASK_ID]}
EOF

# Submit
sbatch job_array.sh
```

## Performance Analysis

### Measuring Performance

Run tests with timing:
```bash
# Sequential tests
time ./build/vector_add 100000000
time ./build/matrix_multiply 1024
```

### Expected Performance Ranges

**Vector Addition** (on modern GPU):
- 1M elements: < 1 ms
- 100M elements: 10-50 ms
- Throughput: 100-600 GB/s

**Matrix Multiplication**:
- 1024×1024: 50-200 ms
- Performance: 100-800 GFLOP/s
- Varies by GPU architecture

## Documentation Structure

- `README.md`: This file - overview and quick start
- `docs/`: Additional documentation
- `Makefile`: Build configuration with comments
- `scripts/`: Job scripts with inline documentation
- `src/`: Source code with detailed comments

## Citation and References

If using this tutorial in academic work, please reference:
- NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- Slurm Workload Manager: https://slurm.schedmd.com/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## License

This project is provided as-is for educational and research purposes.

## Support and Contributing

For issues, questions, or contributions:
1. Check existing documentation
2. Review Troubleshooting section
3. Consult HPC cluster documentation
4. Contact cluster administrator

## Frequently Asked Questions

**Q: How do I check if my GPU supports CUDA?**
A: Run `nvidia-smi` and check compute capability in the output.

**Q: Can I run CUDA programs without Slurm?**
A: Yes, compile with `make compile` and run executables directly (if GPU available).

**Q: How do I submit jobs from a login node?**
A: Use `sbatch script.sh` from the cluster login node.

**Q: What's the difference between GPU and host memory?**
A: GPU memory is on-device (fast, limited ~10-80GB), host memory is system RAM.

**Q: How can I profile my CUDA application?**
A: Use `nsys profile` or `nvprof` for detailed performance analysis.

**Q: I'm on Rice NOTS - what partition should I use?**
A: Start with `debug` (30 min) for testing, use `commons` for production (24h).

**Q: How do I request a specific GPU type on NOTS?**
A: Use `--gres=gpu:volta:1` for Tesla V100, `--gres=gpu:ampere:1` for A40, etc.

**Q: What's the difference between `--mem` and `--mem-per-cpu`?**
A: On NOTS, use `--mem-per-cpu=2G` (per-CPU allocation, total = cpus × mem-per-cpu).

---

**Last Updated**: November 2025
**Version**: 2.0 (Rice NOTS adapted)
**Compatible Clusters**: Generic Slurm, Rice NOTS
