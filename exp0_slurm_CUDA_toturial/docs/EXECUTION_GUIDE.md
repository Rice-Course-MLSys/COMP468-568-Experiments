# Running CUDA Programs on Slurm

Complete reference for compiling and executing CUDA programs on Slurm-managed HPC clusters.

## Quick Reference

### Build Commands
```bash
make build              # Compile all programs
make clean              # Remove build artifacts
make help               # Show all available commands
```

### Local Testing
```bash
make run_gpu_info       # Test GPU detection
make run_vector         # Test vector addition
make run_matrix         # Test matrix multiplication
```

### Job Submission
```bash
make submit_gpu         # Submit GPU info job
make submit_vector      # Submit vector addition job
make submit_matrix      # Submit matrix multiplication job
make submit_all         # Submit comprehensive tests
make submit_multi_gpu   # Submit multi-GPU tests
```

### Monitoring
```bash
make status             # Check job status
make view_logs          # View recent logs
make cancel_all         # Cancel all your jobs
```

## Detailed Workflows

## Workflow 1: Development and Testing

### Step 1: Initial Compilation
```bash
cd /path/to/project
make clean
make build
```

### Step 2: Local Testing (Login Node)
```bash
# Quick GPU check
./build/gpu_info

# Small vector test
./build/vector_add 1000000

# Small matrix test
./build/matrix_multiply 256
```

### Step 3: Submit to Cluster
```bash
# Submit single test
sbatch scripts/job_gpu_info.sh

# Get job ID from output
# Job submitted with ID 12345

# Monitor job
squeue -j 12345

# View output
tail -f logs/gpu_info_12345.out
```

## Workflow 2: Large-Scale Testing

### Step 1: Prepare Comprehensive Test
```bash
# Edit job script for desired size
vim scripts/job_all_tests.sh

# Update vector/matrix sizes as needed
# Line with: ./build/vector_add 50000000
```

### Step 2: Submit Batch
```bash
# Submit comprehensive test
sbatch scripts/job_all_tests.sh

# Get job ID
# Job submitted with ID 12346
```

### Step 3: Monitor Progress
```bash
# Watch queue
watch -n 5 'squeue -u $USER'

# Tail output in real-time
tail -f logs/all_tests_12346.out
```

### Step 4: Analyze Results
```bash
# Wait for job completion
squeue -j 12346    # Check status

# View full output when done
cat logs/all_tests_12346.out

# Extract performance metrics
grep "Performance:" logs/all_tests_12346.out
```

## Workflow 3: Multi-GPU Scaling

### Step 1: Check Available GPUs
```bash
sinfo -N                 # See node info
nvidia-smi               # Check GPU on current node
```

### Step 2: Submit Multi-GPU Job
```bash
sbatch scripts/job_multi_gpu.sh

# Job will use CUDA_VISIBLE_DEVICES to isolate GPUs
```

### Step 3: Compare Performance
```bash
# View results
cat logs/multi_gpu_*.out

# Compare GPU 0 vs GPU 1 execution times
grep "time:" logs/multi_gpu_*.out
```

## Workflow 4: Interactive Session for Debugging

Interactive sessions allow you to directly debug CUDA code on compute nodes without batch job submission.

### Step 1: Request Interactive Node Access

```bash
# Basic interactive session (1 GPU, 1 hour)
salloc --nodes=1 --ntasks=1 --gpus=1 --time=01:00:00 --partition=gpu

# Expected output:
# salloc: Granted job allocation 12347
# salloc: Waiting for job to start
# [user@compute-node-01 ~]$
```

### Step 2: Compile and Test Interactively

Once you have an interactive shell on the compute node:

```bash
# Load CUDA module
module load cuda/12.0

# Navigate to project
cd /path/to/Slurm_toturial

# Recompile with debug flags (optional)
make clean
make build

# Test GPU immediately
./build/gpu_info

# Run with debugging output
./build/vector_add 10000000

# Run matrix multiply with smaller size for quick iteration
./build/matrix_multiply 256
```

### Step 3: Debug with NVIDIA Tools

Within your interactive session, use profiling and debugging tools:

```bash
# Check GPU status in real-time
nvidia-smi

# Continuous monitoring (refreshes every 1 second)
nvidia-smi -l 1

# Detailed GPU metrics
nvidia-smi dmon -s pucvmet

# Profile your program
nsys profile -o profile1 ./build/vector_add 50000000

# Generate statistics
nsys stats profile1.qdrep
```

### Step 4: Make Code Changes and Test

Edit source code and recompile within the interactive session:

```bash
# Edit source code
vim src/vector_add.cu

# Quick recompile
make clean && make build

# Test immediately without batch queue
./build/vector_add 100000000

# Repeat until satisfied
```

### Step 5: Exit Interactive Session

```bash
# Exit the compute node
exit

# Or use
logout

# To kill session early:
scancel <JOB_ID>
```

### Complete Interactive Debugging Example

```bash
# ============================================
# Step 1: Request interactive access
# ============================================
salloc --nodes=1 --ntasks=1 --gpus=1 --time=02:00:00 --partition=gpu

# ============================================
# Step 2: Setup environment (on compute node)
# ============================================
module load cuda/12.0
cd /home/user/Slurm_toturial

# ============================================
# Step 3: Initial compilation and test
# ============================================
make build
./build/gpu_info                          # Check GPU
./build/vector_add 5000000               # Quick test

# ============================================
# Step 4: Debug and profile
# ============================================
nvidia-smi -l 1                           # Monitor in separate terminal
nsys profile -o profile1 ./build/vector_add 50000000
nsys stats profile1.qdrep                # View results

# ============================================
# Step 5: Modify code based on profiling
# ============================================
# Edit src/vector_add.cu to optimize
vim src/vector_add.cu

# Recompile and test immediately
make rebuild
./build/vector_add 50000000

# ============================================
# Step 6: Verify performance improvement
# ============================================
nsys profile -o profile2 ./build/vector_add 50000000
# Compare profile1 vs profile2

# ============================================
# Step 7: Exit when done
# ============================================
exit
```

### Tips for Effective Interactive Sessions

**Allocate sufficient time:**
```bash
# Longer session for complex debugging
salloc --time=04:00:00 --gpus=1

# Add CPU cores for compilation
salloc --time=02:00:00 --gpus=1 --cpus-per-task=8
```

**Use multiple terminals:**
```bash
# Terminal 1: Run your code
salloc --gpus=1
cd /path/to/project
./build/vector_add 100000000

# Terminal 2: Monitor GPU (from login node)
watch -n 1 nvidia-smi
```

**Create a debug script for fast iteration:**
```bash
# debug.sh - Run on interactive node
#!/bin/bash
make clean && make build
echo "Build complete, running tests..."
./build/gpu_info
./build/vector_add 50000000
./build/matrix_multiply 512
echo "Tests complete!"
```

Then execute:
```bash
chmod +x debug.sh
./debug.sh
```

### Interactive Session vs Batch Job

| Aspect | Interactive | Batch Job |
|--------|-------------|-----------|
| **Setup** | `salloc` | `sbatch` |
| **Feedback** | Immediate | Delayed (logs) |
| **Best for** | Development, debugging | Production runs |
| **Overhead** | Queue time + session time | Just queue time |
| **Resource use** | Charged even when idle | Only charged during execution |
| **Iteration** | Quick edits and tests | Multiple submissions |
| **Learning** | Excellent for learning | Better for reproducibility |

### Common Interactive Session Patterns

**Quick test with GPU:**
```bash
salloc --gpus=1 --time=00:30:00 -N1
module load cuda
cd project
./build/gpu_info && exit
```

**Development with recompilation:**
```bash
salloc --gpus=1 --cpus-per-task=4 --time=01:00:00 -N1
module load cuda
cd project
# Edit, compile, test cycle
vim src/vector_add.cu
make rebuild && ./build/vector_add 10000000
# Repeat as needed
```

**Profiling and optimization:**
```bash
salloc --gpus=1 --time=02:00:00 -N1
module load cuda
nsys profile ./build/matrix_multiply 1024
nsys stats profile.qdrep
# Analyze and optimize
```

## Job Script Customization

### Modify Vector Size
```bash
# Edit job script
vim scripts/job_vector_add.sh

# Change this section:
# echo "Testing with 10M elements..."
# ./build/vector_add 10000000

# To:
# echo "Testing with 500M elements..."
# ./build/vector_add 500000000

# Submit modified job
sbatch scripts/job_vector_add.sh
```

### Adjust Resource Requests
```bash
# For larger matrices, increase memory
#SBATCH --mem=32G        # Increase from 16G

# For more parallelism, increase CPUs
#SBATCH --cpus-per-task=16   # Increase from 8

# Increase time limit if needed
#SBATCH --time=00:30:00   # Increase from 00:15:00
```

### Run Multiple Tests in Parallel
```bash
# Submit multiple different jobs
sbatch scripts/job_vector_add.sh
sbatch scripts/job_matrix_multiply.sh
sbatch scripts/job_gpu_info.sh

# Check all jobs
squeue -u $USER
```

## Performance Analysis

### Capturing Performance Data

**Create analysis script:**
```bash
cat > analyze_performance.sh << 'EOF'
#!/bin/bash

echo "=========================================="
echo "Performance Analysis Report"
echo "=========================================="
echo ""

# Extract timing data
echo "Vector Addition Times:"
grep "execution time:" logs/vector_add_*.out | awk '{print $NF}'

echo ""
echo "Matrix Multiplication Times:"
grep "execution time:" logs/matrix_multiply_*.out | awk '{print $NF}'

echo ""
echo "Throughput Data:"
grep "Throughput\|Performance:" logs/*.out

EOF

chmod +x analyze_performance.sh
./analyze_performance.sh
```

### Comparing Implementations

**Test same size with different block sizes:**

Edit source code, change `BLOCK_SIZE`:
```cuda
// Original
#define BLOCK_SIZE 256

// Test alternatives
#define BLOCK_SIZE 128
#define BLOCK_SIZE 512
```

Recompile and run:
```bash
make clean
make build
sbatch scripts/job_vector_add.sh
```

## Troubleshooting Execution

### Issue: "No CUDA-capable device detected"

**In local test:**
```bash
# GPU not visible on login node
# Try on compute node via job script

sbatch scripts/job_gpu_info.sh
tail logs/gpu_info_*.out
```

### Issue: "CUDA out of memory"

```bash
# Reduce workload size
./build/vector_add 10000000    # Smaller test
./build/matrix_multiply 512    # Smaller matrix

# Or request more GPU memory in Slurm script
#SBATCH --mem-gpu=16G   # GPU memory
```

### Issue: Compilation errors in job

```bash
# Check build in login node first
make clean
make build

# If it compiles locally, issue is environment
# Add to job script:
module load cuda/12.0
```

### Issue: Job times out

```bash
# Reduce workload size
# Increase time limit
#SBATCH --time=01:00:00   # Increase time

# Or both in job script:
./build/matrix_multiply 512    # Smaller matrix
```

### Issue: Segmentation fault in job

```bash
# Check memory requirements
# Reduce workload size significantly

# Add debugging to job script
set -x  # Enable debug output
./build/vector_add 1000000
set +x
```

## Advanced Job Control

### Job Dependencies
```bash
# Submit job 1
JOB1=$(sbatch scripts/job_vector_add.sh | awk '{print $4}')

# Submit job 2 dependent on job 1
sbatch --dependency=afterok:$JOB1 scripts/job_matrix_multiply.sh

# Check dependencies
squeue -j $JOB1 -o "%i %D"
```

### Array Jobs (Multiple Parameters)

**Create array job:**
```bash
cat > job_array_vectors.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=vector_array
#SBATCH --output=logs/vector_array_%a.out
#SBATCH --array=0-3
#SBATCH --gpus=1
#SBATCH --time=00:10:00

SIZES=(1000000 10000000 100000000 1000000000)
./build/vector_add ${SIZES[$SLURM_ARRAY_TASK_ID]}
EOF

sbatch job_array_vectors.sh
```

### Batch Submission
```bash
# Submit all tests at once with delays
sbatch scripts/job_gpu_info.sh
sleep 2
sbatch scripts/job_vector_add.sh
sleep 2
sbatch scripts/job_matrix_multiply.sh

# Check all
squeue -u $USER
```

## Output Management

### Create Structured Logs
```bash
# Create timestamped directory
LOG_DIR="logs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Submit with output redirection
sbatch -o "$LOG_DIR/gpu_info.out" scripts/job_gpu_info.sh
```

### Archive Logs
```bash
# Compress old logs
tar -czf logs/archive_$(date +%Y%m%d).tar.gz logs/*.out logs/*.err

# Keep recent logs
find logs -name "*.out" -mtime +30 -delete
```

### Extract Metrics
```bash
# Extract all execution times
grep -h "execution time" logs/*.out | sort

# Extract performance data
grep -h "GFLOP" logs/*.out | awk '{sum += $NF} END {print "Average GFLOP/s:", sum/NR}'
```

## Performance Monitoring During Execution

### Real-time GPU Monitoring
```bash
# On compute node during job
nvidia-smi -l 1           # Refresh every 1 second
nvidia-smi dmon -s pucvmet  # Detailed metrics
```

### Queue Monitoring
```bash
# Watch queue status
watch -n 2 'squeue -u $USER'

# Specific job details
scontrol show job <JOB_ID>

# Resource utilization
sinfo -R             # Show reasons for down nodes
```

## Scripting Workflows

### Automated Testing Script

```bash
#!/bin/bash
# test_suite.sh

SIZES=(1000000 10000000 100000000)
MATRICES=(256 512 1024)

echo "Starting automated test suite..."

# Vector tests
for size in "${SIZES[@]}"; do
    echo "Submitting vector test with size $size"
    sbatch --job-name="vec_$size" scripts/job_vector_add.sh $size
done

# Matrix tests
for size in "${MATRICES[@]}"; do
    echo "Submitting matrix test with size ${size}x${size}"
    sbatch --job-name="mat_${size}" scripts/job_matrix_multiply.sh $size
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
```

## Performance Optimization Workflow

### Step 1: Baseline
```bash
sbatch scripts/job_all_tests.sh
# Record execution times
```

### Step 2: Profile
```bash
# Submit with profiling
nsys profile -o profile.qdrep ./build/vector_add 100000000

# Analyze
nsys stats profile.qdrep
```

### Step 3: Optimize
```bash
# Modify source code (change BLOCK_SIZE, tile size, etc.)
make clean
make build
```

### Step 4: Compare
```bash
sbatch scripts/job_all_tests.sh
# Compare new times vs baseline
```

## Common Slurm Query Commands

```bash
# All your jobs
squeue -u $USER

# Specific job status
squeue -j <JOB_ID>

# Show job script of running job
scontrol write batch_script <JOB_ID> -

# Account statistics
sacct --user=$USER --format=JobID,Start,End,Elapsed,State,ExitCode

# Node status
sinfo -N

# Detailed node info
scontrol show node <NODE_NAME>

# Partition info
sinfo -p <PARTITION>
```

---

**See Also**: README.md, SETUP_GUIDE.md, GPU_PROGRAMMING_GUIDE.md
