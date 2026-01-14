# Rice NOTS Cluster Setup Guide

This document provides Rice University NOTS cluster-specific configuration for running CUDA programs.

## Quick Reference for NOTS

### Cluster Access
```bash
# SSH to NOTS login node
ssh netID@nots.rice.edu

# Check your account access
sacctmgr show assoc user=$USER
```

### Available Partitions

| Partition | Max Time | Use Case | Priority |
|-----------|----------|----------|----------|
| **debug** | 30 min | Debugging, quick tests | High |
| **commons** | 24 hours | General compute jobs | Medium |
| **long** | 72 hours | Long-running jobs | Low |
| **scavenge** | 1 hour | Idle resources (may backfill) | High backfill |

### Available GPU Types on NOTS

| GPU Type | Model | Memory | Compute Cap |
|----------|-------|--------|-------------|
| volta | Tesla V100 | 32 GB | 7.0 |
| ampere | Tesla A40 | 48 GB | 8.6 |
| lovelace | Tesla L40S | 48 GB | 8.9 |
| h100 | Hopper H100 | 80 GB | 9.0 |
| h200 | Hopper H200 | 141 GB | 9.0 |

## Project Setup for NOTS

### Step 1: Clone Project to NOTS

```bash
# SSH into NOTS
ssh netID@nots.rice.edu

# Navigate to shared scratch (recommended for projects)
cd $SHARED_SCRATCH/$USER

# Clone or copy your project
git clone https://github.com/your-repo/Slurm_toturial.git
cd Slurm_toturial
```

### Step 2: Verify CUDA Installation

```bash
# Load CUDA module
module load cuda

# Check CUDA installation
nvcc --version
nvidia-smi
```

### Step 3: Update Makefile for NOTS GPU

Edit `Makefile` and set correct GPU architecture:

```makefile
# For Tesla V100 (Volta)
CUDA_ARCH := -arch=sm_70

# For Tesla A40 (Ampere)
CUDA_ARCH := -arch=sm_86

# For H100 (Hopper)
CUDA_ARCH := -arch=sm_90
```

### Step 4: Build Programs

```bash
# Load CUDA
module load cuda

# Compile
make build

# Verify executables
ls -lh build/
```

## Job Submission Examples

### Debug Session (Quick Testing)

```bash
# Fast iterative development
./scripts/launch_interactive.sh

# Or specify GPU type
./scripts/launch_interactive.sh --gpu-type=volta
```

### Submit Vector Addition Job

```bash
sbatch scripts/job_vector_add.sh
```

**Job script will use:**
- Partition: commons
- GPU: 1 Tesla V100
- Time: 30 minutes
- Memory: 4 GB total (2 GB per CPU)

### Submit Matrix Multiplication Job

```bash
sbatch scripts/job_matrix_multiply.sh
```

### Run All Tests

```bash
sbatch scripts/job_all_tests.sh
```

### Multi-GPU Job

```bash
sbatch scripts/job_multi_gpu.sh
```

## NOTS-Specific Features

### Job Submission Required Directives

All jobs **must** include:
```bash
#SBATCH --account=commons
#SBATCH --partition=commons  # or debug, long, scavenge
```

### GPU Request Format on NOTS

```bash
# Request any GPU
#SBATCH --gres=gpu:1

# Request specific GPU type
#SBATCH --gres=gpu:volta:1      # Tesla V100
#SBATCH --gres=gpu:ampere:1     # Tesla A40
#SBATCH --gres=gpu:lovelace:1   # Tesla L40S
#SBATCH --gres=gpu:h100:1       # Hopper H100
#SBATCH --gres=gpu:h200:1       # Hopper H200

# Request multiple GPUs
#SBATCH --gres=gpu:2            # Any GPU type
#SBATCH --gres=gpu:volta:2      # Specific type
```

### Memory Specification on NOTS

NOTS uses **per-CPU memory** rather than total node memory:

```bash
# Correct for NOTS (per-CPU)
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G    # Total: 8GB

# NOT recommended (old style)
#SBATCH --mem=16G           # Ambiguous on NOTS
```

### Shared Scratch Space

```bash
# Shared scratch for all nodes
$SHARED_SCRATCH/$USER   # Large, shared across cluster
ls $SHARED_SCRATCH/$USER

# Local scratch per node (faster)
$LOCAL_SCRATCH          # Node-local, faster I/O
```

## Monitoring and Control

### Check Queue Status

```bash
# Your jobs
squeue -u $USER

# Specific job details
squeue -j <JOB_ID>

# Estimate start time
squeue --start -j <JOB_ID>
```

### Check Partition Status

```bash
# All partitions
sinfo

# Specific partition
sinfo -p commons

# Node details
sinfo -N
```

### View Your Accounts and QOS

```bash
# Check available partitions and accounts for your user
sacctmgr show assoc user=$USER cluster=nots
```

### Monitor Running Job

```bash
# Real-time monitoring
nvidia-smi -l 1

# In separate terminal, SSH to compute node
ssh node-name
nvidia-smi
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

## Troubleshooting NOTS

### Error: "Invalid account or partition"

**Solution**: Must specify `--account=commons`

```bash
# Check your accounts
sacctmgr show assoc user=$USER

# Use correct account in job scripts
#SBATCH --account=commons
```

### Error: "QOSGrpCpuLimit" or "QOSGrpMemoryLimit"

**Meaning**: Commons partition resources at capacity

**Solutions**:
1. Submit fewer jobs
2. Use scavenge partition (may start faster)
3. Request less resources
4. Submit to `long` partition (may have less contention)

```bash
# Try scavenge partition (1 hour max)
#SBATCH --partition=scavenge
#SBATCH --time=00:60:00
```

### Error: "GPU not found" or "CUDA error"

**Solutions**:
```bash
# Ensure CUDA is loaded
module load cuda
nvidia-smi

# Specify GPU type if needed
#SBATCH --gres=gpu:volta:1   # Request specific GPU
```

### Error: "Time limit exceeded"

**Solutions**:
- Increase time in job script
- Use `long` partition for jobs > 24 hours
- Optimize code to run faster

```bash
# For long jobs
#SBATCH --partition=long
#SBATCH --time=72:00:00   # Max 72 hours
```

## Advanced NOTS Configurations

### CPU Architecture Constraints

```bash
# Specify CPU architecture (if available)
#SBATCH --constraint=icelake        # Intel Ice Lake
#SBATCH --constraint=cascadelake    # Intel Cascade Lake
#SBATCH --constraint=sapphirerapids # Intel Sapphire Rapids
#SBATCH --constraint=milan          # AMD Milan
```

### Network Fabric Selection

```bash
# Specify network
#SBATCH --constraint=opath      # Omni-Path
#SBATCH --constraint=ib         # InfiniBand
#SBATCH --constraint="icelake&opath"  # Combined
```

### Email Notifications (Recommended)

Add to all job scripts:
```bash
#SBATCH --mail-user=netID@rice.edu
#SBATCH --mail-type=ALL    # BEGIN, END, FAIL, REQUEUE
```

## Interactive Session Examples

### Quick Debug Session (Default)
```bash
./scripts/launch_interactive.sh

# Equivalent to:
# salloc --account=commons --partition=debug --ntasks=1 \
#        --cpus-per-task=2 --mem-per-cpu=2G --gres=gpu:1 \
#        --time=00:30:00
```

### Extended Development Session
```bash
./scripts/launch_interactive.sh --partition=commons --time=04:00:00 --cpus=8

# Equivalent to:
# salloc --account=commons --partition=commons --ntasks=1 \
#        --cpus-per-task=8 --mem-per-cpu=2G --gres=gpu:1 \
#        --time=04:00:00
```

### Specific GPU Type
```bash
./scripts/launch_interactive.sh --gpu-type=ampere --partition=commons --time=02:00:00

# Equivalent to:
# salloc --account=commons --partition=commons --ntasks=1 \
#        --cpus-per-task=2 --mem-per-cpu=2G --gres=gpu:ampere:1 \
#        --time=02:00:00
```

### Multi-GPU Session
```bash
./scripts/launch_interactive.sh --gpus=2 --partition=commons --time=02:00:00

# Equivalent to:
# salloc --account=commons --partition=commons --ntasks=1 \
#        --cpus-per-task=2 --mem-per-cpu=2G --gres=gpu:2 \
#        --time=02:00:00
```

## Performance Tips for NOTS

### Choose Right Partition

- **debug**: Quick iteration during development
- **commons**: General compute (default choice)
- **long**: Multi-day simulations
- **scavenge**: Backfill jobs, potentially shorter wait

### GPU Selection

```bash
# If performance matters, specify GPU:
#SBATCH --gres=gpu:h100:1  # Newest, fastest (may have longer queue)
#SBATCH --gres=gpu:volta:1 # V100, reliable (may have shorter queue)

# If cost/efficiency matters:
#SBATCH --gres=gpu:lovelace:1  # Good performance-to-power ratio
```

### Memory Optimization

```bash
# Start conservative, increase if needed
#SBATCH --mem-per-cpu=1G   # Minimum for most applications

# If job needs more:
#SBATCH --mem-per-cpu=4G
#SBATCH --mem-per-cpu=8G   # For large matrices/data
```

## Data Management on NOTS

### Workspace Locations

```bash
# Shared workspace (visible from all nodes)
$SHARED_SCRATCH/$USER          # Large, slower
/home/netID                    # Home directory, slower

# Node-local workspace (fast, node-specific)
$LOCAL_SCRATCH                 # Fast, node-local, auto-cleanup

# In job scripts:
cd $SHARED_SCRATCH/$USER
cp input.dat $LOCAL_SCRATCH/
# ... do computations in $LOCAL_SCRATCH ...
cp output.dat $SHARED_SCRATCH/$USER/
```

## Support and Resources

### Get Help

```bash
# Contact CRC help desk
# https://researchcomputing.rice.edu/request-help

# Check cluster status
sinfo

# View KB articles
# https://kb.rice.edu/147970  (NOTS Getting Started)
# https://kb.rice.edu/147998  (Partitions and QOS)
# https://kb.rice.edu/148046  (Job Submission)
```

### Example Scripts on NOTS

```bash
# Browse examples on cluster
cd /opt/apps/examples
ls -la
```

---

**NOTS Cluster Documentation**: https://kb.rice.edu/147970  
**Job Submission Guide**: https://kb.rice.edu/148046  
**Partitions & QOS**: https://kb.rice.edu/147998
