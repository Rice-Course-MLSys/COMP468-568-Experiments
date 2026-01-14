# NOTS Cluster Adaptation Summary

This document outlines all changes made to adapt the CUDA+Slurm project for Rice NOTS cluster.

## Changes Made

### 1. Job Scripts (`scripts/`)

All job scripts have been updated with NOTS-specific directives:

#### Key Changes in All Scripts:

**Before (Generic Slurm):**
```bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
```

**After (NOTS):**
```bash
#SBATCH --account=commons          # REQUIRED on NOTS
#SBATCH --partition=commons        # Use: debug, commons, long, scavenge
#SBATCH --gres=gpu:volta:1         # Specify GPU type (volta, ampere, etc.)
#SBATCH --mem-per-cpu=2G           # Per-CPU memory, not total
```

#### Specific Script Updates:

##### `job_gpu_info.sh`
- Partition: `debug` (quick testing)
- GPU: `1` (any type)
- Time: 15 min
- Memory: 1G per CPU

##### `job_vector_add.sh`
- Partition: `commons`
- GPU: `1 volta` (Tesla V100)
- Time: 30 min
- Memory: 2G per CPU

##### `job_matrix_multiply.sh`
- Partition: `commons`
- GPU: `1 volta`
- Time: 45 min
- Memory: 2G per CPU

##### `job_all_tests.sh`
- Partition: `commons`
- GPU: `1 volta`
- Time: 1 hour
- Memory: 2G per CPU

##### `job_multi_gpu.sh`
- Partition: `commons`
- GPU: `2` (any types)
- Time: 45 min
- Memory: 4G per CPU

### 2. Interactive Launcher (`scripts/launch_interactive.sh`)

Enhanced to support NOTS-specific features:

**New Options:**
- `--gpu-type=TYPE` - Specify GPU: volta, ampere, lovelace, h100, h200
- `--account=ACCOUNT` - Account specification (default: commons)
- Partition support: debug, commons, long, scavenge

**Default Configuration:**
- Partition: `debug` (30 min max for fast iteration)
- GPU: 1 (any type)
- Memory: 2G per CPU
- Time: 30 minutes

**Examples:**
```bash
./launch_interactive.sh --gpu-type=ampere --time=02:00:00
./launch_interactive.sh --partition=commons --gpus=2
./launch_interactive.sh --gpu-type=h100 --partition=long --time=10:00:00
```

### 3. New Documentation: `docs/NOTS_SETUP.md`

Comprehensive guide for NOTS cluster including:

- **Cluster Access**: SSH and account checking
- **Partitions**: Detailed comparison of debug, commons, long, scavenge
- **GPU Types**: Volta (V100), Ampere (A40), Lovelace (L40S), H100, H200
- **Project Setup**: Step-by-step for NOTS
- **GPU Architecture Selection**: CUDA_ARCH mapping for each GPU
- **Job Submission Examples**: Realistic scenarios for NOTS
- **Memory Specification**: Correct `--mem-per-cpu` format
- **Shared Scratch**: Using `$SHARED_SCRATCH/$USER`
- **Troubleshooting**: Common NOTS-specific issues
- **Advanced Configurations**: CPU constraints, network fabrics, email notifications
- **Performance Tips**: Partition and GPU selection strategies

### 4. Updated README.md

- Added reference to NOTS_SETUP.md
- Updated prerequisites section with NOTS-specific note
- Added NOTS FAQs
- Updated version to 2.0

### 5. Module Loading Changes

**Before:**
```bash
module load cuda/12.0
module load gcc/11.2
```

**After:**
```bash
module load cuda        # NOTS handles versioning automatically
module load gcc         # Version selection automatic
```

## NOTS-Specific Directives Reference

### Account (Required)
```bash
#SBATCH --account=commons
```

### Partitions
```bash
#SBATCH --partition=debug      # Max 30 min, for testing
#SBATCH --partition=commons    # Max 24 hours, general use
#SBATCH --partition=long       # Max 72 hours, long jobs
#SBATCH --partition=scavenge   # Max 1 hour, backfill jobs
```

### GPU Requests
```bash
#SBATCH --gres=gpu:1           # Any GPU type
#SBATCH --gres=gpu:volta:1     # Tesla V100
#SBATCH --gres=gpu:ampere:1    # Tesla A40
#SBATCH --gres=gpu:lovelace:1  # Tesla L40S
#SBATCH --gres=gpu:h100:1      # Hopper H100
#SBATCH --gres=gpu:h200:1      # Hopper H200
#SBATCH --gres=gpu:2           # 2 GPUs of any type
```

### Memory (NOTS Format)
```bash
#SBATCH --mem-per-cpu=1G       # Per-CPU memory
#SBATCH --mem-per-cpu=2G
#SBATCH --mem-per-cpu=4G
```

### CPUs and Tasks
```bash
#SBATCH --ntasks=1             # Number of tasks
#SBATCH --cpus-per-task=2      # CPU cores per task
# Total memory = cpus-per-task × mem-per-cpu
```

## Migration Guide: From Generic Slurm to NOTS

### Step 1: Update Job Scripts
Replace generic partition names with NOTS partitions:
- ❌ `--partition=gpu` → ✅ `--partition=commons`
- Add ✅ `--account=commons` to all jobs

### Step 2: Update GPU Requests
Replace generic GPU syntax:
- ❌ `--gpus=1` → ✅ `--gres=gpu:volta:1`
- ❌ `--gpus=2` → ✅ `--gres=gpu:2`

### Step 3: Update Memory Specification
Replace total memory with per-CPU:
- ❌ `--mem=16G` → ✅ `--mem-per-cpu=2G` + `--cpus-per-task=4` (total: 8GB)
- ❌ `--mem=8G` → ✅ `--mem-per-cpu=2G` + `--cpus-per-task=2` (total: 4GB)

### Step 4: Update Module Loading
Simplify module commands:
- ❌ `module load cuda/12.0` → ✅ `module load cuda`

### Step 5: Use Interactive Launcher
Use updated launcher with NOTS options:
```bash
./scripts/launch_interactive.sh --gpu-type=volta --time=01:00:00
```

## Testing the Adapted Scripts

### Test GPU Info Job (Debug Partition)
```bash
sbatch scripts/job_gpu_info.sh
tail logs/gpu_info_*.out
```

### Test Vector Addition (Commons Partition)
```bash
sbatch scripts/job_vector_add.sh
tail logs/vector_add_*.out
```

### Test Interactive Session
```bash
./scripts/launch_interactive.sh
# Now in interactive session:
module load cuda
./build/gpu_info
exit
```

## Key NOTS Cluster Specs

- **Login Node**: nots.rice.edu
- **GPUs Available**: V100, A40, L40S, H100, H200
- **Shared Scratch**: `$SHARED_SCRATCH/$USER` (visible from all nodes)
- **Local Scratch**: `$LOCAL_SCRATCH` (node-local, faster)
- **Home Directory**: `/home/$USER` (small quota)
- **Default Account**: commons
- **Default QOS**: nots_commons (for commons partition)

## Common NOTS Commands

```bash
# Check your access
sacctmgr show assoc user=$USER

# View partition status
sinfo
sinfo -p commons

# List your jobs
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>

# Estimate start time
squeue --start -j <JOB_ID>

# Cancel job
scancel <JOB_ID>
```

## Backward Compatibility

**Generic Slurm clusters** (non-NOTS):
- Original scripts still work with minor modifications
- Update `--partition=gpu` to your cluster's partition
- Remove `--account=commons` if not required
- Change `--gres=gpu:volta:1` back to `--gpus=1`
- Update `--mem-per-cpu` to `--mem` if preferred

See `docs/SETUP_GUIDE.md` for generic Slurm clusters.

## Documentation Updates

| Document | Changes |
|----------|---------|
| `SETUP_GUIDE.md` | Generic Slurm setup (unchanged) |
| `EXECUTION_GUIDE.md` | Generic Slurm execution (unchanged) |
| `GPU_PROGRAMMING_GUIDE.md` | CUDA concepts (unchanged) |
| `NOTS_SETUP.md` | **NEW** - NOTS-specific guide |
| `README.md` | Updated with NOTS reference |
| All job scripts | Updated for NOTS |
| `launch_interactive.sh` | Enhanced for NOTS |

---

**Adapted**: November 24, 2025  
**For**: Rice University NOTS Cluster  
**Reference**: https://kb.rice.edu/147970
