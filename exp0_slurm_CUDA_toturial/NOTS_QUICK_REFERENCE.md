# Quick Reference: NOTS Cluster Commands

## Accessing NOTS

```bash
ssh netID@nots.rice.edu
```

## Module Commands

```bash
module load cuda          # Load CUDA
module load gcc           # Load C++ compiler
module list               # Show loaded modules
module avail              # Show available modules
```

## Building Your Project

```bash
cd $SHARED_SCRATCH/$USER/Slurm_toturial
module load cuda
make build               # Compile all programs
make clean               # Remove old builds
```

## Interactive Debugging Sessions

```bash
# Quick 30-minute debug session (default)
./scripts/launch_interactive.sh

# Longer session on commons partition
./scripts/launch_interactive.sh --partition=commons --time=02:00:00

# Specific GPU type (Tesla A40)
./scripts/launch_interactive.sh --gpu-type=ampere --partition=commons --time=01:00:00

# Multi-GPU debugging
./scripts/launch_interactive.sh --gpus=2 --partition=commons --time=02:00:00

# See all options
./scripts/launch_interactive.sh --help
```

## Batch Job Submission

```bash
# Quick GPU info test (debug partition)
sbatch scripts/job_gpu_info.sh

# Vector addition test
sbatch scripts/job_vector_add.sh

# Matrix multiplication
sbatch scripts/job_matrix_multiply.sh

# Comprehensive test suite
sbatch scripts/job_all_tests.sh

# Multi-GPU test
sbatch scripts/job_multi_gpu.sh
```

## Job Monitoring

```bash
# Show your jobs
squeue -u $USER

# Detailed job info
squeue -j <JOB_ID>

# Estimate start time
squeue --start -j <JOB_ID>

# Partition status
sinfo

# Node information
sinfo -N

# Check your accounts/access
sacctmgr show assoc user=$USER

# View job history
sacct -u $USER
```

## Job Management

```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Get verbose job details
scontrol show job <JOB_ID>

# View job script
scontrol write batch_script <JOB_ID> -
```

## File Management

```bash
# Navigate to shared workspace
cd $SHARED_SCRATCH/$USER

# Check quota
quota

# Listing files
ls -la $SHARED_SCRATCH/$USER

# Local scratch (fast, node-local)
cd $LOCAL_SCRATCH
pwd
```

## Check Job Output

```bash
# While job is running
tail -f logs/vector_add_*.out

# After job completes
cat logs/vector_add_12345.out
cat logs/vector_add_12345.err

# View recent logs
ls -lht logs/ | head -10
```

## GPU Monitoring

```bash
# Check GPU status
nvidia-smi

# Continuous monitoring (refresh every 1 sec)
nvidia-smi -l 1

# Detailed GPU metrics
nvidia-smi dmon -s pucvmet

# In SSH'd compute node during job run
ssh <node-name>
nvidia-smi
```

## Common Job Parameters for NOTS

```bash
# Always required for NOTS:
#SBATCH --account=commons

# For testing/debugging:
#SBATCH --partition=debug
#SBATCH --time=00:30:00      # 30 min (debug max)

# For production:
#SBATCH --partition=commons
#SBATCH --time=24:00:00      # Up to 24 hours

# For long jobs:
#SBATCH --partition=long
#SBATCH --time=72:00:00      # Up to 72 hours

# GPU selection:
#SBATCH --gres=gpu:1                # Any GPU
#SBATCH --gres=gpu:volta:1          # Tesla V100
#SBATCH --gres=gpu:ampere:1         # Tesla A40
#SBATCH --gres=gpu:lovelace:1       # Tesla L40S
#SBATCH --gres=gpu:h100:1           # Hopper H100
#SBATCH --gres=gpu:h200:1           # Hopper H200

# Memory (per-CPU):
#SBATCH --mem-per-cpu=1G
#SBATCH --mem-per-cpu=2G
#SBATCH --mem-per-cpu=4G

# CPU allocation:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# (Total memory = cpus-per-task × mem-per-cpu)
```

## Typical Workflow

### 1. Compile on Login Node
```bash
module load cuda
cd $SHARED_SCRATCH/$USER/Slurm_toturial
make build
```

### 2. Test Interactively
```bash
./scripts/launch_interactive.sh
# Now on compute node:
./build/gpu_info
./build/vector_add 10000000
exit
```

### 3. Submit Batch Job
```bash
sbatch scripts/job_vector_add.sh
```

### 4. Monitor
```bash
squeue -u $USER
tail logs/vector_add_*.out
```

### 5. Review Results
```bash
cat logs/vector_add_12345.out
grep "Performance:" logs/vector_add_12345.out
```

## Troubleshooting

### "Invalid account or partition"
```bash
# Check your accounts
sacctmgr show assoc user=$USER

# Add to job script:
#SBATCH --account=commons
```

### "QOSGrpCpuLimit" or "QOSGrpMemoryLimit"
```bash
# Commons partition at capacity, try:
#SBATCH --partition=scavenge  # May start faster (1 hr max)
# Or submit fewer jobs, or request fewer resources
```

### "GPU not found"
```bash
# Ensure CUDA is loaded
module load cuda
nvidia-smi

# For interactive session:
./scripts/launch_interactive.sh --gpu-type=volta
```

### "Time limit exceeded"
```bash
# Increase time in job script:
#SBATCH --time=01:00:00  # Increase from current

# Or use long partition:
#SBATCH --partition=long
#SBATCH --time=72:00:00
```

## Useful Links

- **NOTS Getting Started**: https://kb.rice.edu/147970
- **Partitions & QOS**: https://kb.rice.edu/147998
- **Job Submission**: https://kb.rice.edu/148046
- **Available Examples**: `/opt/apps/examples/` (on NOTS)
- **Request Help**: https://researchcomputing.rice.edu/request-help

## Environment Variables in Jobs

```bash
$SLURM_JOB_ID           # Job ID
$SLURM_JOB_NAME         # Job name
$SLURM_NODELIST         # Node(s) assigned
$SLURM_SUBMIT_DIR       # Submission directory
$SHARED_SCRATCH         # Shared scratch path
$LOCAL_SCRATCH          # Local scratch path
$USER                   # Your username
```

---

**Pro Tips**:
- Use `debug` partition for quick iterations
- Use `scavenge` for potentially faster start
- Save results to `$SHARED_SCRATCH` (not `$LOCAL_SCRATCH`)
- Always specify accurate runtime to improve scheduling
- Use `--mail-type=ALL` for email notifications

