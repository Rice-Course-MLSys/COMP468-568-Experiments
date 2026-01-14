# Setup and Installation Guide

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [CUDA Installation](#cuda-installation)
3. [Slurm Configuration](#slurm-configuration)
4. [Project Configuration](#project-configuration)
5. [Initial Build](#initial-build)
6. [Verification](#verification)

## Environment Setup

### Step 1: SSH into Your HPC Cluster

```bash
ssh username@cluster.domain.edu
cd /path/to/project/root
```

### Step 2: Load Required Modules

```bash
# Load CUDA
module load cuda/12.0

# Load compiler (if not already loaded)
module load gcc/11.2

# Verify modules
module list
```

**List available modules:**
```bash
module avail
module avail cuda
```

### Step 3: Verify Environment

```bash
# Check CUDA installation
which nvcc
nvcc --version

# Check GPU availability
nvidia-smi

# Check compiler
gcc --version
g++ --version
```

**Expected output:**
```
nvcc: NVIDIA (R) Cuda compiler driver
CUDA release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267751_0
```

## CUDA Installation

### For Cluster Administrators

**Install CUDA Toolkit:**
```bash
# Download CUDA 12.0 from NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.105.02_linux.run

# Run installer
sudo sh cuda_12.0.0_525.105.02_linux.run

# Add to module system
module add-path /usr/local/cuda-12.0/modules
```

### For Users (without Admin Access)

**Option 1: Use Pre-installed CUDA**
```bash
module load cuda
```

**Option 2: Conda Environment**
```bash
# Create conda environment with CUDA
conda create -n cuda_dev cudatoolkit=12.0 -c conda-forge
conda activate cuda_dev
```

**Option 3: Local Installation (Home Directory)**
```bash
# Download CUDA to home
cd ~
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.105.02_linux.run

# Install to home directory
sh cuda_12.0.0_525.105.02_linux.run --silent --toolkit --toolkitpath=~/cuda-12.0

# Add to .bashrc
echo 'export PATH=~/cuda-12.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=~/cuda-12.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Slurm Configuration

### Understanding Your Cluster

**Get cluster information:**
```bash
sinfo                    # Queue information
sinfo -N                 # Node details
sinfo -p gpu             # GPU partition info
```

**Sample output interpretation:**
```
PARTITION AVAIL TIMELIMIT NODES STATE NODELIST
gpu*         up   4:00:00     4  idle node[01-04]
gpu*         up   4:00:00     2  alloc node[05-06]
```

### Configure Job Scripts

**Update paths in Slurm scripts:**

1. Open `scripts/job_vector_add.sh`
2. Change this line:
   ```bash
   cd /path/to/project  # Update this
   ```
   To your actual project path:
   ```bash
   cd /home/username/Slurm_toturial
   ```

3. Update partition name if needed:
   ```bash
   #SBATCH --partition=gpu      # Change "gpu" to your partition name
   ```

4. Repeat for all job scripts:
   - `job_gpu_info.sh`
   - `job_vector_add.sh`
   - `job_matrix_multiply.sh`
   - `job_all_tests.sh`
   - `job_multi_gpu.sh`

**Find available partitions:**
```bash
sinfo --Format=partition,available,timelimit,nodes,state
```

## Project Configuration

### Step 1: Adjust GPU Architecture

Edit `Makefile` and update `CUDA_ARCH`:

```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Output example: 7.0 (corresponds to sm_70)
```

**Update Makefile:**
```makefile
# Original
CUDA_ARCH := -arch=sm_70

# For different GPUs
CUDA_ARCH := -arch=sm_80      # A100
CUDA_ARCH := -arch=sm_86      # RTX 3090
CUDA_ARCH := -arch=sm_89      # RTX 4090
```

### Step 2: Create Necessary Directories

```bash
# Build directory
mkdir -p build

# Logs directory
mkdir -p logs

# Verify structure
ls -la
# Should show: src/ scripts/ build/ logs/ Makefile README.md
```

### Step 3: Set Script Permissions

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Verify
ls -l scripts/
# Should show rwxr-xr-x permissions
```

## Initial Build

### Step 1: Clean Build

```bash
# Remove any previous builds
make clean

# Build all programs
make build

# Output should show:
# ✓ Built build/vector_add
# ✓ Built build/matrix_multiply
# ✓ Built build/gpu_info
# ✓ All programs compiled successfully!
```

### Step 2: Verify Build Artifacts

```bash
# Check executables
ls -lh build/
```

**Expected output:**
```
-rwxr-xr-x  username  group  1.2M  Nov 24 10:30 vector_add
-rwxr-xr-x  username  group  1.3M  Nov 24 10:30 matrix_multiply
-rwxr-xr-x  username  group  0.8M  Nov 24 10:30 gpu_info
```

### Step 3: Local Testing (Optional)

**Test on login node (if GPU available):**
```bash
# Run GPU info
./build/gpu_info

# Run vector addition with small dataset
./build/vector_add 1000000

# Run matrix multiplication with small matrix
./build/matrix_multiply 256
```

## Verification

### Verification Checklist

- [ ] CUDA modules loaded: `module list | grep cuda`
- [ ] Compiler available: `which nvcc`
- [ ] GPU visible: `nvidia-smi` shows devices
- [ ] Project cloned/downloaded to working directory
- [ ] Makefile configured with correct CUDA_ARCH
- [ ] Build directory exists: `ls -d build`
- [ ] Logs directory exists: `ls -d logs`
- [ ] Scripts executable: `ls -l scripts/*.sh`
- [ ] Executables compiled: `ls build/`
- [ ] Slurm scripts updated with correct paths

### Test Compilation

```bash
# Show compilation details
make clean
make build 2>&1 | tee build.log

# Check for errors
grep -i error build.log
```

### Verify Slurm Configuration

```bash
# Test job submission (dry run)
sbatch --test-only scripts/job_gpu_info.sh

# Output should be positive:
# sbatch: Job script successfully validated
```

### Create Test Job

**Submit simple test:**
```bash
# Submit GPU info job
sbatch scripts/job_gpu_info.sh

# Check job status
squeue

# Wait a few seconds, then check output
tail logs/gpu_info_*.out
```

## Common Setup Issues and Solutions

### Issue: Module not found

```bash
# Wrong module name
module load cuda/12.0    # ✓ Correct
module load cuda-12.0    # ✗ Wrong

# Solution: check available modules
module avail | grep cuda
```

### Issue: CUDA not in PATH

```bash
# Verify PATH
echo $PATH | grep cuda

# Add to PATH manually (temporary)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make permanent (add to ~/.bashrc)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Issue: nvcc: command not found

```bash
# Load CUDA module
module load cuda

# Verify
which nvcc
```

### Issue: Compilation fails with unknown GPU

```bash
# Find your GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv

# Update Makefile with correct -arch value
vim Makefile
# Change: CUDA_ARCH := -arch=sm_XX
# Where XX is your GPU's compute capability

# Rebuild
make clean && make build
```

### Issue: Permission denied on scripts

```bash
# Make executable
chmod +x scripts/*.sh

# Verify permissions
ls -l scripts/
# Should show: rwxr-xr-x or rwx------
```

### Issue: Cannot submit jobs

```bash
# Verify Slurm is available
which sbatch
which squeue

# Check cluster connectivity
sinfo

# If error, may need to load Slurm module
module load slurm
```

### Issue: Job script path incorrect

```bash
# Find correct project path
pwd

# Update all job scripts with absolute path
sed -i 's|/path/to/project|'$(pwd)'|g' scripts/*.sh

# Verify changes
grep "cd " scripts/job_*.sh
```

## Post-Setup Verification

### Quick Test Workflow

```bash
# 1. Verify compilation
make build

# 2. Test on login node (if available)
./build/gpu_info

# 3. Submit test job to cluster
sbatch scripts/job_gpu_info.sh

# 4. Monitor job
watch -n 2 squeue

# 5. Check output when complete
tail logs/gpu_info_*.out
```

### Expected Successful Output

**gpu_info output should include:**
```
========================================
CUDA GPU Information
========================================
Number of GPUs: 1

GPU 0: Tesla V100-SXM2-32GB
  Compute Capability: 7.0
  Total Global Memory: 32.00 GB
  ...
```

**vector_add output should include:**
```
========================================
CUDA Vector Addition Experiment
========================================
Vector size: 1000000 elements
...
✓ Verification passed! All values match.
Kernel execution time: X.XXXX ms
```

## Next Steps

After successful setup:

1. **Run Comprehensive Tests**
   ```bash
   make submit_all
   ```

2. **Analyze Performance**
   - Compare different matrix sizes
   - Analyze kernel execution times
   - Check memory throughput

3. **Explore Advanced Features**
   - Try multi-GPU execution: `make submit_multi_gpu`
   - Profile with NVIDIA tools: `nsys profile`
   - Modify programs for your research

4. **Customize for Your Work**
   - Adapt code to your algorithms
   - Adjust block/grid configurations
   - Optimize memory access patterns

---

For additional help, consult:
- HPC Cluster Documentation
- NVIDIA CUDA Documentation
- Slurm Official Guide
