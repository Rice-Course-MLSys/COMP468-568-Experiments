#!/bin/bash
#SBATCH --job-name=gpu_info
#SBATCH --output=logs/gpu_info_%j.out
#SBATCH --error=logs/gpu_info_%j.err
#SBATCH --account=commons
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

# Load CUDA module (available on NOTS)
module load cuda  # NOTS automatically loads appropriate CUDA version

# Run GPU info program
echo "=========================================="
echo "GPU Information Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "Allocated GPUs: $SLURM_GPUS"
echo "=========================================="
echo ""

cd /path/to/project  # Update with actual project path
./build/gpu_info

echo ""
echo "Job completed at $(date)"
