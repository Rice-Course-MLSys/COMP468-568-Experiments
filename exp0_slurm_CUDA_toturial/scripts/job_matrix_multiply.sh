#!/bin/bash
#SBATCH --job-name=matrix_multiply
#SBATCH --output=logs/matrix_multiply_%j.out
#SBATCH --error=logs/matrix_multiply_%j.err
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:volta:1
#SBATCH --time=00:45:00

# Load CUDA module
module load cuda  # NOTS automatically loads appropriate CUDA version

# Job information
echo "=========================================="
echo "Matrix Multiplication Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="
echo ""

cd /path/to/project  # Update with actual project path

# Run with different matrix sizes
echo "Testing with 512x512 matrix..."
./build/matrix_multiply 512

echo ""
echo "Testing with 1024x1024 matrix..."
./build/matrix_multiply 1024

echo ""
echo "Testing with 2048x2048 matrix..."
./build/matrix_multiply 2048

echo ""
echo "Job completed at $(date)"
