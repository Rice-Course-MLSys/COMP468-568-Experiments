#!/bin/bash
#SBATCH --job-name=vector_add
#SBATCH --output=logs/vector_add_%j.out
#SBATCH --error=logs/vector_add_%j.err
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:volta:1
#SBATCH --time=00:30:00

# Load CUDA module
module load cuda  # NOTS automatically loads appropriate CUDA version

# Job information
echo "=========================================="
echo "Vector Addition Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="
echo ""

cd /path/to/project  # Update with actual project path

# Run with different vector sizes
echo "Testing with default size (1M elements)..."
./build/vector_add

echo ""
echo "Testing with 10M elements..."
./build/vector_add 10000000

echo ""
echo "Testing with 100M elements..."
./build/vector_add 100000000

echo ""
echo "Job completed at $(date)"
