#!/bin/bash
#SBATCH --job-name=cuda_multi_gpu
#SBATCH --output=logs/multi_gpu_%j.out
#SBATCH --error=logs/multi_gpu_%j.err
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:2
#SBATCH --time=00:45:00

# Load CUDA module
module load cuda  # NOTS automatically loads appropriate CUDA version

# Job information
echo "=========================================="
echo "Multi-GPU CUDA Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="
echo ""

cd /path/to/project  # Update with actual project path

# Run tests on multiple GPUs
echo "GPU Information (All GPUs):"
./build/gpu_info

echo ""
echo "Running Vector Addition on GPU 0..."
CUDA_VISIBLE_DEVICES=0 ./build/vector_add 100000000

echo ""
echo "Running Vector Addition on GPU 1..."
CUDA_VISIBLE_DEVICES=1 ./build/vector_add 100000000

echo ""
echo "Running Matrix Multiply on GPU 0..."
CUDA_VISIBLE_DEVICES=0 ./build/matrix_multiply 1024

echo ""
echo "Running Matrix Multiply on GPU 1..."
CUDA_VISIBLE_DEVICES=1 ./build/matrix_multiply 1024

echo ""
echo "Job completed at $(date)"
