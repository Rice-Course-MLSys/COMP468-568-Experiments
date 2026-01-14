#!/bin/bash
#SBATCH --job-name=cuda_all_tests
#SBATCH --output=logs/all_tests_%j.out
#SBATCH --error=logs/all_tests_%j.err
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:volta:1
#SBATCH --time=01:00:00

# Load CUDA module
module load cuda  # NOTS automatically loads appropriate CUDA version

PROJECT_PATH="/path/to/project"  # Update with actual project path
cd "$PROJECT_PATH"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_PATH/logs"

# Job information
echo "=========================================="
echo "Comprehensive CUDA Tests"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Test 1: GPU Information
echo "[TEST 1/3] GPU Information"
echo "-------------------------------------------"
./build/gpu_info
TEST1_STATUS=$?
echo ""

# Test 2: Vector Addition
echo "[TEST 2/3] Vector Addition"
echo "-------------------------------------------"
./build/vector_add 50000000
TEST2_STATUS=$?
echo ""

# Test 3: Matrix Multiplication
echo "[TEST 3/3] Matrix Multiplication"
echo "-------------------------------------------"
./build/matrix_multiply 1024
TEST3_STATUS=$?
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "GPU Info:           $([ $TEST1_STATUS -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "Vector Addition:    $([ $TEST2_STATUS -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "Matrix Multiply:    $([ $TEST3_STATUS -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "End Time: $(date)"
echo "=========================================="

# Exit with error if any test failed
if [ $TEST1_STATUS -ne 0 ] || [ $TEST2_STATUS -ne 0 ] || [ $TEST3_STATUS -ne 0 ]; then
    exit 1
fi

exit 0
