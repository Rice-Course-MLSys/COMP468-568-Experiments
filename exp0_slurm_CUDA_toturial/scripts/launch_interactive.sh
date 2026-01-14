#!/bin/bash
################################################################################
# Interactive Slurm Session Launcher for CUDA Debugging
#
# Description:
#   Launches an interactive Slurm session on a GPU node for debugging and
#   testing CUDA programs. Provides options for customizing resources.
#
# Usage:
#   ./launch_interactive.sh                    # Use defaults
#   ./launch_interactive.sh --gpus=2 --time=2:00:00
#   ./launch_interactive.sh --help
#
# Examples:
#   ./launch_interactive.sh                    # 1 GPU, 1 hour
#   ./launch_interactive.sh --gpus=2           # 2 GPUs, 1 hour
#   ./launch_interactive.sh --time=4:00:00     # 1 GPU, 4 hours
#   ./launch_interactive.sh --cpus=8 --mem=32G # More resources
#
################################################################################

# Default parameters
GPUS=1
TIME="00:30:00"
CPUS=2
MEMORY="2G"
PARTITION="debug"
ACCOUNT="commons"
NODES=1
NTASKS=1
JOB_NAME="cuda_debug"
GPU_TYPE=""  # Can be: volta, ampere, lovelace, h100, h200

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    cat << EOF
${BLUE}Interactive Slurm Session Launcher for CUDA Debugging (Rice NOTS Cluster)${NC}

${GREEN}Usage:${NC}
    $0 [OPTIONS]

${GREEN}Options:${NC}
    --gpus=N              Number of GPUs (default: $GPUS)
    --gpu-type=TYPE       GPU type: volta, ampere, lovelace, h100, h200 (default: any)
    --time=HH:MM:SS       Session duration (default: $TIME, max 30min debug, 24h commons)
    --cpus=N              CPU cores per task (default: $CPUS)
    --mem=SIZE            Memory per CPU (default: $MEMORY, e.g., 2G, 4G)
    --partition=NAME      Slurm partition: debug, commons, long, scavenge (default: $PARTITION)
    --job-name=NAME       Job name (default: $JOB_NAME)
    --help                Show this help message

${GREEN}Examples:${NC}
    # Quick debugging session (30 min, 1 GPU)
    $0

    # Longer development session on commons partition
    $0 --partition=commons --time=02:00:00

    # Specific GPU type (Tesla A40 - Ampere architecture)
    $0 --gpu-type=ampere --partition=commons --time=01:00:00

    # Multi-GPU session
    $0 --gpus=2 --partition=commons --time=01:00:00

    # Extended work session
    $0 --partition=commons --time=12:00:00 --cpus=8

${GREEN}Available GPU Types on NOTS:${NC}
    volta       - Tesla V100 GPUs
    ampere      - Tesla A40 GPUs
    lovelace    - Tesla L40S GPUs
    h100        - Hopper H100 GPUs
    h200        - Hopper H200 GPUs

${GREEN}Available Partitions on NOTS:${NC}
    debug       - Quick testing (max 30 minutes, best for debugging)
    commons     - General jobs (max 24 hours)
    long        - Long-running jobs (max 72 hours)
    scavenge    - Idle resources (max 1 hour, potentially faster start)

${GREEN}Common Time Values:${NC}
    00:15:00    15 minutes (debug)
    00:30:00    30 minutes (debug max)
    01:00:00    1 hour
    02:00:00    2 hours
    04:00:00    4 hours

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpus=*)
                GPUS="${1#*=}"
                shift
                ;;
            --gpu-type=*)
                GPU_TYPE="${1#*=}"
                shift
                ;;
            --time=*)
                TIME="${1#*=}"
                shift
                ;;
            --cpus=*)
                CPUS="${1#*=}"
                shift
                ;;
            --mem=*)
                MEMORY="${1#*=}"
                shift
                ;;
            --partition=*)
                PARTITION="${1#*=}"
                shift
                ;;
            --account=*)
                ACCOUNT="${1#*=}"
                shift
                ;;
            --job-name=*)
                JOB_NAME="${1#*=}"
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate time format
validate_time() {
    if ! [[ $TIME =~ ^[0-9]+:[0-9]{2}:[0-9]{2}$ ]]; then
        echo -e "${RED}Invalid time format: $TIME${NC}"
        echo -e "${YELLOW}Expected format: HH:MM:SS${NC}"
        exit 1
    fi
}

# Check if sbatch is available
check_slurm() {
    if ! command -v salloc &> /dev/null; then
        echo -e "${RED}Error: salloc command not found${NC}"
        echo -e "${YELLOW}Slurm may not be installed or loaded${NC}"
        echo -e "${YELLOW}Try: module load slurm${NC}"
        exit 1
    fi
}

# Print session configuration
print_config() {
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Interactive Slurm Session Configuration (Rice NOTS)${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "  ${GREEN}GPUs:${NC}            $GPUS"
    if [ -n "$GPU_TYPE" ]; then
        echo -e "  ${GREEN}GPU Type:${NC}       $GPU_TYPE"
    fi
    echo -e "  ${GREEN}CPUs:${NC}            $CPUS per task"
    echo -e "  ${GREEN}Memory:${NC}          $MEMORY per CPU"
    echo -e "  ${GREEN}Time Limit:${NC}      $TIME"
    echo -e "  ${GREEN}Partition:${NC}       $PARTITION"
    echo -e "  ${GREEN}Account:${NC}         $ACCOUNT"
    echo -e "  ${GREEN}Job Name:${NC}        $JOB_NAME"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Print helpful tips
print_tips() {
    cat << EOF
${GREEN}Tips for Your Interactive Session:${NC}

${YELLOW}1. Once you connect, load CUDA:${NC}
   module load cuda

${YELLOW}2. Navigate to your project:${NC}
   cd /path/to/Slurm_toturial

${YELLOW}3. Build your programs:${NC}
   make build

${YELLOW}4. Test immediately:${NC}
   ./build/gpu_info
   ./build/vector_add 50000000
   ./build/matrix_multiply 512

${YELLOW}5. Profile with NVIDIA tools:${NC}
   nvidia-smi                    # Check GPU status
   nvidia-smi -l 1              # Continuous monitoring (1 sec refresh)
   nsys profile ./build/vector_add 100000000

${YELLOW}6. Edit and recompile:${NC}
   vim src/vector_add.cu
   make rebuild
   ./build/vector_add 50000000

${YELLOW}7. Check available resources:${NC}
   sinfo                        # Partition info
   sacctmgr show assoc user=\$USER  # Your accounts and QOS

${YELLOW}8. Exit when done:${NC}
   exit
   logout
   Ctrl+D

${YELLOW}NOTS Cluster Notes:${NC}
- Partitions: debug (30 min max), commons (24h), long (72h), scavenge (1h)
- Use --account=commons for all jobs
- GPU types: volta (V100), ampere (A40), lovelace (L40S), h100, h200
- Shared scratch: \$SHARED_SCRATCH/\$USER
- Local scratch per node: \$LOCAL_SCRATCH

${YELLOW}Note:${NC} You are charged for the entire session time,
even if you're not actively running code. Use scavenge partition
for potential faster start with 1-hour time limit.

EOF
}

# Main execution
main() {
    echo -e "${BLUE}Starting Interactive Slurm Session Launcher...${NC}\n"
    
    # Parse arguments
    parse_arguments "$@"
    
    # Validate inputs
    validate_time
    check_slurm
    
    # Print configuration
    print_config
    
    # Print tips
    print_tips
    
    echo -e "${YELLOW}Launching interactive session...${NC}"
    echo -e "${YELLOW}(This may take a moment while waiting for resource allocation)${NC}\n"
    
    # Launch interactive session with GPU specification
    if [ -n "$GPU_TYPE" ]; then
        GPU_REQUEST="gpu:$GPU_TYPE:$GPUS"
    else
        GPU_REQUEST="gpu:$GPUS"
    fi
    
    salloc \
        --account=$ACCOUNT \
        --partition=$PARTITION \
        --nodes=$NODES \
        --ntasks=$NTASKS \
        --cpus-per-task=$CPUS \
        --mem-per-cpu=$MEMORY \
        --gres=$GPU_REQUEST \
        --time=$TIME \
        --job-name=$JOB_NAME
    
    # Capture exit status
    SESSION_EXIT=$?
    
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    
    if [ $SESSION_EXIT -eq 0 ]; then
        echo -e "${GREEN}Interactive session ended successfully${NC}"
        echo -e "${GREEN}Resources have been released${NC}"
    else
        echo -e "${YELLOW}Interactive session ended with exit code: $SESSION_EXIT${NC}"
    fi
    
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
}

# Run main function
main "$@"
