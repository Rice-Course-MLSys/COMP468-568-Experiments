# Makefile for CUDA Programs with Slurm

# Compiler and flags
NVCC := nvcc
CFLAGS := -O3 -std=c++11
CUDA_ARCH := -arch=sm_70  # Adjust for your GPU (sm_70 for V100, sm_80 for A100, sm_86 for RTX 3090)
LDFLAGS := -lm

# Directories
SRC_DIR := src
BUILD_DIR := build
LOGS_DIR := logs

# Source files
VECTOR_ADD_SRC := $(SRC_DIR)/vector_add.cu
MATRIX_MULT_SRC := $(SRC_DIR)/matrix_multiply.cu
GPU_INFO_SRC := $(SRC_DIR)/gpu_info.cu

# Output executables
VECTOR_ADD_BIN := $(BUILD_DIR)/vector_add
MATRIX_MULT_BIN := $(BUILD_DIR)/matrix_multiply
GPU_INFO_BIN := $(BUILD_DIR)/gpu_info

# Phony targets
.PHONY: all clean build run_all help logs interactive

# Default target
all: build

# Help target
help:
	@echo "CUDA + Slurm Build System"
	@echo "========================="
	@echo "Targets:"
	@echo "  make compile        - Compile all CUDA programs"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make logs           - Create logs directory"
	@echo "  make run_gpu_info   - Run GPU info locally (if GPU available)"
	@echo "  make run_vector     - Run vector addition locally"
	@echo "  make run_matrix     - Run matrix multiplication locally"
	@echo "  make interactive    - Launch interactive Slurm session for debugging"
	@echo "  make submit_all     - Submit all jobs to Slurm"
	@echo "  make submit_gpu     - Submit GPU info job"
	@echo "  make submit_vector  - Submit vector addition job"
	@echo "  make submit_matrix  - Submit matrix multiplication job"
	@echo "  make status         - Check Slurm job status"
	@echo "  make help           - Display this help message"
	@echo ""
	@echo "Compiler: $(NVCC)"
	@echo "CUDA Architecture: $(CUDA_ARCH)"

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Create logs directory
logs:
	mkdir -p $(LOGS_DIR)

# Compile vector addition
$(VECTOR_ADD_BIN): $(VECTOR_ADD_SRC) | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $@ $< $(LDFLAGS)
	@echo "✓ Built $@"

# Compile matrix multiplication
$(MATRIX_MULT_BIN): $(MATRIX_MULT_SRC) | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $@ $< $(LDFLAGS)
	@echo "✓ Built $@"

# Compile GPU info
$(GPU_INFO_BIN): $(GPU_INFO_SRC) | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) $(CUDA_ARCH) -o $@ $< $(LDFLAGS)
	@echo "✓ Built $@"

# Build all targets
compile: logs $(VECTOR_ADD_BIN) $(MATRIX_MULT_BIN) $(GPU_INFO_BIN)
	@echo ""
	@echo "✓ All programs compiled successfully!"
	@echo "Executables:"
	@echo "  - $(VECTOR_ADD_BIN)"
	@echo "  - $(MATRIX_MULT_BIN)"
	@echo "  - $(GPU_INFO_BIN)"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	@echo "✓ Clean complete"

# Local run targets (for testing before Slurm submission)
run_gpu_info: $(GPU_INFO_BIN)
	@echo "Running GPU Info..."
	./$(GPU_INFO_BIN)

run_vector: $(VECTOR_ADD_BIN)
	@echo "Running Vector Addition (1M elements)..."
	./$(VECTOR_ADD_BIN) 1000000

run_matrix: $(MATRIX_MULT_BIN)
	@echo "Running Matrix Multiplication (512x512)..."
	./$(MATRIX_MULT_BIN) 512

# Slurm submission targets
submit_gpu: build scripts/job_gpu_info.sh logs
	@echo "Submitting GPU info job..."
	sbatch scripts/job_gpu_info.sh

submit_vector: build scripts/job_vector_add.sh logs
	@echo "Submitting vector addition job..."
	sbatch scripts/job_vector_add.sh

submit_matrix: build scripts/job_matrix_multiply.sh logs
	@echo "Submitting matrix multiplication job..."
	sbatch scripts/job_matrix_multiply.sh

submit_all: build scripts/job_all_tests.sh logs
	@echo "Submitting comprehensive test job..."
	sbatch scripts/job_all_tests.sh

submit_multi_gpu: build scripts/job_multi_gpu.sh logs
	@echo "Submitting multi-GPU test job..."
	sbatch scripts/job_multi_gpu.sh

# Interactive session for debugging
interactive: build
	@echo "Launching interactive Slurm session..."
	@chmod +x scripts/launch_interactive.sh
	@scripts/launch_interactive.sh

# Slurm utility targets
status:
	@echo "Current Slurm jobs:"
	squeue -u $$USER

cancel_all:
	@echo "Cancelling all jobs for user $$USER..."
	scancel -u $$USER

view_logs:
	@echo "Recent log files:"
	ls -lht $(LOGS_DIR) | head -10

# Display compiler info
info:
	@echo "CUDA Compiler Information:"
	@nvcc --version
	@echo ""
	@echo "Build Configuration:"
	@echo "  Compiler: $(NVCC)"
	@echo "  C Flags: $(CFLAGS)"
	@echo "  CUDA Arch: $(CUDA_ARCH)"
	@echo "  Linker Flags: $(LDFLAGS)"
