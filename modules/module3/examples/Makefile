# Module 3: Advanced GPU Algorithms
# Makefile for comprehensive build and testing

# Compiler settings
NVCC = nvcc
HIPCC = hipcc
CXX = g++

# GPU vendor detection
NVIDIA_GPU := $(shell nvidia-smi > /dev/null 2>&1 && echo 1 || echo 0)
AMD_GPU := $(shell rocm-smi > /dev/null 2>&1 && echo 1 || echo 0)

# Determine build target based on GPU vendor
ifeq ($(NVIDIA_GPU),1)
BUILD_CUDA = 1
BUILD_HIP = 0
GPU_VENDOR = NVIDIA
else ifeq ($(AMD_GPU),1)
BUILD_CUDA = 0
BUILD_HIP = 1
GPU_VENDOR = AMD
else
BUILD_CUDA = 0
BUILD_HIP = 0
GPU_VENDOR = NONE
endif

# Compiler flags
CUDA_FLAGS = -std=c++17 -O2 -arch=sm_75 -lcudart -lcuda
CUDA_DEBUG_FLAGS = -std=c++17 -g -G -arch=sm_75 -lcudart -lcuda
HIP_FLAGS = -std=c++17 -O2
HIP_DEBUG_FLAGS = -std=c++17 -g

# ROCm 7: Ensure hipcc can find HIP runtime by passing --rocm-path
ROCM_PATH ?= $(shell ls -d /opt/rocm-7.0.0 2>/dev/null || ls -d /opt/rocm* 2>/dev/null | head -1 || echo /opt/rocm)
# Auto-detect ROCm path from hipcc if headers not found
ifeq ($(wildcard $(ROCM_PATH)/include/hip/hip_runtime.h),)
	HIPCC_BIN := $(shell command -v hipcc 2>/dev/null)
	ifneq ($(HIPCC_BIN),)
		ROCM_PATH_DETECTED := $(shell dirname $$(dirname $$(realpath $(HIPCC_BIN))))
		ROCM_PATH := $(ROCM_PATH_DETECTED)
	endif
endif
HIP_ROCM_FLAG = --rocm-path=$(ROCM_PATH)
HIP_FLAGS += $(HIP_ROCM_FLAG)
HIP_DEBUG_FLAGS += $(HIP_ROCM_FLAG)

# GPU architecture detection - get actual GPU architecture from rocminfo
GPU_ARCH := $(shell if command -v rocminfo >/dev/null 2>&1; then rocminfo 2>/dev/null | grep -o 'gfx[0-9]*' | head -1; else echo gfx1030; fi)
ifeq ($(strip $(GPU_ARCH)),)
	GPU_ARCH := gfx1030
endif

# Add detected GPU architecture to HIP flags
HIP_FLAGS += --offload-arch=$(GPU_ARCH)
HIP_DEBUG_FLAGS += --offload-arch=$(GPU_ARCH)

# ROCm library linking for advanced examples
HIP_LIB_DIR := $(ROCM_PATH)/lib
HIP_LDFLAGS := -L$(HIP_LIB_DIR) -lrocblas -Wl,-rpath,$(HIP_LIB_DIR)

CXX_FLAGS = -std=c++17 -O2

# Directories
EXAMPLES_DIR = .
BUILD_DIR = build
PROFILE_DIR = profiles

# CUDA Examples
CUDA_SOURCES = $(wildcard $(EXAMPLES_DIR)/*_cuda.cu $(EXAMPLES_DIR)/0*.cu)
CUDA_TARGETS = $(patsubst $(EXAMPLES_DIR)/%.cu,$(BUILD_DIR)/%,$(CUDA_SOURCES))

# HIP Examples
HIP_SOURCES = $(wildcard $(EXAMPLES_DIR)/*_hip.cpp)
HIP_TARGETS = $(patsubst $(EXAMPLES_DIR)/%.cpp,$(BUILD_DIR)/%,$(HIP_SOURCES))

# Check for hipcc availability
HIPCC_AVAILABLE := $(shell command -v hipcc >/dev/null 2>&1 && echo 1 || echo 0)

# Active targets based on detected GPU vendor and compiler availability
ifeq ($(BUILD_CUDA),1)
  ALL_TARGETS = $(CUDA_TARGETS)
else ifeq ($(BUILD_HIP),1)
  ALL_TARGETS = $(HIP_TARGETS)
else
  ALL_TARGETS = 
endif

# Default target
.PHONY: all
all: setup $(ALL_TARGETS)

# Setup directories
.PHONY: setup
setup:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(PROFILE_DIR)
ifeq ($(GPU_VENDOR),NVIDIA)
	@echo "✓ NVIDIA GPU detected - building CUDA examples"
else ifeq ($(GPU_VENDOR),AMD)
	@echo "✓ AMD GPU detected - building HIP examples"
	@echo "ℹ Using ROCm path: $(ROCM_PATH)"
else
	@echo "⚠ No compatible GPU detected"
endif

# CUDA compilation rules
.PHONY: cuda
ifeq ($(BUILD_CUDA),1)
cuda: setup $(CUDA_TARGETS)
else
cuda: setup
	@echo "⚠ CUDA build requested but no NVIDIA GPU detected"
endif

$(BUILD_DIR)/%_cuda: $(EXAMPLES_DIR)/%_cuda.cu
	@echo "Building CUDA example: $@"
	$(NVCC) $(CUDA_FLAGS) $< -o $@

# Pattern for numbered CUDA examples
$(BUILD_DIR)/0%: $(EXAMPLES_DIR)/0%.cu
	@echo "Building CUDA example: $@"
	$(NVCC) $(CUDA_FLAGS) $< -o $@

# HIP compilation rules
.PHONY: hip
ifeq ($(BUILD_HIP),1)
hip: setup $(HIP_TARGETS)
else
hip: setup
	@echo "⚠ HIP build requested but no AMD GPU detected"
endif

ifeq ($(BUILD_HIP),1)
$(BUILD_DIR)/%_hip: $(EXAMPLES_DIR)/%_hip.cpp
	@echo "Building HIP example: $@"
	$(HIPCC) $(HIP_FLAGS) $< -o $@ $(HIP_LDFLAGS)
endif

# Debug builds
.PHONY: debug
debug: CUDA_FLAGS = $(CUDA_DEBUG_FLAGS)
debug: HIP_FLAGS = $(HIP_DEBUG_FLAGS)
debug: all

# Profile builds
.PHONY: profile
profile: CUDA_FLAGS += -lineinfo
profile: HIP_FLAGS += -g
profile: all
	@echo "Generating profile data..."
	@mkdir -p $(PROFILE_DIR)
ifeq ($(BUILD_HIP),1)
	@echo "Running HIP profiling..."
	@for target in $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Profiling $$target..."; \
			rocprofv3 --runtime-trace --output-format csv -d $(PROFILE_DIR) -o $$(basename $$target).csv -- $$target 2>/dev/null || echo "rocprofv3 completed"; \
		fi; \
	done
endif
ifeq ($(BUILD_CUDA),1)
	@echo "Running CUDA profiling..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Profiling $$target..."; \
			nsys profile --cuda-event-trace=false -o $(PROFILE_DIR)/$$(basename $$target).nsys-rep $$target 2>/dev/null || echo "nsys completed"; \
		fi; \
	done
endif

# Clean\n.PHONY: clean\nclean:\n\t@echo \"Cleaning build artifacts...\"\n\trm -rf $(BUILD_DIR) $(PROFILE_DIR)

# Help
.PHONY: help
help:
	@echo "Module 3: Advanced GPU Algorithms"
	@echo "Available targets:"
	@echo "  all     - Build all examples for detected GPU vendor"
	@echo "  cuda    - Build CUDA examples (requires NVIDIA GPU)"
	@echo "  hip     - Build HIP examples (requires AMD GPU)"
	@echo "  debug   - Build with debug flags"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"

# Legacy target mappings for convenience
.PHONY: reduction scan sorting convolution matrix graph cooperative
reduction: $(BUILD_DIR)/01_reduction_algorithms_cuda
scan: $(BUILD_DIR)/02_scan_prefix_sum_cuda
sorting: $(BUILD_DIR)/03_sorting_algorithms_cuda
convolution: $(BUILD_DIR)/04_convolution_stencil_cuda
matrix: $(BUILD_DIR)/05_matrix_operations_cuda
graph: $(BUILD_DIR)/06_graph_algorithms_cuda
cooperative: $(BUILD_DIR)/07_cooperative_groups_cuda



# Test targets
test: test_cuda

test_cuda: cuda
	@echo "Running Module 3 CUDA tests..."
	@if command -v nvidia-smi > /dev/null; then \
		echo "=== Testing Advanced Algorithm Examples ==="; \
		echo "1. Reduction Algorithms..."; \
		$(BUILD_DIR)/01_reduction_algorithms_cuda || echo "✗ Reduction algorithms failed"; \
		echo "2. Scan (Prefix Sum)..."; \
		$(BUILD_DIR)/02_scan_prefix_sum_cuda || echo "✗ Scan algorithms failed"; \
		echo "3. Sorting Algorithms..."; \
		$(BUILD_DIR)/03_sorting_algorithms_cuda || echo "✗ Sorting algorithms failed"; \
		echo "4. Convolution/Stencil..."; \
		$(BUILD_DIR)/04_convolution_stencil_cuda || echo "✗ Convolution failed"; \
		echo "5. Matrix Operations..."; \
		$(BUILD_DIR)/05_matrix_operations_cuda || echo "✗ Matrix operations failed"; \
		echo "6. Graph Algorithms..."; \
		$(BUILD_DIR)/06_graph_algorithms_cuda || echo "✗ Graph algorithms failed"; \
		echo "7. Cooperative Groups..."; \
		$(BUILD_DIR)/07_cooperative_groups_cuda || echo "✗ Cooperative groups failed"; \
		echo "✓ Module 3 CUDA tests completed"; \
	else \
		echo "No NVIDIA GPU detected, skipping CUDA tests"; \
	fi

test_hip: hip
	@echo "Running Module 3 HIP tests..."
	@if command -v rocm-smi > /dev/null || command -v nvidia-smi > /dev/null; then \
		echo "=== Testing HIP Algorithm Examples ==="; \
		echo "1. Reduction Algorithms..."; \
		$(BUILD_DIR)/01_reduction_algorithms_hip || echo "✗ HIP reduction algorithms failed"; \
		echo "2. Scan (Prefix Sum)..."; \
		$(BUILD_DIR)/02_scan_prefix_sum_hip || echo "✗ HIP scan algorithms failed"; \
		echo "3. Sorting Algorithms..."; \
		$(BUILD_DIR)/03_sorting_algorithms_hip || echo "✗ HIP sorting algorithms failed"; \
		echo "4. Convolution/Stencil..."; \
		$(BUILD_DIR)/04_convolution_stencil_hip || echo "✗ HIP convolution failed"; \
		echo "✓ Module 3 HIP tests completed"; \
	else \
		echo "No compatible GPU detected, skipping HIP tests"; \
	fi

test_all: test_cuda test_hip

# Quick test - just compile everything
test_compile: cuda hip
	@echo "✓ All Module 3 examples compiled successfully"

# Performance benchmark suite
test_performance: cuda
	@echo "=== Module 3 Performance Benchmarks ==="
	@if command -v nvidia-smi > /dev/null; then \
		echo "Running algorithm performance benchmarks..."; \
		echo "Reduction Performance:"; \
		./01_reduction_algorithms_cuda | grep -E "(speedup|faster)"; \
		echo "Scan Performance:"; \
		./02_scan_prefix_sum_cuda | grep -E "(speedup|vs)"; \
		echo "Sorting Performance:"; \
		./03_sorting_algorithms_cuda | grep -E "(speedup|time)"; \
		echo "Matrix Operations Performance:"; \
		./05_matrix_operations_cuda | grep -E "(GFLOPS|speedup)"; \
		echo "Graph Algorithms Performance:"; \
		./06_graph_algorithms_cuda | grep -E "(time|Performance)"; \
	else \
		echo "No NVIDIA GPU detected for performance testing"; \
	fi

# Algorithm complexity analysis
analyze_complexity: cuda
	@echo "=== Algorithm Complexity Analysis ==="
	@echo "Testing different array sizes for complexity analysis..."
	@if command -v nvidia-smi > /dev/null; then \
		echo "This would run algorithms with different input sizes"; \
		echo "to demonstrate O(n), O(n log n), O(n²) behaviors"; \
		echo "Implementation left as advanced exercise"; \
	else \
		echo "No NVIDIA GPU detected for complexity analysis"; \
	fi

# Profiling helpers
profile_algorithms: cuda
	@echo "=== Algorithm Profiling Helpers ==="
	@echo "Use these commands to profile algorithm performance:"
	@echo ""
	@echo "NVIDIA Profiling:"
	@echo "  ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./01_reduction_algorithms_cuda"
	@echo "  ncu --metrics gpu__time_duration.avg ./02_scan_prefix_sum_cuda"
	@echo "  ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./04_convolution_stencil_cuda"
	@echo ""
	@echo "Memory Analysis:"
	@echo "  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./02_scan_prefix_sum_cuda"
	@echo "  ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct ./03_sorting_algorithms_cuda"



# List available examples
list:
	@echo "Available Module 3 Examples:"
	@echo "============================"
	@echo ""
	@echo "CUDA Examples:"
	@ls -1 *_cuda.cu 2>/dev/null | sed 's/_cuda.cu//' | nl -w2 -s'. '
	@echo ""
	@echo "HIP Examples:"
	@ls -1 *_hip.cpp 2>/dev/null | sed 's/_hip.cpp//' | nl -w2 -s'. '

# Extended help for module 3 specifics
help_extended:
	@echo "Module 3: Advanced GPU Algorithms and Parallel Patterns"
	@echo "======================================================="
	@echo "Learning Objectives:"
	@echo "  - Master fundamental parallel algorithm patterns"
	@echo "  - Understand reduction and scan operations"
	@echo "  - Implement efficient GPU sorting algorithms"
	@echo "  - Apply stencil and convolution patterns"
	@echo "  - Analyze algorithm complexity and performance"
	@echo "  - Design custom parallel algorithms"

.PHONY: all cuda hip test test_cuda test_hip test_all test_compile test_performance
.PHONY: analyze_complexity profile_algorithms clean list help
.PHONY: reduction_cuda reduction_hip scan_cuda sorting_cuda convolution_cuda
.PHONY: reduction scan sorting convolution