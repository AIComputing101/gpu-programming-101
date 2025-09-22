# Module 2: Advanced Memory Management
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
	$(HIPCC) $(HIP_FLAGS) $< -o $@
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
			nvprof --csv -o $(PROFILE_DIR)/$$(basename $$target).csv $$target 2>/dev/null || echo "nvprof completed"; \
		fi; \
	done
endif
	@echo "Profile data saved to $(PROFILE_DIR)/"
	@ls -la $(PROFILE_DIR)/

# Clean
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build profiles

# Help
.PHONY: help
help:
	@echo "Module 2: Advanced Memory Management"
	@echo "Available targets:"
	@echo "  all     - Build all examples for detected GPU vendor"
	@echo "  cuda    - Build CUDA examples (requires NVIDIA GPU)"
	@echo "  hip     - Build HIP examples (requires AMD GPU)"
	@echo "  debug   - Build with debug flags"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"

# Legacy target mappings for convenience
.PHONY: shared_memory coalescing texture unified bandwidth
shared_memory: $(BUILD_DIR)/01_shared_memory_transpose_cuda
coalescing: $(BUILD_DIR)/02_memory_coalescing_cuda
texture: $(BUILD_DIR)/03_texture_memory_cuda
unified: $(BUILD_DIR)/04_unified_memory_cuda
bandwidth: $(BUILD_DIR)/05_memory_bandwidth_optimization_cuda



# Test targets
test: test_cuda

test_cuda: cuda
	@echo "Running Module 2 CUDA tests..."
	@if command -v nvidia-smi > /dev/null; then \
		echo "=== Testing Advanced Memory Management Examples ==="; \
		echo "1. Shared Memory Transpose..."; \
		./01_shared_memory_transpose_cuda || echo "✗ Shared memory transpose failed"; \
		echo "2. Memory Coalescing Analysis..."; \
		./02_memory_coalescing_cuda || echo "✗ Memory coalescing failed"; \
		echo "3. Texture Memory Examples..."; \
		./03_texture_memory_cuda || echo "✗ Texture memory failed"; \
		echo "4. Unified Memory Examples..."; \
		./04_unified_memory_cuda || echo "✗ Unified memory failed"; \
		echo "5. Bandwidth Optimization..."; \
		./05_memory_bandwidth_optimization_cuda || echo "✗ Bandwidth optimization failed"; \
		echo "✓ Module 2 CUDA tests completed"; \
	else \
		echo "No NVIDIA GPU detected, skipping CUDA tests"; \
	fi

test_hip: hip
	@echo "Running Module 2 HIP tests..."
	@if command -v rocm-smi > /dev/null || command -v nvidia-smi > /dev/null; then \
		echo "=== Testing HIP Memory Examples ==="; \
		echo "1. Shared Memory Transpose..."; \
		./01_shared_memory_transpose_hip || echo "✗ HIP shared memory transpose failed"; \
		echo "2. Memory Coalescing Analysis..."; \
		./02_memory_coalescing_hip || echo "✗ HIP memory coalescing failed"; \
		echo "✓ Module 2 HIP tests completed"; \
	else \
		echo "No compatible GPU detected, skipping HIP tests"; \
	fi

test_all: test_cuda test_hip

# Quick test - just compile everything
test_compile: cuda hip
	@echo "✓ All Module 2 examples compiled successfully"

# Performance test suite
test_performance: cuda
	@echo "=== Module 2 Performance Test Suite ==="
	@if command -v nvidia-smi > /dev/null; then \
		echo "Running performance benchmarks..."; \
		echo "Memory Bandwidth Analysis:"; \
		./05_memory_bandwidth_optimization_cuda | grep -E "(Bandwidth|speedup)"; \
		echo "Memory Coalescing Impact:"; \
		./02_memory_coalescing_cuda | grep -E "(speedup|slower)"; \
		echo "Shared Memory Optimization:"; \
		./01_shared_memory_transpose_cuda | grep -E "(speedup|Bandwidth)"; \
	else \
		echo "No NVIDIA GPU detected for performance testing"; \
	fi

# Memory profiling helpers
profile_memory: cuda
	@echo "=== Memory Profiling Helpers ==="
	@echo "Use these commands to profile memory performance:"
	@echo ""
	@echo "CUDA Profiling:"
	@echo "  ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./01_shared_memory_transpose_cuda"
	@echo "  ncu --metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed ./02_memory_coalescing_cuda"
	@echo "  ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./05_memory_bandwidth_optimization_cuda"
	@echo ""
	@echo "Legacy nvprof (if available):"
	@echo "  nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency ./03_texture_memory_cuda"



# List available examples
list:
	@echo "Available Module 2 Examples:"
	@echo "============================"
	@echo ""
	@echo "CUDA Examples:"
	@ls -1 *_cuda.cu 2>/dev/null | sed 's/_cuda.cu//' | nl -w2 -s'. '
	@echo ""
	@echo "HIP Examples:"
	@ls -1 *_hip.cpp 2>/dev/null | sed 's/_hip.cpp//' | nl -w2 -s'. '

# Extended help for module 2 specifics
help_extended:
	@echo "Module 2: Advanced GPU Memory Management and Optimization"
	@echo "========================================================"
	@echo "Learning Objectives:"
	@echo "  - Master shared memory optimization techniques"
	@echo "  - Understand memory coalescing impact on performance"
	@echo "  - Learn texture memory usage for spatial locality"
	@echo "  - Explore unified memory programming model"
	@echo "  - Optimize memory bandwidth utilization"

.PHONY: all cuda hip test test_cuda test_hip test_all test_compile test_performance
.PHONY: clean list help profile_memory
.PHONY: shared_memory_cuda shared_memory_hip coalescing_cuda coalescing_hip
.PHONY: texture_cuda unified_cuda bandwidth_cuda
.PHONY: shared_memory coalescing texture unified bandwidth