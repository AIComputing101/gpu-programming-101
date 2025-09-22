# Module 5: Performance Considerations and GPU Optimization
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
CUDA_FLAGS = -std=c++17 -O3 -arch=sm_90 -lineinfo
CUDA_DEBUG_FLAGS = -std=c++17 -g -G -arch=sm_90
HIP_FLAGS = -std=c++17 -O3
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
CXX_FLAGS = -std=c++17 -O3 -fopenmp

# Profiling flags
NVPROF_FLAGS = --print-gpu-trace --log-file %s.nvprof
ROCPROF_FLAGS = --hip-trace --stats --output-file %s.csv

# Directories
EXAMPLES_DIR = .
BUILD_DIR = build
PROFILE_DIR = profiles

# CUDA Examples
CUDA_SOURCES = $(wildcard $(EXAMPLES_DIR)/*_cuda.cu)
CUDA_TARGETS = $(patsubst $(EXAMPLES_DIR)/%.cu,$(BUILD_DIR)/%,$(CUDA_SOURCES))

# HIP Examples (only if AMD GPU detected)
ifeq ($(BUILD_HIP),1)
HIP_SOURCES = $(wildcard $(EXAMPLES_DIR)/*_hip.cpp)
HIP_TARGETS = $(patsubst $(EXAMPLES_DIR)/%.cpp,$(BUILD_DIR)/%,$(HIP_SOURCES))
else
HIP_SOURCES = 
HIP_TARGETS = 
endif

# Active targets based on detected GPU vendor
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
	@echo "⚠ No compatible GPU detected - no examples will be built"
endif

# CUDA compilation rules
$(BUILD_DIR)/%_cuda: $(EXAMPLES_DIR)/%_cuda.cu
	@echo "Building CUDA example: $@"
	$(NVCC) $(CUDA_FLAGS) $< -o $@

# HIP compilation rules (only if AMD GPU detected)
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

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(PROFILE_DIR)

# Run all examples
.PHONY: run
run: all
	@echo "Running all performance optimization examples..."
	@for target in $(ALL_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Running $$target..."; \
			$$target; \
			echo ""; \
		fi; \
	done

# Performance profiling targets
.PHONY: profile-cuda
profile-cuda: $(CUDA_TARGETS)
	@echo "Profiling CUDA examples with nvprof..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Profiling $$target..."; \
			nvprof $(NVPROF_FLAGS) $$target > $(PROFILE_DIR)/$$(basename $$target).nvprof 2>&1; \
		fi; \
	done

.PHONY: profile-hip  
profile-hip: 
ifeq ($(HIP_AVAILABLE),1)
	@echo "Profiling HIP examples with rocprof..."
	@for target in $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Profiling $$target..."; \
			rocprof $(ROCPROF_FLAGS) $$target; \
			mv results.csv $(PROFILE_DIR)/$$(basename $$target).csv 2>/dev/null || true; \
		fi; \
	done
else
	@echo "ℹ HIP not available - skipping HIP profiling"
endif

# Comprehensive profiling with Nsight Compute (CUDA)
.PHONY: profile-detailed-cuda
profile-detailed-cuda: $(CUDA_TARGETS)
	@echo "Detailed profiling with Nsight Compute..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Profiling $$target with Nsight Compute..."; \
			ncu --set full --export $(PROFILE_DIR)/$$(basename $$target) $$target; \
		fi; \
	done

# Memory checking with cuda-memcheck
.PHONY: memcheck-cuda
memcheck-cuda: $(CUDA_TARGETS)
	@echo "Running CUDA memory checker..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Memory checking $$target..."; \
			cuda-memcheck --log-file $(PROFILE_DIR)/$$(basename $$target).memcheck $$target; \
		fi; \
	done

# Performance benchmarking
.PHONY: benchmark
benchmark: all
	@echo "Running performance benchmarks..."
	@echo "Benchmark results will be saved to $(PROFILE_DIR)/benchmark_results.txt"
	@echo "GPU Performance Optimization Benchmark Results" > $(PROFILE_DIR)/benchmark_results.txt
	@echo "Generated on: $$(date)" >> $(PROFILE_DIR)/benchmark_results.txt
	@echo "=============================================" >> $(PROFILE_DIR)/benchmark_results.txt
	@for target in $(ALL_TARGETS); do \
		if [ -f $$target ]; then \
			echo ""; \
			echo "Benchmarking $$target..." | tee -a $(PROFILE_DIR)/benchmark_results.txt; \
			echo "-------------------------------------------" >> $(PROFILE_DIR)/benchmark_results.txt; \
			$$target | tee -a $(PROFILE_DIR)/benchmark_results.txt; \
			echo "" >> $(PROFILE_DIR)/benchmark_results.txt; \
		fi; \
	done

# Optimization validation
.PHONY: validate
validate: all
	@echo "Validating optimization implementations..."
	@echo "This will run examples with different optimization levels to verify correctness"
	@$(MAKE) clean
	@$(MAKE) CUDA_FLAGS="-std=c++17 -O0 -arch=sm_70" HIP_FLAGS="-std=c++17 -O0" all
	@echo "Running unoptimized versions for correctness baseline..."
	@for target in $(ALL_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Validating $$target (unoptimized)..."; \
			$$target > $(PROFILE_DIR)/$$(basename $$target).unoptimized.out 2>&1; \
		fi; \
	done
	@$(MAKE) clean
	@$(MAKE) all
	@echo "Running optimized versions..."
	@for target in $(ALL_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Validating $$target (optimized)..."; \
			$$target > $(PROFILE_DIR)/$$(basename $$target).optimized.out 2>&1; \
		fi; \
	done

# Performance regression testing
.PHONY: regression-test
regression-test: benchmark
	@echo "Running performance regression tests..."
	@echo "Comparing current performance against baseline (if available)"
	@if [ -f $(PROFILE_DIR)/baseline_benchmark.txt ]; then \
		echo "Baseline found, running comparison..."; \
		diff $(PROFILE_DIR)/baseline_benchmark.txt $(PROFILE_DIR)/benchmark_results.txt > $(PROFILE_DIR)/performance_diff.txt || true; \
		echo "Performance comparison saved to $(PROFILE_DIR)/performance_diff.txt"; \
	else \
		echo "No baseline found, creating baseline from current results..."; \
		cp $(PROFILE_DIR)/benchmark_results.txt $(PROFILE_DIR)/baseline_benchmark.txt; \
		echo "Baseline created at $(PROFILE_DIR)/baseline_benchmark.txt"; \
	fi

# Generate performance report
.PHONY: report
report: benchmark profile-cuda profile-hip
	@echo "Generating comprehensive performance report..."
	@echo "GPU Performance Optimization Report" > $(PROFILE_DIR)/performance_report.md
	@echo "===================================" >> $(PROFILE_DIR)/performance_report.md
	@echo "" >> $(PROFILE_DIR)/performance_report.md
	@echo "## System Information" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Date: $$(date)" >> $(PROFILE_DIR)/performance_report.md
	@echo "- GPU: $$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'Unknown')" >> $(PROFILE_DIR)/performance_report.md
	@echo "- CUDA Version: $$(nvcc --version | grep "release" | awk '{print $$6}' 2>/dev/null || echo 'Not available')" >> $(PROFILE_DIR)/performance_report.md
	@echo "" >> $(PROFILE_DIR)/performance_report.md
	@echo "## Benchmark Results" >> $(PROFILE_DIR)/performance_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/performance_report.md
	@cat $(PROFILE_DIR)/benchmark_results.txt >> $(PROFILE_DIR)/performance_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/performance_report.md
	@echo "" >> $(PROFILE_DIR)/performance_report.md
	@echo "## Profiling Files Generated" >> $(PROFILE_DIR)/performance_report.md
	@ls -la $(PROFILE_DIR)/ >> $(PROFILE_DIR)/performance_report.md
	@echo "Performance report generated: $(PROFILE_DIR)/performance_report.md"

# Development helpers
.PHONY: watch
watch:
	@echo "Watching for changes in $(EXAMPLES_DIR)..."
	@while true; do \
		inotifywait -e modify $(EXAMPLES_DIR)/* 2>/dev/null && $(MAKE) all || true; \
	done

# Install development dependencies (Ubuntu/Debian)
.PHONY: install-deps
install-deps:
	@echo "Installing development dependencies..."
	@echo "Note: This requires sudo privileges"
	sudo apt-get update
	sudo apt-get install -y build-essential
	@echo "CUDA Toolkit and ROCm should be installed separately"
	@echo "Visit: https://developer.nvidia.com/cuda-downloads"
	@echo "Visit: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html"

# Help target
.PHONY: help
help:
	@echo "GPU Performance Optimization Module 5 - Makefile Help"
	@echo "======================================================"
	@echo ""
	@echo "Build Targets:"
	@echo "  all                 - Build all examples (default)"
	@echo "  debug              - Build with debug flags"
	@echo "  clean              - Remove build artifacts"
	@echo ""
	@echo "Execution Targets:"
	@echo "  run                - Run all examples"
	@echo "  benchmark          - Run performance benchmarks"
	@echo "  validate           - Validate optimization correctness"
	@echo ""
	@echo "Profiling Targets:"
	@echo "  profile-cuda       - Profile CUDA examples with nvprof"
	@echo "  profile-hip        - Profile HIP examples with rocprof"
	@echo "  profile-detailed-cuda - Detailed profiling with Nsight Compute"
	@echo "  memcheck-cuda      - Run CUDA memory checker"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  regression-test    - Run performance regression tests"
	@echo "  report            - Generate comprehensive performance report"
	@echo ""
	@echo "Development Targets:"
	@echo "  watch             - Watch for file changes and rebuild"
	@echo "  install-deps      - Install system dependencies"
	@echo "  help              - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build all examples"
	@echo "  make run               # Build and run all examples"
	@echo "  make benchmark         # Run performance benchmarks"
	@echo "  make profile-cuda      # Profile CUDA examples"
	@echo "  make report           # Generate full performance report"

# Additional individual example targets
.PHONY: profiling
profiling: $(BUILD_DIR)/01_gpu_profiling_cuda 01_hip_profiling

.PHONY: memory-opt
memory-opt: $(BUILD_DIR)/02_memory_optimization_cuda

.PHONY: kernel-opt  
kernel-opt: $(BUILD_DIR)/03_kernel_optimization_cuda

# Quick test target for development
.PHONY: test
test: all
	@echo "Quick test run..."
	@if [ -f $(BUILD_DIR)/01_gpu_profiling_cuda ]; then $(BUILD_DIR)/01_gpu_profiling_cuda || echo "CUDA profiling test completed"; fi
	@if [ -f $(BUILD_DIR)/02_memory_optimization_cuda ]; then $(BUILD_DIR)/02_memory_optimization_cuda || echo "Memory optimization test completed"; fi
	@if [ -f $(BUILD_DIR)/03_kernel_optimization_cuda ]; then $(BUILD_DIR)/03_kernel_optimization_cuda || echo "Kernel optimization test completed"; fi