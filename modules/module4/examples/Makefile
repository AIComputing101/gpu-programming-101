# Module 4: Advanced Multi-GPU Programming
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
CUDA_FLAGS = -std=c++17 -O2 -arch=sm_75 -rdc=true -lcudart -lcuda
CUDA_DP_FLAGS = -std=c++17 -O2 -arch=sm_75 -rdc=true -lcudadevrt -lcudart -lcuda
CUDA_DEBUG_FLAGS = -std=c++17 -g -G -arch=sm_75 -rdc=true -lcudart -lcuda
HIP_FLAGS = -std=c++17 -O2 -fopenmp
HIP_DEBUG_FLAGS = -std=c++17 -g -fopenmp

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

# OpenMP support for multi-GPU examples
OMP_FLAGS = -Xcompiler -fopenmp -lgomp

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

# Special rule for dynamic parallelism examples
$(BUILD_DIR)/%dynamic_parallelism: $(EXAMPLES_DIR)/%dynamic_parallelism.cu
	@echo "Building CUDA dynamic parallelism example: $@"
	$(NVCC) $(CUDA_DP_FLAGS) $< -o $@

$(BUILD_DIR)/%_cuda: $(EXAMPLES_DIR)/%_cuda.cu
	@echo "Building CUDA example: $@"
	$(NVCC) $(CUDA_FLAGS) $(OMP_FLAGS) $< -o $@

# Pattern for numbered CUDA examples
$(BUILD_DIR)/0%: $(EXAMPLES_DIR)/0%.cu
	@echo "Building CUDA example: $@"
	@case "$<" in \
		*dynamic_parallelism*) \
			$(NVCC) $(CUDA_DP_FLAGS) $< -o $@ ;; \
		*) \
			$(NVCC) $(CUDA_FLAGS) $(OMP_FLAGS) $< -o $@ ;; \
	esac

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
profile: CUDA_FLAGS = $(CUDA_FLAGS) -lineinfo
profile: HIP_FLAGS = $(HIP_FLAGS) -g
profile: all

# Clean
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(PROFILE_DIR)

# Help
.PHONY: help
help:
	@echo "Module 4: Advanced Multi-GPU Programming"
	@echo "Available targets:"
	@echo "  all     - Build all examples for detected GPU vendor"
	@echo "  cuda    - Build CUDA examples (requires NVIDIA GPU)"
	@echo "  hip     - Build HIP examples (requires AMD GPU)"
	@echo "  debug   - Build with debug flags"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"

# Individual example targets with specific requirements

01_cuda_streams_basics: 01_cuda_streams_basics.cu
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) $< -o $@

02_multi_gpu_programming: 02_multi_gpu_programming.cu
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) $< -o $@

03_unified_memory: 03_unified_memory.cu
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) $< -o $@

04_peer_to_peer_communication: 04_peer_to_peer_communication.cu
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) $< -o $@

05_dynamic_parallelism: 05_dynamic_parallelism.cu
	$(NVCC) $(NVCC_DP_FLAGS) $< -o $@

# HIP example targets
01_hip_streams_basics: 01_hip_streams_basics.cpp
	$(HIPCC) $(HIP_FLAGS) $< -o $@

02_hip_multi_gpu_programming: 02_hip_multi_gpu_programming.cpp
	$(HIPCC) $(HIP_FLAGS) $< -o $@

03_hip_unified_memory: 03_hip_unified_memory.cpp
	$(HIPCC) $(HIP_FLAGS) $< -o $@

04_hip_peer_to_peer_communication: 04_hip_peer_to_peer_communication.cpp
	$(HIPCC) $(HIP_FLAGS) $< -o $@

# Generic compilation rules
%: %.cu
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) $< -o $@

%hip%: %hip%.cpp
	$(HIPCC) $(HIP_FLAGS) $< -o $@

# Legacy targets for convenience
streams: 01_cuda_streams_basics
multi_gpu: 02_multi_gpu_programming
unified_memory: 03_unified_memory
p2p: 04_peer_to_peer_communication
dynamic: 05_dynamic_parallelism

# HIP convenience targets
streams_hip: 01_hip_streams_basics
multi_gpu_hip: 02_hip_multi_gpu_programming
unified_memory_hip: 03_hip_unified_memory
p2p_hip: 04_hip_peer_to_peer_communication

# Test targets
test: test_cuda

test_cuda: cuda
	@echo "Running Module 4 Advanced GPU Programming tests..."
	@if command -v nvidia-smi > /dev/null; then \
		echo "=== Testing Advanced GPU Programming Examples ==="; \
		echo "1. CUDA Streams..."; \
		./01_cuda_streams_basics || echo "✗ CUDA Streams failed"; \
		echo "2. Multi-GPU Programming..."; \
		./02_multi_gpu_programming || echo "✗ Multi-GPU Programming failed"; \
		echo "3. Unified Memory..."; \
		./03_unified_memory || echo "✗ Unified Memory failed"; \
		echo "4. Peer-to-Peer Communication..."; \
		./04_peer_to_peer_communication || echo "✗ P2P Communication failed"; \
		echo "5. Dynamic Parallelism (requires compute capability 3.5+)..."; \
		./05_dynamic_parallelism || echo "✗ Dynamic Parallelism failed (may require newer GPU)"; \
		echo "✓ Module 4 tests completed"; \
	else \
		echo "No NVIDIA GPU detected, skipping GPU tests"; \
	fi

# HIP test targets
test_hip: hip
	@echo "Running Module 4 HIP Advanced GPU Programming tests..."
	@if command -v rocm-smi > /dev/null 2>&1 || command -v nvidia-smi > /dev/null 2>&1; then \
		echo "=== Testing HIP Advanced GPU Programming Examples ==="; \
		echo "1. HIP Streams..."; \
		./01_hip_streams_basics || echo "✗ HIP Streams failed"; \
		echo "2. HIP Multi-GPU Programming..."; \
		./02_hip_multi_gpu_programming || echo "✗ HIP Multi-GPU Programming failed"; \
		echo "3. HIP Unified Memory..."; \
		./03_hip_unified_memory || echo "✗ HIP Unified Memory failed"; \
		echo "4. HIP Peer-to-Peer Communication..."; \
		./04_hip_peer_to_peer_communication || echo "✗ HIP P2P Communication failed"; \
		echo "✓ HIP Module 4 tests completed"; \
	else \
		echo "No GPU detected (ROCm or CUDA), skipping HIP tests"; \
	fi

# Test both CUDA and HIP
test_both: test_cuda test_hip

# Quick test - just compile everything
test_compile_cuda: cuda
	@echo "✓ All Module 4 CUDA examples compiled successfully"

test_compile_hip: hip
	@echo "✓ All Module 4 HIP examples compiled successfully"

test_compile: test_compile_cuda
	@if command -v hipcc > /dev/null 2>&1; then \
		$(MAKE) test_compile_hip; \
	else \
		echo "HIP compiler not found - skipping HIP compilation test"; \
	fi

# Performance benchmark suite
test_performance_cuda: cuda
	@echo "=== Module 4 CUDA Performance Benchmarks ==="
	@if command -v nvidia-smi > /dev/null; then \
		echo "Running advanced GPU performance benchmarks..."; \
		echo "CUDA Streams Performance:"; \
		./01_cuda_streams_basics | grep -E "(speedup|faster|bandwidth)"; \
		echo "Multi-GPU Scaling:"; \
		./02_multi_gpu_programming | grep -E "(speedup|efficiency|GFLOPS)"; \
		echo "Unified Memory Performance:"; \
		./03_unified_memory | grep -E "(speedup|bandwidth)"; \
		echo "P2P Communication Bandwidth:"; \
		./04_peer_to_peer_communication | grep -E "(bandwidth|GB/s)"; \
	else \
		echo "No NVIDIA GPU detected for performance testing"; \
	fi

test_performance_hip: hip
	@echo "=== Module 4 HIP Performance Benchmarks ==="
	@if command -v rocm-smi > /dev/null 2>&1 || command -v nvidia-smi > /dev/null 2>&1; then \
		echo "Running HIP advanced GPU performance benchmarks..."; \
		echo "HIP Streams Performance:"; \
		./01_hip_streams_basics | grep -E "(speedup|faster|bandwidth)" || echo "HIP Streams completed"; \
		echo "HIP Multi-GPU Scaling:"; \
		./02_hip_multi_gpu_programming | grep -E "(speedup|efficiency|GFLOPS)" || echo "HIP Multi-GPU completed"; \
		echo "HIP Unified Memory Performance:"; \
		./03_hip_unified_memory | grep -E "(speedup|bandwidth)" || echo "HIP Unified Memory completed"; \
		echo "HIP P2P Communication Bandwidth:"; \
		./04_hip_peer_to_peer_communication | grep -E "(bandwidth|GB/s)" || echo "HIP P2P completed"; \
	else \
		echo "No GPU detected for HIP performance testing"; \
	fi

test_performance: test_performance_cuda

# Multi-GPU specific tests
test_multi_gpu: 02_multi_gpu_programming 04_peer_to_peer_communication
	@echo "=== Multi-GPU Specific Tests ==="
	@if command -v nvidia-smi > /dev/null; then \
		GPU_COUNT=$$(nvidia-smi -L | wc -l); \
		echo "Detected $$GPU_COUNT GPU(s)"; \
		if [ $$GPU_COUNT -ge 2 ]; then \
			echo "Running multi-GPU programming test..."; \
			./02_multi_gpu_programming; \
			echo "Running P2P communication test..."; \
			./04_peer_to_peer_communication; \
		else \
			echo "Multi-GPU tests require at least 2 GPUs"; \
		fi; \
	else \
		echo "No NVIDIA GPU detected"; \
	fi

# Streams and concurrency tests
test_streams: 01_cuda_streams_basics 03_unified_memory
	@echo "=== Streams and Concurrency Tests ==="
	@if command -v nvidia-smi > /dev/null; then \
		echo "Testing CUDA streams..."; \
		./01_cuda_streams_basics; \
		echo "Testing unified memory with concurrent access..."; \
		./03_unified_memory; \
	else \
		echo "No NVIDIA GPU detected"; \
	fi

# Dynamic parallelism test (requires compute capability 3.5+)
test_dynamic: 05_dynamic_parallelism
	@echo "=== Dynamic Parallelism Test ==="
	@if command -v nvidia-smi > /dev/null; then \
		COMPUTE_CAP=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1); \
		echo "GPU Compute Capability: $$COMPUTE_CAP"; \
		if echo "$$COMPUTE_CAP >= 3.5" | bc -l > /dev/null 2>&1; then \
			echo "Running dynamic parallelism test..."; \
			./05_dynamic_parallelism; \
		else \
			echo "Dynamic parallelism requires compute capability 3.5+"; \
		fi; \
	else \
		echo "No NVIDIA GPU detected"; \
	fi

# System information
system_info:
	@echo "=== GPU System Information ==="
	@if command -v nvidia-smi > /dev/null; then \
		echo "NVIDIA System Management Interface:"; \
		nvidia-smi; \
		echo ""; \
		echo "CUDA Runtime Version:"; \
		nvcc --version | grep "release"; \
		echo ""; \
		echo "GPU Memory Information:"; \
		nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv; \
		echo ""; \
		echo "P2P Topology:"; \
		nvidia-smi topo -m 2>/dev/null || echo "P2P topology information not available"; \
	else \
		echo "NVIDIA drivers not found"; \
	fi

# Profiling helpers
profile_examples: all
	@echo "=== Profiling Helpers ==="
	@echo "Use these commands to profile the examples:"
	@echo ""
	@echo "NVIDIA Nsight Compute (detailed kernel analysis):"
	@echo "  ncu --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed ./01_cuda_streams_basics"
	@echo "  ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./02_multi_gpu_programming"
	@echo "  ncu --metrics gpu__time_duration.avg ./03_unified_memory"
	@echo "  ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./04_peer_to_peer_communication"
	@echo ""
	@echo "NVIDIA Nsight Systems (timeline analysis):"
	@echo "  nsys profile --trace=cuda,nvtx,osrt --stats=true ./01_cuda_streams_basics"
	@echo "  nsys profile --trace=cuda,nvtx --stats=true ./02_multi_gpu_programming"
	@echo "  nsys profile --trace=cuda,nvtx,osrt --stats=true ./03_unified_memory"
	@echo ""
	@echo "Legacy nvprof:"
	@echo "  nvprof --print-gpu-trace ./01_cuda_streams_basics"
	@echo "  nvprof --print-api-trace ./02_multi_gpu_programming"
	@echo ""
	@echo "Multi-GPU Analysis:"
	@echo "  nsys profile --trace=cuda,nvtx --stats=true -o multi_gpu_trace ./02_multi_gpu_programming"
	@echo "  ncu --target-processes all ./04_peer_to_peer_communication"

# Memory analysis helpers  
analyze_memory: all
	@echo "=== Memory Analysis Helpers ==="
	@echo "Check unified memory usage:"
	@echo "  cuda-memcheck --tool=racecheck ./03_unified_memory"
	@echo ""
	@echo "P2P bandwidth analysis:"
	@echo "  ./04_peer_to_peer_communication > p2p_results.txt"
	@echo ""
	@echo "Memory access pattern analysis:"
	@echo "  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./01_cuda_streams_basics"

# Deep clean including profiling outputs
clean_all: clean
	rm -f *.nsys-rep *.ncu-rep *.sqlite *.qdrep
	rm -f *_trace.* *.log

# List available examples
list:
	@echo "Available Module 4 Examples:"
	@echo "============================"
	@echo ""
	@echo "Advanced GPU Programming Examples:"
	@ls -1 *.cu 2>/dev/null | sed 's/.cu//' | nl -w2 -s'. '
	@echo ""
	@echo "Quick Access Targets:"
	@echo "  streams      - CUDA streams and asynchronous execution"
	@echo "  multi_gpu    - Multi-GPU programming and load balancing"  
	@echo "  unified_memory - Unified memory programming"
	@echo "  p2p          - Peer-to-peer GPU communication"
	@echo "  dynamic      - Dynamic parallelism (GPU launches GPU)"

# Extended help target for module 4 specifics
help_extended:
	@echo "GPU Programming 101 - Module 4 Examples Makefile"
	@echo "================================================="
	@echo ""
	@echo "Module 4: Advanced GPU Programming - Multi-GPU, Streams, and Scalability"
	@echo ""
	@echo "Build Targets:"
	@echo "  all / cuda             - Build all CUDA examples (default)"
	@echo "  hip                    - Build all HIP examples"
	@echo "  both                   - Build both CUDA and HIP examples"
	@echo "  [example_name]         - Build specific example"
	@echo ""
	@echo "Test Targets:"
	@echo "  test / test_cuda       - Run CUDA tests"
	@echo "  test_hip               - Run HIP tests"
	@echo "  test_both              - Run both CUDA and HIP tests"
	@echo "  test_compile           - Test compilation only"
	@echo "  test_performance       - Run CUDA performance benchmarks"
	@echo "  test_performance_hip   - Run HIP performance benchmarks"
	@echo "  test_multi_gpu         - Multi-GPU specific tests"
	@echo "  test_streams           - Streams and concurrency tests"
	@echo "  test_dynamic           - Dynamic parallelism test"
	@echo ""
	@echo "Individual CUDA Examples:"
	@echo "  01_cuda_streams_basics          - CUDA streams and asynchronous execution"
	@echo "  02_multi_gpu_programming        - Multi-GPU programming techniques"
	@echo "  03_unified_memory              - Unified memory management"
	@echo "  04_peer_to_peer_communication  - P2P GPU communication"
	@echo "  05_dynamic_parallelism         - Dynamic parallelism (GPU-launched kernels)"
	@echo ""
	@echo "Individual HIP Examples:"
	@echo "  01_hip_streams_basics           - HIP streams and asynchronous execution"
	@echo "  02_hip_multi_gpu_programming    - HIP multi-GPU programming techniques"
	@echo "  03_hip_unified_memory          - HIP unified memory management"
	@echo "  04_hip_peer_to_peer_communication - HIP P2P GPU communication"
	@echo ""
	@echo "Quick Access (CUDA):"
	@echo "  streams, multi_gpu, unified_memory, p2p, dynamic"
	@echo ""
	@echo "Quick Access (HIP):"
	@echo "  streams_hip, multi_gpu_hip, unified_memory_hip, p2p_hip"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  system_info            - Display GPU system information"
	@echo "  profile_examples       - Show profiling commands"
	@echo "  analyze_memory         - Show memory analysis commands"
	@echo ""
	@echo "Utility Targets:"
	@echo "  list                   - List all available examples"
	@echo "  clean                  - Remove executables"
	@echo "  clean_all              - Remove executables and profiling outputs"
	@echo "  help                   - Show this help"
	@echo ""
	@echo "Example Usage:"
	@echo "  make all                       # Build all examples"
	@echo "  make streams                   # Build and focus on streams"
	@echo "  make test_multi_gpu           # Test multi-GPU capabilities"
	@echo "  make test_performance         # Run performance benchmarks"
	@echo "  make system_info              # Check GPU system info"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA Toolkit 10.0+ (for CUDA examples)"
	@echo "  - HIP/ROCm 4.0+ (for HIP examples)" 
	@echo "  - Compute Capability 5.0+ (3.5+ for dynamic parallelism)"
	@echo "  - Multi-GPU system recommended for full testing"
	@echo "  - OpenMP support for parallel host code"
	@echo ""
	@echo "Learning Objectives:"
	@echo "  - Master CUDA/HIP streams for asynchronous execution"
	@echo "  - Implement scalable multi-GPU applications"
	@echo "  - Utilize unified memory effectively"
	@echo "  - Optimize peer-to-peer GPU communication"
	@echo "  - Apply dynamic parallelism for recursive algorithms"
	@echo "  - Profile and optimize advanced GPU applications"
	@echo "  - Understand cross-platform GPU programming with HIP"

.PHONY: all cuda hip both test test_cuda test_hip test_both test_compile test_compile_cuda test_compile_hip
.PHONY: test_performance test_performance_cuda test_performance_hip test_multi_gpu test_streams test_dynamic
.PHONY: system_info profile_examples analyze_memory clean clean_all list help help_extended
.PHONY: streams multi_gpu unified_memory p2p dynamic
.PHONY: streams_hip multi_gpu_hip unified_memory_hip p2p_hip