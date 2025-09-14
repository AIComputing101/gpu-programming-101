# GPU Programming Module 2 Examples Makefile
# Advanced Memory Management and Optimization Examples

# Compiler settings
NVCC = nvcc
HIPCC = hipcc
NVCC_FLAGS = -O2 -std=c++11 -arch=sm_75
HIPCC_FLAGS = -O2 -std=c++11

# Source files (following consistent naming pattern)
CUDA_SOURCES = $(wildcard *_cuda.cu)
HIP_SOURCES = $(wildcard *_hip.cpp)

# Executable names
CUDA_EXECUTABLES = $(CUDA_SOURCES:_cuda.cu=_cuda)
HIP_EXECUTABLES = $(HIP_SOURCES:_hip.cpp=_hip)

# Default target
all: cuda

# CUDA targets
cuda: $(CUDA_EXECUTABLES)

%_cuda: %_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# HIP targets  
hip: $(HIP_EXECUTABLES)

%_hip: %_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

# Individual example targets
01_shared_memory_transpose_cuda: 01_shared_memory_transpose_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

01_shared_memory_transpose_hip: 01_shared_memory_transpose_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

02_memory_coalescing_cuda: 02_memory_coalescing_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

02_memory_coalescing_hip: 02_memory_coalescing_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

03_texture_memory_cuda: 03_texture_memory_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

04_unified_memory_cuda: 04_unified_memory_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

05_memory_bandwidth_optimization_cuda: 05_memory_bandwidth_optimization_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Legacy targets for backward compatibility
shared_memory_cuda: 01_shared_memory_transpose_cuda
shared_memory_hip: 01_shared_memory_transpose_hip
coalescing_cuda: 02_memory_coalescing_cuda
coalescing_hip: 02_memory_coalescing_hip
texture_cuda: 03_texture_memory_cuda
unified_cuda: 04_unified_memory_cuda
bandwidth_cuda: 05_memory_bandwidth_optimization_cuda

# Generic legacy targets (default to CUDA)
shared_memory: 01_shared_memory_transpose_cuda
coalescing: 02_memory_coalescing_cuda
texture: 03_texture_memory_cuda
unified: 04_unified_memory_cuda
bandwidth: 05_memory_bandwidth_optimization_cuda

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

# Clean targets
clean:
	rm -f $(CUDA_EXECUTABLES) $(HIP_EXECUTABLES)
	rm -f 01_shared_memory_transpose_cuda 01_shared_memory_transpose_hip
	rm -f 02_memory_coalescing_cuda 02_memory_coalescing_hip  
	rm -f 03_texture_memory_cuda 04_unified_memory_cuda
	rm -f 05_memory_bandwidth_optimization_cuda
	rm -f shared_memory_cuda shared_memory_hip coalescing_cuda coalescing_hip
	rm -f texture_cuda unified_cuda bandwidth_cuda
	rm -f shared_memory coalescing texture unified bandwidth

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

# Help target
help:
	@echo "GPU Programming 101 - Module 2 Examples Makefile"
	@echo "================================================"
	@echo ""
	@echo "Module 2: Advanced GPU Memory Management and Optimization"
	@echo ""
	@echo "Build Targets:"
	@echo "  all               - Build all CUDA examples (default)"
	@echo "  cuda              - Build all CUDA examples"
	@echo "  hip               - Build all HIP examples"
	@echo ""
	@echo "Test Targets:"
	@echo "  test              - Run CUDA tests"
	@echo "  test_cuda         - Run CUDA tests"
	@echo "  test_hip          - Run HIP tests"
	@echo "  test_all          - Run both CUDA and HIP tests"
	@echo "  test_compile      - Test compilation only"
	@echo "  test_performance  - Run performance benchmarks"
	@echo ""
	@echo "Individual Examples (CUDA):"
	@echo "  01_shared_memory_transpose_cuda    - Shared memory matrix transpose"
	@echo "  02_memory_coalescing_cuda          - Memory coalescing analysis"
	@echo "  03_texture_memory_cuda             - Texture memory examples"
	@echo "  04_unified_memory_cuda             - Unified memory demonstrations"
	@echo "  05_memory_bandwidth_optimization_cuda - Bandwidth optimization techniques"
	@echo ""
	@echo "Individual Examples (HIP):"
	@echo "  01_shared_memory_transpose_hip     - Shared memory matrix transpose"
	@echo "  02_memory_coalescing_hip           - Memory coalescing analysis"
	@echo ""
	@echo "Legacy Targets:"
	@echo "  shared_memory, coalescing, texture, unified, bandwidth"
	@echo "  (These default to CUDA versions)"
	@echo ""
	@echo "Utility Targets:"
	@echo "  list              - List all available examples"
	@echo "  clean             - Remove all executables"
	@echo "  help              - Show this help"
	@echo "  profile_memory    - Show memory profiling commands"
	@echo ""
	@echo "Example Usage:"
	@echo "  make cuda                              # Build all CUDA examples"
	@echo "  make hip                               # Build all HIP examples"
	@echo "  make 01_shared_memory_transpose_cuda   # Build specific example"
	@echo "  make test_performance                  # Run performance tests"
	@echo "  make profile_memory                    # Show profiling commands"
	@echo ""
	@echo "Requirements:"
	@echo "  CUDA: nvcc compiler and NVIDIA GPU"
	@echo "  HIP:  hipcc compiler and AMD/NVIDIA GPU"
	@echo ""
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