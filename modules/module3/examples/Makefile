# GPU Programming Module 3 Examples Makefile
# Advanced GPU Algorithms and Parallel Patterns

# Compiler settings
NVCC = nvcc
HIPCC = hipcc
NVCC_FLAGS = -O2 -std=c++11 -arch=sm_50 -lcuda
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
01_reduction_algorithms_cuda: 01_reduction_algorithms_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

01_reduction_algorithms_hip: 01_reduction_algorithms_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

02_scan_prefix_sum_cuda: 02_scan_prefix_sum_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

02_scan_prefix_sum_hip: 02_scan_prefix_sum_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

03_sorting_algorithms_cuda: 03_sorting_algorithms_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

03_sorting_algorithms_hip: 03_sorting_algorithms_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

04_convolution_stencil_cuda: 04_convolution_stencil_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

04_convolution_stencil_hip: 04_convolution_stencil_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

05_matrix_operations_cuda: 05_matrix_operations_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

06_graph_algorithms_cuda: 06_graph_algorithms_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

07_cooperative_groups_cuda: 07_cooperative_groups_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Legacy targets for backward compatibility
reduction_cuda: 01_reduction_algorithms_cuda
reduction_hip: 01_reduction_algorithms_hip
scan_cuda: 02_scan_prefix_sum_cuda
scan_hip: 02_scan_prefix_sum_hip
sorting_cuda: 03_sorting_algorithms_cuda
sorting_hip: 03_sorting_algorithms_hip
convolution_cuda: 04_convolution_stencil_cuda
convolution_hip: 04_convolution_stencil_hip
matrix_cuda: 05_matrix_operations_cuda
graph_cuda: 06_graph_algorithms_cuda
cooperative_cuda: 07_cooperative_groups_cuda

# Generic legacy targets (default to CUDA)
reduction: 01_reduction_algorithms_cuda
scan: 02_scan_prefix_sum_cuda
sorting: 03_sorting_algorithms_cuda
convolution: 04_convolution_stencil_cuda
matrix: 05_matrix_operations_cuda
graph: 06_graph_algorithms_cuda
cooperative: 07_cooperative_groups_cuda

# Test targets
test: test_cuda

test_cuda: cuda
	@echo "Running Module 3 CUDA tests..."
	@if command -v nvidia-smi > /dev/null; then \
		echo "=== Testing Advanced Algorithm Examples ==="; \
		echo "1. Reduction Algorithms..."; \
		./01_reduction_algorithms_cuda || echo "✗ Reduction algorithms failed"; \
		echo "2. Scan (Prefix Sum)..."; \
		./02_scan_prefix_sum_cuda || echo "✗ Scan algorithms failed"; \
		echo "3. Sorting Algorithms..."; \
		./03_sorting_algorithms_cuda || echo "✗ Sorting algorithms failed"; \
		echo "4. Convolution/Stencil..."; \
		./04_convolution_stencil_cuda || echo "✗ Convolution failed"; \
		echo "5. Matrix Operations..."; \
		./05_matrix_operations_cuda || echo "✗ Matrix operations failed"; \
		echo "6. Graph Algorithms..."; \
		./06_graph_algorithms_cuda || echo "✗ Graph algorithms failed"; \
		echo "7. Cooperative Groups..."; \
		./07_cooperative_groups_cuda || echo "✗ Cooperative groups failed"; \
		echo "✓ Module 3 CUDA tests completed"; \
	else \
		echo "No NVIDIA GPU detected, skipping CUDA tests"; \
	fi

test_hip: hip
	@echo "Running Module 3 HIP tests..."
	@if command -v rocm-smi > /dev/null || command -v nvidia-smi > /dev/null; then \
		echo "=== Testing HIP Algorithm Examples ==="; \
		echo "1. Reduction Algorithms..."; \
		./01_reduction_algorithms_hip || echo "✗ HIP reduction algorithms failed"; \
		echo "2. Scan (Prefix Sum)..."; \
		./02_scan_prefix_sum_hip || echo "✗ HIP scan algorithms failed"; \
		echo "3. Sorting Algorithms..."; \
		./03_sorting_algorithms_hip || echo "✗ HIP sorting algorithms failed"; \
		echo "4. Convolution/Stencil..."; \
		./04_convolution_stencil_hip || echo "✗ HIP convolution failed"; \
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

# Clean targets
clean:
	rm -f $(CUDA_EXECUTABLES) $(HIP_EXECUTABLES)
	rm -f 01_reduction_algorithms_cuda 01_reduction_algorithms_hip
	rm -f 02_scan_prefix_sum_cuda 02_scan_prefix_sum_hip
	rm -f 03_sorting_algorithms_cuda 03_sorting_algorithms_hip
	rm -f 04_convolution_stencil_cuda 04_convolution_stencil_hip
	rm -f 05_matrix_operations_cuda
	rm -f 06_graph_algorithms_cuda
	rm -f 07_cooperative_groups_cuda
	rm -f reduction_cuda reduction_hip scan_cuda scan_hip sorting_cuda sorting_hip
	rm -f convolution_cuda convolution_hip matrix_cuda graph_cuda cooperative_cuda
	rm -f reduction scan sorting convolution matrix graph cooperative

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

# Help target
help:
	@echo "GPU Programming 101 - Module 3 Examples Makefile"
	@echo "================================================"
	@echo ""
	@echo "Module 3: Advanced GPU Algorithms and Parallel Patterns"
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
	@echo "  01_reduction_algorithms_cuda       - Parallel reduction patterns"
	@echo "  02_scan_prefix_sum_cuda            - Scan/prefix sum algorithms"
	@echo "  03_sorting_algorithms_cuda         - GPU sorting implementations"
	@echo "  04_convolution_stencil_cuda        - Stencil computation patterns"
	@echo "  05_matrix_operations_cuda          - Matrix multiplication and operations"
	@echo "  06_graph_algorithms_cuda           - Graph processing algorithms"
	@echo "  07_cooperative_groups_cuda         - Modern cooperative groups API"
	@echo ""
	@echo "Individual Examples (HIP):"
	@echo "  01_reduction_algorithms_hip        - Cross-platform reduction"
	@echo "  02_scan_prefix_sum_hip             - Cross-platform scan algorithms"
	@echo "  03_sorting_algorithms_hip          - Cross-platform sorting"
	@echo "  04_convolution_stencil_hip         - Cross-platform convolution"
	@echo ""
	@echo "Legacy Targets:"
	@echo "  reduction, scan, sorting, convolution, matrix, graph, cooperative"
	@echo "  (These default to CUDA versions)"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  analyze_complexity - Algorithm complexity analysis"
	@echo "  profile_algorithms - Show profiling commands"
	@echo ""
	@echo "Utility Targets:"
	@echo "  list              - List all available examples"
	@echo "  clean             - Remove all executables"
	@echo "  help              - Show this help"
	@echo ""
	@echo "Example Usage:"
	@echo "  make cuda                              # Build all CUDA examples"
	@echo "  make hip                               # Build all HIP examples"
	@echo "  make 01_reduction_algorithms_cuda      # Build specific example"
	@echo "  make test_performance                  # Run performance tests"
	@echo "  make profile_algorithms                # Show profiling commands"
	@echo ""
	@echo "Requirements:"
	@echo "  CUDA: nvcc compiler and NVIDIA GPU"
	@echo "  HIP:  hipcc compiler and AMD/NVIDIA GPU"
	@echo ""
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