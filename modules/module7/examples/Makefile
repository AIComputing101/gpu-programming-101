# Module 7: Advanced Algorithmic Patterns
# Makefile for sophisticated parallel algorithm examples

# Compiler settings
NVCC = nvcc
HIPCC = hipcc

# Compiler flags
CUDA_FLAGS = -std=c++17 -O3 -arch=sm_75 -lineinfo
CUDA_DEBUG_FLAGS = -std=c++17 -g -G -arch=sm_75
HIP_FLAGS = -std=c++17 -O3
HIP_DEBUG_FLAGS = -std=c++17 -g

# Directories
BUILD_DIR = build
PROFILE_DIR = profiles

# Source files
CUDA_SOURCES = $(wildcard *_cuda.cu)
HIP_SOURCES = $(wildcard *_hip.cpp)

# Target executables
CUDA_TARGETS = $(CUDA_SOURCES:%.cu=$(BUILD_DIR)/%)
HIP_TARGETS = $(HIP_SOURCES:%.cpp=$(BUILD_DIR)/%)

# Default target
.PHONY: all
all: setup cuda hip

# Setup directories
.PHONY: setup
setup:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(PROFILE_DIR)

# Build CUDA examples
.PHONY: cuda
cuda: setup $(CUDA_TARGETS)

# Build HIP examples
.PHONY: hip
hip: setup $(HIP_TARGETS)

# Individual CUDA compilation
$(BUILD_DIR)/%_cuda: %_cuda.cu
	@echo "Building CUDA example: $@"
	$(NVCC) $(CUDA_FLAGS) $< -o $@

# Special rule for sparse matrix example that needs cuSPARSE
$(BUILD_DIR)/02_sparse_matrix_cuda: 02_sparse_matrix_cuda.cu
	@echo "Building CUDA example: $@"
	$(NVCC) $(CUDA_FLAGS) -lcusparse $< -o $@

# Individual HIP compilation
$(BUILD_DIR)/%_hip: %_hip.cpp
	@echo "Building HIP example: $@"
	$(HIPCC) $(HIP_FLAGS) $< -o $@

# Debug builds
.PHONY: debug
debug: CUDA_FLAGS = $(CUDA_DEBUG_FLAGS)
debug: HIP_FLAGS = $(HIP_DEBUG_FLAGS)
debug: all

# Algorithm-specific targets
.PHONY: sorting
sorting: setup
	@if [ -f 01_sorting_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 01_sorting_cuda.cu -o $(BUILD_DIR)/01_sorting_cuda; fi
	@if [ -f 01_sorting_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 01_sorting_hip.cpp -o $(BUILD_DIR)/01_sorting_hip; fi

.PHONY: sparse_matrix
sparse_matrix: setup
	@if [ -f 02_sparse_matrix_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) -lcusparse 02_sparse_matrix_cuda.cu -o $(BUILD_DIR)/02_sparse_matrix_cuda; fi
	@if [ -f 02_sparse_matrix_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 02_sparse_matrix_hip.cpp -o $(BUILD_DIR)/02_sparse_matrix_hip; fi

.PHONY: graph_algorithms
graph_algorithms: setup
	@if [ -f 03_graph_algorithms_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 03_graph_algorithms_cuda.cu -o $(BUILD_DIR)/03_graph_algorithms_cuda; fi
	@if [ -f 03_graph_algorithms_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 03_graph_algorithms_hip.cpp -o $(BUILD_DIR)/03_graph_algorithms_hip; fi

.PHONY: dynamic_programming
dynamic_programming: setup
	@if [ -f 04_dynamic_programming_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 04_dynamic_programming_cuda.cu -o $(BUILD_DIR)/04_dynamic_programming_cuda; fi
	@if [ -f 04_dynamic_programming_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 04_dynamic_programming_hip.cpp -o $(BUILD_DIR)/04_dynamic_programming_hip; fi

.PHONY: load_balancing
load_balancing: setup
	@if [ -f 05_load_balancing_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 05_load_balancing_cuda.cu -o $(BUILD_DIR)/05_load_balancing_cuda; fi
	@if [ -f 05_load_balancing_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 05_load_balancing_hip.cpp -o $(BUILD_DIR)/05_load_balancing_hip; fi

.PHONY: memory_compute
memory_compute: setup
	@if [ -f 06_memory_compute_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 06_memory_compute_cuda.cu -o $(BUILD_DIR)/06_memory_compute_cuda; fi
	@if [ -f 06_memory_compute_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 06_memory_compute_hip.cpp -o $(BUILD_DIR)/06_memory_compute_hip; fi

# Testing targets
.PHONY: test
test: all
	@echo "Running Module 7 Advanced Algorithm Tests..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Testing $$target..."; \
			$$target; \
			echo ""; \
		fi; \
	done

.PHONY: test_cuda
test_cuda: cuda
	@echo "Running CUDA Advanced Algorithm Tests..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Testing $$target..."; \
			$$target; \
			echo ""; \
		fi; \
	done

.PHONY: test_hip
test_hip: hip
	@echo "Running HIP Advanced Algorithm Tests..."
	@for target in $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Testing $$target..."; \
			$$target; \
			echo ""; \
		fi; \
	done

# Algorithm-specific tests
.PHONY: test_sorting
test_sorting: sorting
	@if [ -f $(BUILD_DIR)/01_sorting_cuda ]; then echo "CUDA Sorting:"; $(BUILD_DIR)/01_sorting_cuda; fi
	@if [ -f $(BUILD_DIR)/01_sorting_hip ]; then echo "HIP Sorting:"; $(BUILD_DIR)/01_sorting_hip; fi

.PHONY: test_sparse_matrix
test_sparse_matrix: sparse_matrix
	@if [ -f $(BUILD_DIR)/02_sparse_matrix_cuda ]; then echo "CUDA Sparse Matrix:"; $(BUILD_DIR)/02_sparse_matrix_cuda; fi
	@if [ -f $(BUILD_DIR)/02_sparse_matrix_hip ]; then echo "HIP Sparse Matrix:"; $(BUILD_DIR)/02_sparse_matrix_hip; fi

.PHONY: test_graph_algorithms
test_graph_algorithms: graph_algorithms
	@if [ -f $(BUILD_DIR)/03_graph_algorithms_cuda ]; then echo "CUDA Graph Algorithms:"; $(BUILD_DIR)/03_graph_algorithms_cuda; fi
	@if [ -f $(BUILD_DIR)/03_graph_algorithms_hip ]; then echo "HIP Graph Algorithms:"; $(BUILD_DIR)/03_graph_algorithms_hip; fi

# Profiling targets
.PHONY: profile_cuda
profile_cuda: cuda
	@echo "Profiling CUDA advanced algorithms with Nsight Compute..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Profiling $$target..."; \
			ncu --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active --log-file $(PROFILE_DIR)/$$(basename $$target).ncu-rep $$target; \
		fi; \
	done

.PHONY: profile_hip
profile_hip: hip
	@echo "Profiling HIP advanced algorithms with ROCProfiler..."
	@for target in $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Profiling $$target..."; \
			rocprof --stats --output-file $(PROFILE_DIR)/$$(basename $$target).csv $$target; \
		fi; \
	done

# Performance benchmarking
.PHONY: benchmark
benchmark: all
	@echo "Running Advanced Algorithm Performance Benchmarks..."
	@echo "Results will be saved to $(PROFILE_DIR)/benchmark_results.txt"
	@echo "Module 7: Advanced Algorithmic Patterns - Benchmark Results" > $(PROFILE_DIR)/benchmark_results.txt
	@echo "Generated on: $$(date)" >> $(PROFILE_DIR)/benchmark_results.txt
	@echo "=============================================================" >> $(PROFILE_DIR)/benchmark_results.txt
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo ""; \
			echo "Benchmarking $$target..." | tee -a $(PROFILE_DIR)/benchmark_results.txt; \
			echo "-------------------------------------------" >> $(PROFILE_DIR)/benchmark_results.txt; \
			$$target | tee -a $(PROFILE_DIR)/benchmark_results.txt; \
			echo "" >> $(PROFILE_DIR)/benchmark_results.txt; \
		fi; \
	done

# Algorithm complexity validation
.PHONY: validate_complexity
validate_complexity:
	@echo "Advanced Algorithm Complexity Analysis:"
	@echo "======================================"
	@echo "Sorting Algorithms:"
	@echo "  - Bitonic Sort: O(N log² N) work, O(log² N) depth"
	@echo "  - Radix Sort: O(d·N) work, O(d·log N) depth"
	@echo "  - Merge Sort: O(N log N) work, O(log² N) depth"
	@echo ""
	@echo "Sparse Matrix Operations:"
	@echo "  - SpMV (CSR): O(NNZ) work, O(1) depth"
	@echo "  - SpMV (ELL): O(N·max_nnz_per_row) work, O(1) depth"
	@echo ""
	@echo "Graph Algorithms:"
	@echo "  - BFS: O(V + E) work, O(D) depth (D = diameter)"
	@echo "  - SSSP: O(V·E) work, O(V) depth"
	@echo "  - Connected Components: O(V + E) work, O(log V) depth"
	@echo ""
	@echo "Dynamic Programming:"
	@echo "  - 1D DP: O(N·M) work, O(log N) depth"
	@echo "  - 2D DP: O(N²·M) work, O(N) depth"

# Scaling analysis
.PHONY: scaling_analysis
scaling_analysis: all
	@echo "Running scaling analysis for advanced algorithms..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Scaling analysis for $$target..."; \
			$$target --scaling-test 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_scaling.txt || true; \
		fi; \
	done

# Multi-GPU testing
.PHONY: test_multi_gpu
test_multi_gpu: all
	@echo "Testing multi-GPU capabilities..."
	@nvidia-smi --list-gpus || echo "NVIDIA GPUs not available"
	@rocm-smi --showallinfo | grep "GPU ID" || echo "AMD GPUs not available"
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Multi-GPU test for $$target..."; \
			$$target --multi-gpu-test 2>&1 || echo "Multi-GPU not supported for $$target"; \
		fi; \
	done

# Large-scale problem testing
.PHONY: test_large_scale
test_large_scale: all
	@echo "Running large-scale problem tests..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Large-scale test for $$target..."; \
			$$target --large-scale 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_large_scale.txt || true; \
		fi; \
	done

# Memory analysis
.PHONY: analyze_memory
analyze_memory: all
	@echo "Memory Usage Analysis for Advanced Algorithms:"
	@echo "=============================================="
	@echo "Running memory analysis with CUDA memcheck..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Analyzing $$target..."; \
			cuda-memcheck --tool=racecheck --log-file $(PROFILE_DIR)/$$(basename $$target).memcheck $$target 2>/dev/null || echo "Memory check failed for $$target"; \
		fi; \
	done

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(PROFILE_DIR)

# Cross-platform comparison
.PHONY: compare_platforms
compare_platforms: all
	@echo "Cross-Platform Performance Comparison:"
	@echo "====================================="
	@echo "Comparing CUDA vs HIP implementations..."
	@for cuda_target in $(CUDA_TARGETS); do \
		if [ -f $$cuda_target ]; then \
			base_name=$$(basename $$cuda_target _cuda); \
			hip_target=$(BUILD_DIR)/$${base_name}_hip; \
			if [ -f $$hip_target ]; then \
				echo "Comparing $$cuda_target vs $$hip_target"; \
				echo "CUDA:"; $$cuda_target --benchmark 2>&1 || true; \
				echo "HIP:"; $$hip_target --benchmark 2>&1 || true; \
				echo ""; \
			fi; \
		fi; \
	done

# System information
.PHONY: system_info
system_info:
	@echo "System Information for Module 7:"
	@echo "================================="
	@echo "CUDA Configuration:"
	@if command -v nvcc > /dev/null 2>&1; then \
		echo "  NVCC Version: $$(nvcc --version | grep release)"; \
		echo "  GPU Information:"; \
		nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | head -1; \
	else \
		echo "  NVCC not found"; \
	fi
	@echo ""
	@echo "HIP Configuration:"
	@if command -v hipcc > /dev/null 2>&1; then \
		echo "  HIPCC Version: $$(hipcc --version | head -1)"; \
		echo "  GPU Information:"; \
		rocm-smi --showproductname 2>/dev/null | head -3 || echo "  ROCm not available"; \
	else \
		echo "  HIPCC not found"; \
	fi

# Generate comprehensive performance report
.PHONY: report
report: benchmark profile_cuda profile_hip
	@echo "Generating comprehensive advanced algorithms performance report..."
	@echo "Module 7: Advanced Algorithmic Patterns - Performance Report" > $(PROFILE_DIR)/performance_report.md
	@echo "===========================================================" >> $(PROFILE_DIR)/performance_report.md
	@echo "" >> $(PROFILE_DIR)/performance_report.md
	@echo "## System Information" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Date: $$(date)" >> $(PROFILE_DIR)/performance_report.md
	@echo "- GPU: $$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'Unknown')" >> $(PROFILE_DIR)/performance_report.md
	@echo "- CUDA Version: $$(nvcc --version | grep release | awk '{print $$6}' 2>/dev/null || echo 'Not available')" >> $(PROFILE_DIR)/performance_report.md
	@echo "" >> $(PROFILE_DIR)/performance_report.md
	@echo "## Algorithm Categories" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Sorting: Bitonic, radix, merge sort implementations" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Sparse Matrix: SpMV operations with various storage formats" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Graph Algorithms: BFS, shortest path, connected components" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Dynamic Programming: 1D/2D DP problems with parallelization" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Load Balancing: Dynamic and static load balancing techniques" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Memory-Compute Optimization: Trade-off analysis and optimization" >> $(PROFILE_DIR)/performance_report.md
	@echo "" >> $(PROFILE_DIR)/performance_report.md
	@echo "## Benchmark Results" >> $(PROFILE_DIR)/performance_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/performance_report.md
	@cat $(PROFILE_DIR)/benchmark_results.txt >> $(PROFILE_DIR)/performance_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/performance_report.md
	@echo "Performance report generated: $(PROFILE_DIR)/performance_report.md"

# Help target
.PHONY: help
help:
	@echo "Module 7: Advanced Algorithmic Patterns - Build System"
	@echo "====================================================="
	@echo ""
	@echo "Build Targets:"
	@echo "  all              - Build all CUDA and HIP examples"
	@echo "  cuda             - Build CUDA examples only"
	@echo "  hip              - Build HIP examples only"
	@echo "  debug            - Build with debug flags"
	@echo "  clean            - Remove build artifacts"
	@echo ""
	@echo "Algorithm Targets:"
	@echo "  sorting          - Build sorting algorithm examples"
	@echo "  sparse_matrix    - Build sparse matrix operation examples"
	@echo "  graph_algorithms - Build graph algorithm examples"
	@echo "  dynamic_programming - Build dynamic programming examples"
	@echo "  load_balancing   - Build load balancing examples"
	@echo "  memory_compute   - Build memory-compute optimization examples"
	@echo ""
	@echo "Testing Targets:"
	@echo "  test             - Run all algorithm tests"
	@echo "  test_cuda        - Run CUDA algorithm tests only"
	@echo "  test_hip         - Run HIP algorithm tests only"
	@echo "  test_<algorithm> - Run specific algorithm tests"
	@echo "  test_large_scale - Run large-scale problem tests"
	@echo "  test_multi_gpu   - Test multi-GPU capabilities"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  benchmark        - Run performance benchmarks"
	@echo "  profile_cuda     - Profile CUDA algorithms with Nsight Compute"
	@echo "  profile_hip      - Profile HIP algorithms with ROCProfiler"
	@echo "  validate_complexity - Show algorithm complexity analysis"
	@echo "  analyze_memory   - Analyze memory access patterns"
	@echo "  scaling_analysis - Analyze algorithm scaling properties"
	@echo "  compare_platforms - Compare CUDA vs HIP performance"
	@echo "  report           - Generate comprehensive performance report"
	@echo ""
	@echo "Utility Targets:"
	@echo "  system_info      - Display system configuration"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make sorting && make test_sorting"
	@echo "  make cuda && make profile_cuda"
	@echo "  make all && make benchmark"
	@echo "  make report  # Complete performance analysis"