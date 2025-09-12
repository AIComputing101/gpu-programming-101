# Module 6: Fundamental Parallel Algorithms
# Makefile for comprehensive algorithm examples

# Compiler settings
NVCC = nvcc
HIPCC = hipcc

# Compiler flags
CUDA_FLAGS = -std=c++17 -O3 -arch=sm_70 -lineinfo
CUDA_DEBUG_FLAGS = -std=c++17 -g -G -arch=sm_70
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
.PHONY: convolution
convolution: setup
	@if [ -f 01_convolution_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 01_convolution_cuda.cu -o $(BUILD_DIR)/01_convolution_cuda; fi
	@if [ -f 01_convolution_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 01_convolution_hip.cpp -o $(BUILD_DIR)/01_convolution_hip; fi

.PHONY: stencil
stencil: setup
	@if [ -f 02_stencil_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 02_stencil_cuda.cu -o $(BUILD_DIR)/02_stencil_cuda; fi
	@if [ -f 02_stencil_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 02_stencil_hip.cpp -o $(BUILD_DIR)/02_stencil_hip; fi

.PHONY: histogram
histogram: setup
	@if [ -f 03_histogram_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 03_histogram_cuda.cu -o $(BUILD_DIR)/03_histogram_cuda; fi
	@if [ -f 03_histogram_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 03_histogram_hip.cpp -o $(BUILD_DIR)/03_histogram_hip; fi

.PHONY: reduction
reduction: setup
	@if [ -f 04_reduction_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 04_reduction_cuda.cu -o $(BUILD_DIR)/04_reduction_cuda; fi
	@if [ -f 04_reduction_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 04_reduction_hip.cpp -o $(BUILD_DIR)/04_reduction_hip; fi

.PHONY: prefix_sum
prefix_sum: setup
	@if [ -f 05_prefix_sum_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 05_prefix_sum_cuda.cu -o $(BUILD_DIR)/05_prefix_sum_cuda; fi
	@if [ -f 05_prefix_sum_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 05_prefix_sum_hip.cpp -o $(BUILD_DIR)/05_prefix_sum_hip; fi

# Testing targets
.PHONY: test
test: all
	@echo "Running Module 6 Algorithm Tests..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Testing $$target..."; \
			$$target; \
			echo ""; \
		fi; \
	done

.PHONY: test_cuda
test_cuda: cuda
	@echo "Running CUDA Algorithm Tests..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Testing $$target..."; \
			$$target; \
			echo ""; \
		fi; \
	done

.PHONY: test_hip
test_hip: hip
	@echo "Running HIP Algorithm Tests..."
	@for target in $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Testing $$target..."; \
			$$target; \
			echo ""; \
		fi; \
	done

# Algorithm-specific tests
.PHONY: test_convolution
test_convolution: convolution
	@if [ -f $(BUILD_DIR)/01_convolution_cuda ]; then echo "CUDA Convolution:"; $(BUILD_DIR)/01_convolution_cuda; fi
	@if [ -f $(BUILD_DIR)/01_convolution_hip ]; then echo "HIP Convolution:"; $(BUILD_DIR)/01_convolution_hip; fi

.PHONY: test_stencil
test_stencil: stencil
	@if [ -f $(BUILD_DIR)/02_stencil_cuda ]; then echo "CUDA Stencil:"; $(BUILD_DIR)/02_stencil_cuda; fi
	@if [ -f $(BUILD_DIR)/02_stencil_hip ]; then echo "HIP Stencil:"; $(BUILD_DIR)/02_stencil_hip; fi

.PHONY: test_histogram
test_histogram: histogram
	@if [ -f $(BUILD_DIR)/03_histogram_cuda ]; then echo "CUDA Histogram:"; $(BUILD_DIR)/03_histogram_cuda; fi
	@if [ -f $(BUILD_DIR)/03_histogram_hip ]; then echo "HIP Histogram:"; $(BUILD_DIR)/03_histogram_hip; fi

# Profiling targets
.PHONY: profile_cuda
profile_cuda: cuda
	@echo "Profiling CUDA algorithms with Nsight Compute..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Profiling $$target..."; \
			ncu --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed --log-file $(PROFILE_DIR)/$$(basename $$target).ncu-rep $$target; \
		fi; \
	done

.PHONY: profile_hip
profile_hip: hip
	@echo "Profiling HIP algorithms with ROCProfiler..."
	@for target in $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Profiling $$target..."; \
			rocprof --stats --output-file $(PROFILE_DIR)/$$(basename $$target).csv $$target; \
		fi; \
	done

# Performance benchmarking
.PHONY: benchmark
benchmark: all
	@echo "Running Algorithm Performance Benchmarks..."
	@echo "Results will be saved to $(PROFILE_DIR)/benchmark_results.txt"
	@echo "Module 6: Fundamental Parallel Algorithms - Benchmark Results" > $(PROFILE_DIR)/benchmark_results.txt
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

# Algorithm complexity analysis
.PHONY: analyze_complexity
analyze_complexity:
	@echo "Algorithm Complexity Analysis:"
	@echo "============================="
	@echo "Convolution:"
	@echo "  - 1D: O(N*M) work, O(1) depth"
	@echo "  - 2D: O(N²*M²) work, O(1) depth"
	@echo "  - Separable 2D: O(N²*M) work, O(1) depth"
	@echo ""
	@echo "Stencil:"
	@echo "  - 1D 3-point: O(N) work, O(1) depth"
	@echo "  - 2D 5-point: O(N²) work, O(1) depth"
	@echo "  - 3D 7-point: O(N³) work, O(1) depth"
	@echo ""
	@echo "Histogram:"
	@echo "  - Atomic: O(N) work, O(1) depth"
	@echo "  - Privatized: O(N + P*B) work, O(log P) depth"
	@echo ""
	@echo "Reduction:"
	@echo "  - Tree: O(N) work, O(log N) depth"
	@echo "  - Warp-optimized: O(N) work, O(log N) depth"
	@echo ""
	@echo "Prefix Sum:"
	@echo "  - Hillis-Steele: O(N log N) work, O(log N) depth"
	@echo "  - Blelloch: O(N) work, O(log N) depth"

# Memory analysis
.PHONY: analyze_memory
analyze_memory: all
	@echo "Memory Access Pattern Analysis:"
	@echo "==============================="
	@echo "Running memory analysis with CUDA memcheck..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Analyzing $$target..."; \
			cuda-memcheck --tool=racecheck --log-file $(PROFILE_DIR)/$$(basename $$target).memcheck $$target; \
		fi; \
	done

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(PROFILE_DIR)

# System information
.PHONY: system_info
system_info:
	@echo "System Information for Module 6:"
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

# Generate performance report
.PHONY: report
report: benchmark profile_cuda profile_hip
	@echo "Generating comprehensive algorithm performance report..."
	@echo "Module 6: Fundamental Parallel Algorithms - Performance Report" > $(PROFILE_DIR)/performance_report.md
	@echo "=============================================================" >> $(PROFILE_DIR)/performance_report.md
	@echo "" >> $(PROFILE_DIR)/performance_report.md
	@echo "## System Information" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Date: $$(date)" >> $(PROFILE_DIR)/performance_report.md
	@echo "- GPU: $$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'Unknown')" >> $(PROFILE_DIR)/performance_report.md
	@echo "- CUDA Version: $$(nvcc --version | grep release | awk '{print $$6}' 2>/dev/null || echo 'Not available')" >> $(PROFILE_DIR)/performance_report.md
	@echo "" >> $(PROFILE_DIR)/performance_report.md
	@echo "## Algorithm Categories" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Convolution: 1D/2D/3D filtering operations" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Stencil: Finite difference computations" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Histogram: Frequency counting with atomic operations" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Reduction: Parallel aggregation patterns" >> $(PROFILE_DIR)/performance_report.md
	@echo "- Prefix Sum: Scan operations for cumulative computations" >> $(PROFILE_DIR)/performance_report.md
	@echo "" >> $(PROFILE_DIR)/performance_report.md
	@echo "## Benchmark Results" >> $(PROFILE_DIR)/performance_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/performance_report.md
	@cat $(PROFILE_DIR)/benchmark_results.txt >> $(PROFILE_DIR)/performance_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/performance_report.md
	@echo "Performance report generated: $(PROFILE_DIR)/performance_report.md"

# Help target
.PHONY: help
help:
	@echo "Module 6: Fundamental Parallel Algorithms - Build System"
	@echo "========================================================"
	@echo ""
	@echo "Build Targets:"
	@echo "  all              - Build all CUDA and HIP examples"
	@echo "  cuda             - Build CUDA examples only"
	@echo "  hip              - Build HIP examples only"
	@echo "  debug            - Build with debug flags"
	@echo "  clean            - Remove build artifacts"
	@echo ""
	@echo "Algorithm Targets:"
	@echo "  convolution      - Build convolution examples"
	@echo "  stencil          - Build stencil computation examples"
	@echo "  histogram        - Build histogram operation examples"
	@echo "  reduction        - Build reduction pattern examples"
	@echo "  prefix_sum       - Build prefix sum algorithm examples"
	@echo ""
	@echo "Testing Targets:"
	@echo "  test             - Run all algorithm tests"
	@echo "  test_cuda        - Run CUDA algorithm tests only"
	@echo "  test_hip         - Run HIP algorithm tests only"
	@echo "  test_<algorithm> - Run specific algorithm tests"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  benchmark        - Run performance benchmarks"
	@echo "  profile_cuda     - Profile CUDA algorithms with Nsight Compute"
	@echo "  profile_hip      - Profile HIP algorithms with ROCProfiler"
	@echo "  analyze_complexity - Show algorithm complexity analysis"
	@echo "  analyze_memory   - Analyze memory access patterns"
	@echo "  report           - Generate comprehensive performance report"
	@echo ""
	@echo "Utility Targets:"
	@echo "  system_info      - Display system configuration"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make convolution && make test_convolution"
	@echo "  make cuda && make profile_cuda"
	@echo "  make all && make benchmark"
	@echo "  make report  # Complete performance analysis"