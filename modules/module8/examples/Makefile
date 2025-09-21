# Module 8: Domain-Specific Applications
# Makefile for real-world GPU application examples

# Compiler settings
NVCC = nvcc
HIPCC = hipcc

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

# Compiler flags for production-quality applications
CUDA_FLAGS = -std=c++17 -O3 -arch=sm_70 -lineinfo --use_fast_math
CUDA_DEBUG_FLAGS = -std=c++17 -g -G -arch=sm_70
HIP_FLAGS = -std=c++17 -O3 -ffast-math
HIP_DEBUG_FLAGS = -std=c++17 -g

# Library flags for domain-specific libraries
CUDA_LIBS = -lcublas -lcurand -lcufft
HIP_LIBS = -lrocblas -lrocrand -lrocfft -lMIOpen

# Directories
BUILD_DIR = build
PROFILE_DIR = profiles
DATA_DIR = data

# Source files
CUDA_SOURCES = $(wildcard *_cuda.cu)
HIP_SOURCES = $(wildcard *_hip.cpp)

# Target executables based on GPU vendor
ifeq ($(BUILD_CUDA),1)
ACTIVE_TARGETS = $(CUDA_SOURCES:%.cu=$(BUILD_DIR)/%)
ACTIVE_SOURCES = $(CUDA_SOURCES)
COMPILER = $(NVCC)
COMPILE_FLAGS = $(CUDA_FLAGS)
else ifeq ($(BUILD_HIP),1)
ACTIVE_TARGETS = $(HIP_SOURCES:%.cpp=$(BUILD_DIR)/%)
ACTIVE_SOURCES = $(HIP_SOURCES)
COMPILER = $(HIPCC)
COMPILE_FLAGS = $(HIP_FLAGS)
else
ACTIVE_TARGETS = 
ACTIVE_SOURCES = 
endif

# Legacy target definitions (for compatibility)
CUDA_TARGETS = $(CUDA_SOURCES:%.cu=$(BUILD_DIR)/%)
HIP_TARGETS = $(HIP_SOURCES:%.cpp=$(BUILD_DIR)/%)

# Default target
.PHONY: all
all: setup $(ACTIVE_TARGETS)

# Setup directories
.PHONY: setup
setup:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(PROFILE_DIR)
	@mkdir -p $(DATA_DIR)
ifeq ($(GPU_VENDOR),NVIDIA)
	@echo "✓ NVIDIA GPU detected - building CUDA examples"
else ifeq ($(GPU_VENDOR),AMD)
	@echo "✓ AMD GPU detected - building HIP examples"
else
	@echo "⚠ No compatible GPU detected - no examples will be built"
endif

# Build CUDA examples (if NVIDIA GPU detected)
.PHONY: cuda
ifeq ($(BUILD_CUDA),1)
cuda: setup $(CUDA_TARGETS)
else
cuda: setup
	@echo "⚠ CUDA build requested but no NVIDIA GPU detected"
endif

# Build HIP examples (if AMD GPU detected)  
.PHONY: hip
ifeq ($(BUILD_HIP),1)
hip: setup $(HIP_TARGETS)
else
hip: setup
	@echo "⚠ HIP build requested but no AMD GPU detected"
endif

# Vendor-specific compilation rules
ifeq ($(BUILD_CUDA),1)
$(BUILD_DIR)/%_cuda: %_cuda.cu
	@echo "Building CUDA domain application: $@"
	$(NVCC) $(CUDA_FLAGS) $< -o $@ $(CUDA_LIBS)
endif

ifeq ($(BUILD_HIP),1)
$(BUILD_DIR)/%_hip: %_hip.cpp
	@echo "Building HIP domain application: $@"
	$(HIPCC) $(HIP_FLAGS) $< -o $@ $(HIP_LIBS)
endif

# Legacy compilation rules (for compatibility with explicit targets)
$(BUILD_DIR)/%_cuda: %_cuda.cu
	@echo "Building CUDA domain application: $@"
	$(NVCC) $(CUDA_FLAGS) $< -o $@ $(CUDA_LIBS)

$(BUILD_DIR)/%_hip: %_hip.cpp
	@echo "Building HIP domain application: $@"
	$(HIPCC) $(HIP_FLAGS) $< -o $@ $(HIP_LIBS)

# Debug builds
.PHONY: debug
debug: CUDA_FLAGS = $(CUDA_DEBUG_FLAGS)
debug: HIP_FLAGS = $(HIP_DEBUG_FLAGS)
debug: all

# Production builds with maximum optimization
.PHONY: production
production: CUDA_FLAGS += -DNDEBUG -Xptxas -O3
production: HIP_FLAGS += -DNDEBUG
production: all

# Domain-specific application targets
.PHONY: deep_learning
deep_learning: setup
ifeq ($(BUILD_CUDA),1)
	@if [ -f 01_deep_learning_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 01_deep_learning_cuda.cu -o $(BUILD_DIR)/01_deep_learning_cuda $(CUDA_LIBS); fi
endif
ifeq ($(BUILD_HIP),1)
	@if [ -f 01_deep_learning_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 01_deep_learning_hip.cpp -o $(BUILD_DIR)/01_deep_learning_hip $(HIP_LIBS); fi
endif

.PHONY: scientific
scientific: setup
ifeq ($(BUILD_CUDA),1)
	@if [ -f 02_scientific_computing_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 02_scientific_computing_cuda.cu -o $(BUILD_DIR)/02_scientific_computing_cuda $(CUDA_LIBS); fi
endif
ifeq ($(BUILD_HIP),1)
	@if [ -f 02_scientific_computing_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 02_scientific_computing_hip.cpp -o $(BUILD_DIR)/02_scientific_computing_hip $(HIP_LIBS); fi
endif

.PHONY: image_processing
image_processing: setup
ifeq ($(BUILD_CUDA),1)
	@if [ -f 03_image_signal_processing_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 03_image_signal_processing_cuda.cu -o $(BUILD_DIR)/03_image_signal_processing_cuda $(CUDA_LIBS); fi
endif
ifeq ($(BUILD_HIP),1)
	@if [ -f 03_image_signal_processing_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 03_image_signal_processing_hip.cpp -o $(BUILD_DIR)/03_image_signal_processing_hip $(HIP_LIBS); fi
endif

.PHONY: monte_carlo
monte_carlo: setup
ifeq ($(BUILD_CUDA),1)
	@if [ -f 04_monte_carlo_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 04_monte_carlo_cuda.cu -o $(BUILD_DIR)/04_monte_carlo_cuda $(CUDA_LIBS); fi
endif
ifeq ($(BUILD_HIP),1)
	@if [ -f 04_monte_carlo_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 04_monte_carlo_hip.cpp -o $(BUILD_DIR)/04_monte_carlo_hip $(HIP_LIBS); fi
endif

.PHONY: finance
finance: setup
ifeq ($(BUILD_CUDA),1)
	@if [ -f 05_computational_finance_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 05_computational_finance_cuda.cu -o $(BUILD_DIR)/05_computational_finance_cuda $(CUDA_LIBS); fi
endif
ifeq ($(BUILD_HIP),1)
	@if [ -f 05_computational_finance_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 05_computational_finance_hip.cpp -o $(BUILD_DIR)/05_computational_finance_hip $(HIP_LIBS); fi
endif

.PHONY: library_integration
library_integration: setup
ifeq ($(BUILD_CUDA),1)
	@if [ -f 06_library_integration_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 06_library_integration_cuda.cu -o $(BUILD_DIR)/06_library_integration_cuda $(CUDA_LIBS); fi
endif
ifeq ($(BUILD_HIP),1)
	@if [ -f 06_library_integration_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 06_library_integration_hip.cpp -o $(BUILD_DIR)/06_library_integration_hip $(HIP_LIBS); fi
endif
	@if [ -f 06_library_integration_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 06_library_integration_hip.cpp -o $(BUILD_DIR)/06_library_integration_hip $(HIP_LIBS); fi

# Testing targets with domain-specific validation
.PHONY: test
test: all
	@echo "Running Module 8 Domain-Specific Application Tests..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Testing $$target..."; \
			$$target --test-mode; \
			echo ""; \
		fi; \
	done

.PHONY: test_cuda
test_cuda: cuda
	@echo "Running CUDA Domain Application Tests..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Testing $$target..."; \
			$$target --test-mode; \
			echo ""; \
		fi; \
	done

.PHONY: test_hip
test_hip: hip
	@echo "Running HIP Domain Application Tests..."
	@for target in $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Testing $$target..."; \
			$$target --test-mode; \
			echo ""; \
		fi; \
	done

# Domain-specific application tests
.PHONY: test_deep_learning
test_deep_learning: deep_learning
	@if [ -f $(BUILD_DIR)/01_deep_learning_cuda ]; then echo "CUDA Deep Learning:"; $(BUILD_DIR)/01_deep_learning_cuda --validation; fi
	@if [ -f $(BUILD_DIR)/01_deep_learning_hip ]; then echo "HIP Deep Learning:"; $(BUILD_DIR)/01_deep_learning_hip --validation; fi

.PHONY: test_scientific
test_scientific: scientific
	@if [ -f $(BUILD_DIR)/02_scientific_computing_cuda ]; then echo "CUDA Scientific Computing:"; $(BUILD_DIR)/02_scientific_computing_cuda --accuracy-test; fi
	@if [ -f $(BUILD_DIR)/02_scientific_computing_hip ]; then echo "HIP Scientific Computing:"; $(BUILD_DIR)/02_scientific_computing_hip --accuracy-test; fi

.PHONY: test_image_processing
test_image_processing: image_processing
	@if [ -f $(BUILD_DIR)/03_image_signal_processing_cuda ]; then echo "CUDA Image Processing:"; $(BUILD_DIR)/03_image_signal_processing_cuda --benchmark; fi
	@if [ -f $(BUILD_DIR)/03_image_signal_processing_hip ]; then echo "HIP Image Processing:"; $(BUILD_DIR)/03_image_signal_processing_hip --benchmark; fi

.PHONY: test_monte_carlo
test_monte_carlo: monte_carlo
	@if [ -f $(BUILD_DIR)/04_monte_carlo_cuda ]; then echo "CUDA Monte Carlo:"; $(BUILD_DIR)/04_monte_carlo_cuda --convergence-test; fi
	@if [ -f $(BUILD_DIR)/04_monte_carlo_hip ]; then echo "HIP Monte Carlo:"; $(BUILD_DIR)/04_monte_carlo_hip --convergence-test; fi

.PHONY: test_finance
test_finance: finance
	@if [ -f $(BUILD_DIR)/05_computational_finance_cuda ]; then echo "CUDA Computational Finance:"; $(BUILD_DIR)/05_computational_finance_cuda --market-simulation; fi
	@if [ -f $(BUILD_DIR)/05_computational_finance_hip ]; then echo "HIP Computational Finance:"; $(BUILD_DIR)/05_computational_finance_hip --market-simulation; fi

# Production validation tests
.PHONY: validate_performance
validate_performance: production
	@echo "Validating production performance targets..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Performance validation for $$target..."; \
			$$target --performance-validation 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_performance.txt; \
		fi; \
	done

.PHONY: validate_accuracy
validate_accuracy: all
	@echo "Validating numerical accuracy for domain applications..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Accuracy validation for $$target..."; \
			$$target --accuracy-validation 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_accuracy.txt; \
		fi; \
	done

# Profiling with domain-specific metrics
.PHONY: profile_cuda
profile_cuda: cuda
	@echo "Profiling CUDA domain applications with Nsight Compute..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Profiling $$target..."; \
			ncu --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__sass_thread_inst_executed_op_fp32_pred_on.sum,tensor__inst_executed.sum --log-file $(PROFILE_DIR)/$$(basename $$target).ncu-rep $$target --profile-mode; \
		fi; \
	done

.PHONY: profile_hip
profile_hip: hip
	@echo "Profiling HIP domain applications with ROCProfiler..."
	@for target in $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Profiling $$target..."; \
			rocprof --stats --hsa-trace --output-file $(PROFILE_DIR)/$$(basename $$target).csv $$target --profile-mode; \
		fi; \
	done

# Comprehensive domain-specific benchmarking
.PHONY: benchmark_all
benchmark_all: production
	@echo "Running comprehensive domain application benchmarks..."
	@echo "Domain-Specific Applications Benchmark Results" > $(PROFILE_DIR)/domain_benchmark_results.txt
	@echo "Generated on: $$(date)" >> $(PROFILE_DIR)/domain_benchmark_results.txt
	@echo "=============================================" >> $(PROFILE_DIR)/domain_benchmark_results.txt
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo ""; \
			echo "Benchmarking $$target..." | tee -a $(PROFILE_DIR)/domain_benchmark_results.txt; \
			echo "-------------------------------------------" >> $(PROFILE_DIR)/domain_benchmark_results.txt; \
			$$target --comprehensive-benchmark | tee -a $(PROFILE_DIR)/domain_benchmark_results.txt; \
			echo "" >> $(PROFILE_DIR)/domain_benchmark_results.txt; \
		fi; \
	done

# Domain-specific benchmarking by category
.PHONY: benchmark_deep_learning
benchmark_deep_learning: deep_learning
	@echo "Deep Learning Inference Benchmarks:"
	@for target in $(BUILD_DIR)/01_deep_learning_*; do \
		if [ -f $$target ]; then \
			echo "$$target throughput test:"; \
			$$target --throughput-benchmark; \
		fi; \
	done

.PHONY: benchmark_scientific
benchmark_scientific: scientific
	@echo "Scientific Computing Benchmarks:"
	@for target in $(BUILD_DIR)/02_scientific_computing_*; do \
		if [ -f $$target ]; then \
			echo "$$target solver performance:"; \
			$$target --solver-benchmark; \
		fi; \
	done

.PHONY: benchmark_image_processing
benchmark_image_processing: image_processing
	@echo "Image Processing Pipeline Benchmarks:"
	@for target in $(BUILD_DIR)/03_image_signal_processing_*; do \
		if [ -f $$target ]; then \
			echo "$$target real-time performance:"; \
			$$target --realtime-benchmark; \
		fi; \
	done

.PHONY: benchmark_monte_carlo
benchmark_monte_carlo: monte_carlo
	@echo "Monte Carlo Simulation Benchmarks:"
	@for target in $(BUILD_DIR)/04_monte_carlo_*; do \
		if [ -f $$target ]; then \
			echo "$$target convergence rate:"; \
			$$target --convergence-benchmark; \
		fi; \
	done

.PHONY: benchmark_finance
benchmark_finance: finance
	@echo "Computational Finance Benchmarks:"
	@for target in $(BUILD_DIR)/05_computational_finance_*; do \
		if [ -f $$target ]; then \
			echo "$$target latency test:"; \
			$$target --latency-benchmark; \
		fi; \
	done

# Large-scale problem testing
.PHONY: test_large_scale
test_large_scale: production
	@echo "Running large-scale domain application tests..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Large-scale test for $$target..."; \
			$$target --large-scale-test 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_large_scale.txt || true; \
		fi; \
	done

# Multi-GPU scaling tests
.PHONY: test_multi_gpu_scaling
test_multi_gpu_scaling: production
	@echo "Testing multi-GPU scaling for domain applications..."
	@nvidia-smi --list-gpus | wc -l > $(PROFILE_DIR)/gpu_count.txt || echo "0" > $(PROFILE_DIR)/gpu_count.txt
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Multi-GPU scaling test for $$target..."; \
			$$target --multi-gpu-scaling 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_multi_gpu.txt || true; \
		fi; \
	done

# Cross-platform comparison
.PHONY: compare_platforms
compare_platforms: all
	@echo "Cross-Platform Performance Comparison for Domain Applications:"
	@echo "============================================================"
	@for cuda_target in $(CUDA_TARGETS); do \
		if [ -f $$cuda_target ]; then \
			base_name=$$(basename $$cuda_target _cuda); \
			hip_target=$(BUILD_DIR)/$${base_name}_hip; \
			if [ -f $$hip_target ]; then \
				echo "Comparing $$cuda_target vs $$hip_target"; \
				echo "CUDA Performance:"; \
				$$cuda_target --platform-benchmark 2>&1 || true; \
				echo "HIP Performance:"; \
				$$hip_target --platform-benchmark 2>&1 || true; \
				echo ""; \
			fi; \
		fi; \
	done

# Domain-specific performance analysis
.PHONY: analyze_performance_targets
analyze_performance_targets:
	@echo "Domain-Specific Performance Target Analysis:"
	@echo "==========================================="
	@echo "Deep Learning:"
	@echo "  Target: >90% Tensor Core utilization, >1000x CPU speedup"
	@echo "  Metric: Samples/second, inference latency"
	@echo ""
	@echo "Scientific Computing:"
	@echo "  Target: 10-100x CPU speedup, numerical accuracy preservation"
	@echo "  Metric: Time-to-solution, relative error"
	@echo ""
	@echo "Image Processing:"
	@echo "  Target: 30-60 FPS real-time processing"
	@echo "  Metric: Frame rate, pipeline latency"
	@echo ""
	@echo "Monte Carlo:"
	@echo "  Target: 100-1000x CPU speedup, fast convergence"
	@echo "  Metric: Samples/second, convergence rate"
	@echo ""
	@echo "Computational Finance:"
	@echo "  Target: Sub-millisecond latency, high throughput"
	@echo "  Metric: Transactions/second, latency percentiles"

# Memory and compute analysis
.PHONY: analyze_memory_compute
analyze_memory_compute: all
	@echo "Memory-Compute Analysis for Domain Applications:"
	@echo "=============================================="
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Analyzing $$target..."; \
			cuda-memcheck --tool=memcheck --log-file $(PROFILE_DIR)/$$(basename $$target).memcheck $$target --memory-analysis 2>/dev/null || echo "Memory analysis failed for $$target"; \
		fi; \
	done

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(PROFILE_DIR)
	rm -rf $(DATA_DIR)

# System requirements check
.PHONY: check_requirements
check_requirements:
	@echo "Checking system requirements for domain applications:"
	@echo "=================================================="
	@echo "GPU Memory Requirements:"
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "NVIDIA GPU not detected"
	@rocm-smi --showmeminfo vram 2>/dev/null || echo "AMD GPU not detected"
	@echo ""
	@echo "Library Requirements:"
	@echo "CUDA Libraries:"
	@ls /usr/local/cuda/lib64/libcublas* 2>/dev/null | head -1 || echo "  cuBLAS not found"
	@ls /usr/local/cuda/lib64/libcurand* 2>/dev/null | head -1 || echo "  cuRAND not found"
	@ls /usr/local/cuda/lib64/libcufft* 2>/dev/null | head -1 || echo "  cuFFT not found"
	@ls /usr/local/cuda/lib64/libcudnn* 2>/dev/null | head -1 || echo "  cuDNN not found"
	@echo ""
	@echo "ROCm Libraries:"
	@ls /opt/rocm/lib/librocblas* 2>/dev/null | head -1 || echo "  rocBLAS not found"
	@ls /opt/rocm/lib/librocrand* 2>/dev/null | head -1 || echo "  rocRAND not found"
	@ls /opt/rocm/lib/librocfft* 2>/dev/null | head -1 || echo "  rocFFT not found"

# Generate comprehensive domain applications report
.PHONY: report
report: benchmark_all profile_cuda profile_hip
	@echo "Generating comprehensive domain applications performance report..."
	@echo "Module 8: Domain-Specific Applications - Performance Report" > $(PROFILE_DIR)/domain_performance_report.md
	@echo "=========================================================" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "## System Information" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "- Date: $$(date)" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "- GPU: $$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'Unknown')" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "- GPU Memory: $$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'Unknown') MB" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "- CUDA Version: $$(nvcc --version | grep release | awk '{print $$6}' 2>/dev/null || echo 'Not available')" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "## Domain Applications" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "- Deep Learning: Neural network inference with Tensor Core optimization" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "- Scientific Computing: High-precision numerical simulations" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "- Image Processing: Real-time computer vision pipelines" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "- Monte Carlo: Stochastic simulation and risk analysis" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "- Computational Finance: High-frequency trading and risk management" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "- Library Integration: Optimized library usage patterns" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "## Benchmark Results" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/domain_performance_report.md
	@cat $(PROFILE_DIR)/domain_benchmark_results.txt >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/domain_performance_report.md
	@echo "Domain applications performance report generated: $(PROFILE_DIR)/domain_performance_report.md"

# System information
.PHONY: system_info
system_info:
	@echo "System Information for Module 8:"
	@echo "================================="
	@echo "CUDA Configuration:"
	@if command -v nvcc > /dev/null 2>&1; then \
		echo "  NVCC Version: $$(nvcc --version | grep release)"; \
		echo "  GPU Information:"; \
		nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | head -1; \
		echo "  Tensor Core Support: $$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | grep -i 'V100\|A100\|RTX' >/dev/null && echo 'Available' || echo 'Not Available')"; \
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

# Help target
.PHONY: help
help:
	@echo "Module 8: Domain-Specific Applications - Build System"
	@echo "=================================================="
	@echo ""
	@echo "Build Targets:"
	@echo "  all              - Build all CUDA and HIP domain applications"
	@echo "  cuda             - Build CUDA applications only"
	@echo "  hip              - Build HIP applications only"
	@echo "  debug            - Build with debug flags"
	@echo "  production       - Build with maximum optimization"
	@echo "  clean            - Remove build artifacts"
	@echo ""
	@echo "Domain Application Targets:"
	@echo "  deep_learning    - Build deep learning inference examples"
	@echo "  scientific       - Build scientific computing examples"
	@echo "  image_processing - Build image and signal processing examples"
	@echo "  monte_carlo      - Build Monte Carlo simulation examples"
	@echo "  finance          - Build computational finance examples"
	@echo "  library_integration - Build library integration examples"
	@echo ""
	@echo "Testing Targets:"
	@echo "  test             - Run all domain application tests"
	@echo "  test_<domain>    - Run specific domain application tests"
	@echo "  validate_performance - Validate production performance targets"
	@echo "  validate_accuracy - Validate numerical accuracy"
	@echo "  test_large_scale - Run large-scale problem tests"
	@echo "  test_multi_gpu_scaling - Test multi-GPU scaling"
	@echo ""
	@echo "Benchmarking Targets:"
	@echo "  benchmark_all    - Comprehensive domain application benchmarks"
	@echo "  benchmark_<domain> - Domain-specific benchmarks"
	@echo "  compare_platforms - Compare CUDA vs HIP performance"
	@echo "  analyze_performance_targets - Show domain performance targets"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  profile_cuda     - Profile with Nsight Compute (domain-specific metrics)"
	@echo "  profile_hip      - Profile with ROCProfiler"
	@echo "  analyze_memory_compute - Memory and compute analysis"
	@echo "  check_requirements - Check system requirements"
	@echo "  report           - Generate comprehensive domain performance report"
	@echo ""
	@echo "Examples:"
	@echo "  make deep_learning && make test_deep_learning"
	@echo "  make production && make validate_performance"
	@echo "  make benchmark_all"
	@echo "  make report  # Complete domain application analysis"