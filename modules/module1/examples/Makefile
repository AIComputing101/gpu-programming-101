# GPU Programming Examples Makefile
# Supports both CUDA and HIP compilation with consistent naming

# Compiler settings
NVCC = nvcc
HIPCC = hipcc
NVCC_FLAGS = -O2 -std=c++11 -arch=sm_50
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

# Cross-platform example (HIP only)
cross_platform: 07_cross_platform_comparison.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

# Individual example targets
01_vector_addition_cuda: 01_vector_addition_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

01_vector_addition_hip: 01_vector_addition_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

02_matrix_addition_cuda: 02_matrix_addition_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

02_matrix_addition_hip: 02_matrix_addition_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

03_matrix_multiplication_cuda: 03_matrix_multiplication_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

03_matrix_multiplication_hip: 03_matrix_multiplication_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

04_device_info_cuda: 04_device_info_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

04_device_info_hip: 04_device_info_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

05_performance_comparison_cuda: 05_performance_comparison_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

05_performance_comparison_hip: 05_performance_comparison_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

06_debug_example_cuda: 06_debug_example_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

06_debug_example_hip: 06_debug_example_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

# Legacy targets for backward compatibility
vector_add_cuda: 01_vector_addition_cuda
vector_add_hip: 01_vector_addition_hip
matrix_add_cuda: 02_matrix_addition_cuda
matrix_add_hip: 02_matrix_addition_hip
matrix_mult_cuda: 03_matrix_multiplication_cuda
matrix_mult_hip: 03_matrix_multiplication_hip
device_info_cuda: 04_device_info_cuda
device_info_hip: 04_device_info_hip
performance_cuda: 05_performance_comparison_cuda
performance_hip: 05_performance_comparison_hip
debug_cuda: 06_debug_example_cuda
debug_hip: 06_debug_example_hip

# Generic legacy targets (default to CUDA)
vector_add: 01_vector_addition_cuda
matrix_add: 02_matrix_addition_cuda
matrix_mult: 03_matrix_multiplication_cuda
device_info: 04_device_info_cuda
performance: 05_performance_comparison_cuda
debug: 06_debug_example_cuda

# Test targets
test: test_cuda

test_cuda: cuda
	@echo "Running CUDA tests..."
	@if command -v nvidia-smi > /dev/null; then \
		echo "=== Testing CUDA Examples ==="; \
		echo "1. Vector Addition..."; \
		./01_vector_addition_cuda || echo "✗ Vector addition failed"; \
		echo "2. Matrix Addition..."; \
		./02_matrix_addition_cuda || echo "✗ Matrix addition failed"; \
		echo "3. Matrix Multiplication..."; \
		./03_matrix_multiplication_cuda || echo "✗ Matrix multiplication failed"; \
		echo "4. Device Info..."; \
		./04_device_info_cuda || echo "✗ Device info failed"; \
		echo "5. Performance Comparison..."; \
		./05_performance_comparison_cuda || echo "✗ Performance test failed"; \
		echo "6. Debug Example..."; \
		./06_debug_example_cuda || echo "✗ Debug example failed"; \
		echo "✓ CUDA tests completed"; \
	else \
		echo "No NVIDIA GPU detected, skipping CUDA tests"; \
	fi

test_hip: hip cross_platform
	@echo "Running HIP tests..."
	@if command -v rocm-smi > /dev/null || command -v nvidia-smi > /dev/null; then \
		echo "=== Testing HIP Examples ==="; \
		echo "1. Vector Addition..."; \
		./01_vector_addition_hip || echo "✗ HIP vector addition failed"; \
		echo "2. Matrix Addition..."; \
		./02_matrix_addition_hip || echo "✗ HIP matrix addition failed"; \
		echo "3. Matrix Multiplication..."; \
		./03_matrix_multiplication_hip || echo "✗ HIP matrix multiplication failed"; \
		echo "4. Device Info..."; \
		./04_device_info_hip || echo "✗ HIP device info failed"; \
		echo "5. Performance Comparison..."; \
		./05_performance_comparison_hip || echo "✗ HIP performance test failed"; \
		echo "6. Debug Example..."; \
		./06_debug_example_hip || echo "✗ HIP debug example failed"; \
		echo "7. Cross-Platform Comparison..."; \
		./07_cross_platform_comparison || echo "✗ Cross-platform test failed"; \
		echo "✓ HIP tests completed"; \
	else \
		echo "No compatible GPU detected, skipping HIP tests"; \
	fi

test_all: test_cuda test_hip

# Quick test - just compile everything
test_compile: cuda hip cross_platform
	@echo "✓ All examples compiled successfully"

# Clean targets
clean:
	rm -f $(CUDA_EXECUTABLES) $(HIP_EXECUTABLES)
	rm -f 01_vector_addition_cuda 01_vector_addition_hip
	rm -f 02_matrix_addition_cuda 02_matrix_addition_hip  
	rm -f 03_matrix_multiplication_cuda 03_matrix_multiplication_hip
	rm -f 04_device_info_cuda 04_device_info_hip
	rm -f 05_performance_comparison_cuda 05_performance_comparison_hip
	rm -f 06_debug_example_cuda 06_debug_example_hip
	rm -f 07_cross_platform_comparison
	rm -f vector_add_cuda vector_add_hip matrix_add_cuda matrix_add_hip
	rm -f matrix_mult_cuda matrix_mult_hip device_info_cuda device_info_hip
	rm -f performance_cuda performance_hip debug_cuda debug_hip
	rm -f vector_add matrix_add matrix_mult device_info performance debug cross_platform

# List available examples
list:
	@echo "Available Examples:"
	@echo "=================="
	@echo ""
	@echo "CUDA Examples:"
	@ls -1 *_cuda.cu 2>/dev/null | sed 's/_cuda.cu//' | nl -w2 -s'. '
	@echo ""
	@echo "HIP Examples:"
	@ls -1 *_hip.cpp 2>/dev/null | sed 's/_hip.cpp//' | nl -w2 -s'. '
	@echo ""
	@echo "Cross-Platform:"
	@ls -1 07_cross_platform_comparison.cpp 2>/dev/null | sed 's/.cpp//' | nl -w2 -s'. '

# Help target
help:
	@echo "GPU Programming 101 - Module 1 Examples Makefile"
	@echo "================================================"
	@echo ""
	@echo "Build Targets:"
	@echo "  all               - Build all CUDA examples (default)"
	@echo "  cuda              - Build all CUDA examples"
	@echo "  hip               - Build all HIP examples"
	@echo "  cross_platform    - Build cross-platform comparison example"
	@echo ""
	@echo "Test Targets:"
	@echo "  test              - Run CUDA tests"
	@echo "  test_cuda         - Run CUDA tests"
	@echo "  test_hip          - Run HIP tests"
	@echo "  test_all          - Run both CUDA and HIP tests"
	@echo "  test_compile      - Test compilation only"
	@echo ""
	@echo "Individual Examples (CUDA):"
	@echo "  01_vector_addition_cuda        - Vector addition"
	@echo "  02_matrix_addition_cuda        - Matrix addition"
	@echo "  03_matrix_multiplication_cuda  - Matrix multiplication"
	@echo "  04_device_info_cuda            - Device information"
	@echo "  05_performance_comparison_cuda - Performance comparison"
	@echo "  06_debug_example_cuda          - Debug example"
	@echo ""
	@echo "Individual Examples (HIP):"
	@echo "  01_vector_addition_hip         - Vector addition"
	@echo "  02_matrix_addition_hip         - Matrix addition"
	@echo "  03_matrix_multiplication_hip   - Matrix multiplication"
	@echo "  04_device_info_hip             - Device information"
	@echo "  05_performance_comparison_hip  - Performance comparison"
	@echo "  06_debug_example_hip           - Debug example"
	@echo "  07_cross_platform_comparison   - Cross-platform comparison"
	@echo ""
	@echo "Legacy Targets:"
	@echo "  vector_add, matrix_add, matrix_mult, device_info, performance, debug"
	@echo "  (These default to CUDA versions)"
	@echo ""
	@echo "Utility Targets:"
	@echo "  list              - List all available examples"
	@echo "  clean             - Remove all executables"
	@echo "  help              - Show this help"
	@echo ""
	@echo "Example Usage:"
	@echo "  make cuda                              # Build all CUDA examples"
	@echo "  make hip                               # Build all HIP examples"
	@echo "  make 01_vector_addition_cuda           # Build specific example"
	@echo "  make test_all                          # Test everything"
	@echo ""
	@echo "Requirements:"
	@echo "  CUDA: nvcc compiler and NVIDIA GPU"
	@echo "  HIP:  hipcc compiler and AMD/NVIDIA GPU"

.PHONY: all cuda hip cross_platform test test_cuda test_hip test_all test_compile clean list help
.PHONY: vector_add_cuda vector_add_hip matrix_add_cuda matrix_add_hip matrix_mult_cuda matrix_mult_hip
.PHONY: device_info_cuda device_info_hip performance_cuda performance_hip debug_cuda debug_hip
.PHONY: vector_add matrix_add matrix_mult device_info performance debug