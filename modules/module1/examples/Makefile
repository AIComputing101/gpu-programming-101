# Module 1: GPU Programming Fundamentals
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
CUDA_FLAGS = -std=c++17 -O2 -arch=sm_70
CUDA_DEBUG_FLAGS = -std=c++17 -g -G -arch=sm_70
HIP_FLAGS = -std=c++17 -O2
HIP_DEBUG_FLAGS = -std=c++17 -g
CXX_FLAGS = -std=c++17 -O2

# Directories
EXAMPLES_DIR = .
BUILD_DIR = build
PROFILE_DIR = profiles

# CUDA Examples
CUDA_SOURCES = $(wildcard $(EXAMPLES_DIR)/*_cuda.cu)
CUDA_TARGETS = $(patsubst $(EXAMPLES_DIR)/%.cu,$(BUILD_DIR)/%,$(CUDA_SOURCES))

# HIP Examples
HIP_SOURCES = $(wildcard $(EXAMPLES_DIR)/*_hip.cpp)
HIP_TARGETS = $(patsubst $(EXAMPLES_DIR)/%.cpp,$(BUILD_DIR)/%,$(HIP_SOURCES))

# Cross-platform Examples (HIP-based, requires hipcc)
CPP_SOURCES = $(wildcard $(EXAMPLES_DIR)/*_comparison.cpp)
CPP_TARGETS = $(patsubst $(EXAMPLES_DIR)/%.cpp,$(BUILD_DIR)/%,$(CPP_SOURCES))

# Check for hipcc availability
HIPCC_AVAILABLE := $(shell command -v hipcc >/dev/null 2>&1 && echo 1 || echo 0)

# Active targets based on detected GPU vendor and compiler availability
ifeq ($(BUILD_CUDA),1)
  ifeq ($(HIPCC_AVAILABLE),1)
    ALL_TARGETS = $(CUDA_TARGETS) $(CPP_TARGETS)
  else
    ALL_TARGETS = $(CUDA_TARGETS)
  endif
else ifeq ($(BUILD_HIP),1)
ALL_TARGETS = $(HIP_TARGETS) $(CPP_TARGETS)
else
  ifeq ($(HIPCC_AVAILABLE),1)
    ALL_TARGETS = $(CPP_TARGETS)
  else
    ALL_TARGETS = 
  endif
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
  ifeq ($(HIPCC_AVAILABLE),1)
	@echo "✓ hipcc available - including cross-platform examples"
  else
	@echo "⚠ hipcc not available - skipping cross-platform examples"
  endif
else ifeq ($(GPU_VENDOR),AMD)
	@echo "✓ AMD GPU detected - building HIP examples"
else
	@echo "⚠ No compatible GPU detected - building CPU examples only"
  ifeq ($(HIPCC_AVAILABLE),0)
	@echo "⚠ hipcc not available - no examples will be built"
  endif
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

# Cross-platform examples (only if hipcc is available)
ifeq ($(HIPCC_AVAILABLE),1)
$(BUILD_DIR)/%_comparison: $(EXAMPLES_DIR)/%_comparison.cpp
	@echo "Building cross-platform example: $@"
	$(HIPCC) $(HIP_FLAGS) $< -o $@
endif

# Debug builds
.PHONY: debug
debug: CUDA_FLAGS = $(CUDA_DEBUG_FLAGS)
debug: HIP_FLAGS = $(HIP_DEBUG_FLAGS)
debug: all

# Clean
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(PROFILE_DIR)

# Help
.PHONY: help
help:
	@echo "Module 1: GPU Programming Fundamentals"
	@echo "Available targets:"
	@echo "  all     - Build all examples for detected GPU vendor"
	@echo "  cuda    - Build CUDA examples (requires NVIDIA GPU)"
	@echo "  hip     - Build HIP examples (requires AMD GPU)"
	@echo "  debug   - Build with debug flags"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"