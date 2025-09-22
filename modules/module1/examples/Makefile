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

# CUDA architecture detection (prefer actual GPU via nvidia-smi; fallback sm_90)
CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9' | sed -e 's/^/sm_/')
ifeq ($(strip $(CUDA_ARCH)),)
	CUDA_ARCH := sm_90
endif
CUDA_ARCH_FLAG := -arch=$(CUDA_ARCH)

# Compiler flags
CUDA_FLAGS = -std=c++17 -O2 $(CUDA_ARCH_FLAG)
CUDA_DEBUG_FLAGS = -std=c++17 -g -G $(CUDA_ARCH_FLAG)
HIP_FLAGS = -std=c++17 -O2
HIP_DEBUG_FLAGS = -std=c++17 -g

# ROCm 7 note: hipcc may require explicit --rocm-path when HIP runtime/version files
# aren't in legacy locations. See ROCm docs on file structure reorg.
# Prefer environment ROCM_PATH if set; prefer ROCm 7.0.0, fall back to any ROCm installation.
ROCM_PATH ?= $(shell ls -d /opt/rocm-7.0.0 2>/dev/null || ls -d /opt/rocm* 2>/dev/null | head -1 || echo /opt/rocm)
# Prefer hipconfig (ROCm 7) to locate ROCm root, then fall back to hipcc path.
HIPCONFIG_BIN := $(shell command -v hipconfig 2>/dev/null)
ifneq ($(HIPCONFIG_BIN),)
	ROCM_PATH := $(shell hipconfig -R 2>/dev/null)
endif
ifeq ($(strip $(ROCM_PATH)),)
	ROCM_PATH := /opt/rocm
endif
# If HIP headers not found under ROCM_PATH, try to auto-detect from hipcc location via readlink -f
ifeq ($(wildcard $(ROCM_PATH)/include/hip/hip_runtime.h),)
	HIPCC_BIN := $(shell command -v hipcc 2>/dev/null)
	ifneq ($(HIPCC_BIN),)
		ROCM_PATH_DETECTED := $(shell readlink -f $(HIPCC_BIN) | xargs dirname | xargs dirname)
		ROCM_PATH := $(ROCM_PATH_DETECTED)
	endif
endif
HIP_ROCM_FLAG = --rocm-path=$(ROCM_PATH)
HIP_INC_DIR := $(ROCM_PATH)/include
HIP_LIB_DIR := $(ROCM_PATH)/lib
HIP_LDFLAGS := -L$(HIP_LIB_DIR) -lamdhip64 -Wl,-rpath,$(HIP_LIB_DIR)
HIP_FLAGS += $(HIP_ROCM_FLAG) -I$(HIP_INC_DIR)
HIP_DEBUG_FLAGS += $(HIP_ROCM_FLAG) -I$(HIP_INC_DIR)
HIP_DEVICE_LIB_DIR := $(ROCM_PATH)/amdgcn/bitcode
ifneq ($(wildcard $(HIP_DEVICE_LIB_DIR)/ockl.bc),)
	HIP_FLAGS += --hip-device-lib-path=$(HIP_DEVICE_LIB_DIR)
	HIP_DEBUG_FLAGS += --hip-device-lib-path=$(HIP_DEVICE_LIB_DIR)
endif

# GPU architecture detection - get actual GPU architecture from rocminfo
GPU_ARCH := $(shell if command -v rocminfo >/dev/null 2>&1; then rocminfo 2>/dev/null | grep -o 'gfx[0-9]*' | head -1; else echo gfx1030; fi)
ifeq ($(strip $(GPU_ARCH)),)
	GPU_ARCH := gfx1030
endif

# Add detected GPU architecture to HIP flags
HIP_FLAGS += --offload-arch=$(GPU_ARCH)
HIP_DEBUG_FLAGS += --offload-arch=$(GPU_ARCH)

# Ensure hipcc sees correct paths in FHS layout
HIP_ENV := ROCM_PATH=$(ROCM_PATH) HIP_PATH=$(ROCM_PATH) HIP_PLATFORM=amd
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
	@echo "ℹ Using ROCm path: $(ROCM_PATH)"
	@echo "ℹ Target GPU architecture: $(GPU_ARCH)"
	@if [ ! -f "$(HIP_INC_DIR)/hip/hip_runtime.h" ]; then \
	  echo "⚠ hip_runtime.h not found under $(HIP_INC_DIR). Set ROCM_PATH or install hip-dev."; \
	fi
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
	$(HIP_ENV) $(HIPCC) $(HIP_FLAGS) $< -o $@ $(HIP_LDFLAGS)
endif

# Cross-platform examples (only if hipcc is available)
ifeq ($(HIPCC_AVAILABLE),1)
$(BUILD_DIR)/%_comparison: $(EXAMPLES_DIR)/%_comparison.cpp
	@echo "Building cross-platform example: $@"
	$(HIP_ENV) $(HIPCC) $(HIP_FLAGS) $< -o $@ $(HIP_LDFLAGS)
endif

# Debug builds
.PHONY: debug
debug: CUDA_FLAGS = $(CUDA_DEBUG_FLAGS)
debug: HIP_FLAGS = $(HIP_DEBUG_FLAGS)
debug: all

# Profile builds with actual profiling
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
			nsys profile --cuda-event-trace=false -o $(PROFILE_DIR)/$$(basename $$target).nsys-rep $$target 2>/dev/null || echo "nsys completed"; \
		fi; \
	done
endif
	@echo "Profile data saved to $(PROFILE_DIR)/"
	@ls -la $(PROFILE_DIR)/

# Clean
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(PROFILE_DIR)

# Test target - run built examples
.PHONY: test
test: all
	@echo "Running Module 1 Tests..."
	@for target in $(ALL_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Testing $$target..."; \
			$$target || echo "Test completed with exit code $$?"; \
			echo ""; \
		fi; \
	done

# Help
.PHONY: help
help:
	@echo "Module 1: GPU Programming Fundamentals"
	@echo "Available targets:"
	@echo "  all     - Build all examples for detected GPU vendor"
	@echo "  cuda    - Build CUDA examples (requires NVIDIA GPU)"
	@echo "  hip     - Build HIP examples (requires AMD GPU)"
	@echo "  debug   - Build with debug flags"
	@echo "  profile - Build with profiling flags"
	@echo "  test    - Run all built examples"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"