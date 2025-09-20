# Module 2: Advanced GPU Memory Management

## Overview
This module focuses on GPU memory hierarchy mastery and performance optimization: shared memory tiling, memory coalescing, texture/read-only memory usage, unified memory, and bandwidth optimization.

## Learning Objectives
After completing this module, you will be able to:
- Organize GPU threads in multidimensional grids
- Map threads efficiently to data structures  
- Implement image processing algorithms on GPU
- Create optimized matrix multiplication kernels
- Handle boundary conditions in multidimensional algorithms

## Module Content
- **[content.md](content.md)** - Complete module content
- **[examples/](examples/)** - Practical code examples

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support OR AMD GPU with ROCm support
- CUDA Toolkit 12.0+ or ROCm 6.0+ (Docker images provide CUDA 12.9.1 and ROCm 7.0)
- C/C++ compiler (GCC, Clang, or MSVC)

Recommended: use our Docker dev environment
```
./docker/scripts/run.sh --auto
```

### Build and Run
```bash
cd modules/module2/examples
make            # auto-detects your GPU and builds accordingly

# Run a few examples (binaries in build/)
./build/01_shared_memory_transpose_cuda    # or _hip on AMD
./build/02_memory_coalescing_cuda          # or _hip on AMD
./build/04_unified_memory_cuda
```

## Topics to be Covered

### 1. Multidimensional Grid Organization
- 2D and 3D thread block configurations
- Grid size calculations for arbitrary data sizes
- Thread-to-data mapping strategies

### 2. Memory Access Patterns
- Coalesced vs strided access
- Structure of Arrays vs Array of Structures
- Read-only/texture cache benefits

### 3. Shared Memory and Tiling
- Tiled transpose with bank-conflict avoidance
- Block-level cooperation and synchronization
- Padding strategies to avoid bank conflicts

### 4. Unified Memory and Bandwidth
- Unified memory prefetch and advice
- Measuring and optimizing memory bandwidth
- Analyzing profiler metrics for memory performance

---
**Duration**: 6-8 hours  
**Difficulty**: Beginner-Intermediate  
**Prerequisites**: Module 1 completion