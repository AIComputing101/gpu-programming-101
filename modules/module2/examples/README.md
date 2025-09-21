# Module 2: Advanced GPU Memory Management Examples

This directory contains practical examples for GPU memory optimization techniques using both CUDA and HIP. These examples accompany Module 2 and focus on shared memory tiling, memory coalescing, texture/read-only memory, unified memory, and bandwidth optimization.

## Learning Objectives

- Master shared memory optimization for high-performance computing
- Understand memory coalescing and its impact on bandwidth
- Learn texture memory usage for spatial locality patterns  
- Explore unified memory programming models
- Optimize memory bandwidth utilization techniques
- Analyze memory access patterns and bottlenecks

## Examples Overview

### 01. Shared Memory Matrix Transpose
**Files:** `01_shared_memory_transpose_cuda.cu`, `01_shared_memory_transpose_hip.cpp`

Demonstrates shared memory optimization through matrix transpose:
- Naive vs shared memory implementations
- Bank conflict avoidance techniques
- Cross-platform HIP optimizations (AMD vs NVIDIA)
- Performance analysis and bandwidth measurements

### 02. Memory Coalescing Analysis
**Files:** `02_memory_coalescing_cuda.cu`, `02_memory_coalescing_hip.cpp`

Comprehensive analysis of memory access patterns:
- Structure of Arrays (SoA) vs Array of Structures (AoS)
- Strided memory access patterns
- Vectorized memory operations (float4)
- Platform-specific optimizations

### 03. Texture Memory Optimization (CUDA Only)
**File:** `03_texture_memory_cuda.cu`

Advanced texture memory usage for spatial locality:
- Modern texture object API
- Hardware interpolation and filtering
- Boundary condition handling
- Texture-based convolution and image processing

### 04. Unified Memory Programming (CUDA Only)
**File:** `04_unified_memory_cuda.cu`

Unified Memory programming model and optimization:
- Automatic data migration between CPU and GPU
- Memory prefetching and hints
- Performance comparison with explicit management
- Memory pool management

### 05. Memory Bandwidth Optimization (CUDA Only)
**File:** `05_memory_bandwidth_optimization_cuda.cu`

Comprehensive memory bandwidth optimization techniques:
- Vectorized memory operations
- Streaming memory patterns
- Pinned memory benefits
- Memory hierarchy utilization

## Building and Running Examples

### Prerequisites
- CUDA Toolkit 12.0+ (for CUDA examples)
- ROCm 6.0+ (for HIP examples)
- Compatible GPU (NVIDIA or AMD)
- C++17 compatible compiler

### Quick Start
```bash
 # Build all examples (auto-detects your GPU)
make

# Run performance tests
make test

# Build specific example
make 01_shared_memory_transpose_cuda
```

## Performance Analysis

### Profiling Commands

**NVIDIA Nsight Compute:**
```bash
# Memory bandwidth analysis
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./build/01_shared_memory_transpose_cuda

# Memory coalescing efficiency
ncu --metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed ./build/02_memory_coalescing_cuda
```

**AMD ROCProfiler:**
```bash
# HIP memory analysis
rocprof --hip-trace ./build/01_shared_memory_transpose_hip

# Detailed memory metrics
rocprof --stats ./build/02_memory_coalescing_hip
```

### Expected Performance Improvements

- **Shared Memory Optimizations:** 2-5x speedup over naive approaches
- **Memory Coalescing:** 2-10x performance difference between coalesced vs strided access
- **Texture Memory:** 1.5-3x speedup for spatial locality patterns

## Notes

These examples are designed to be educational and performance-oriented. Use the provided Docker environment for consistent toolchains (CUDA 12.9.1, ROCm latest). Binaries are emitted to the `build/` directory by the Makefile.