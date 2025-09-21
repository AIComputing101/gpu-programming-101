# Module 1: Foundations of GPU Computing

## Overview
This module introduces the fundamentals of GPU programming using CUDA and HIP. You'll learn about GPU architecture, parallel execution models, memory management, and basic optimization techniques.

## Learning Objectives
After completing this module, you will be able to:
- Understand the fundamental differences between CPU and GPU architectures
- Set up CUDA and HIP development environments  
- Write, compile, and execute basic GPU kernels
- Manage GPU memory allocation and data transfers
- Debug common GPU programming issues
- Apply basic optimization techniques for parallel execution

## Module Content
- **[content.md](content.md)** - Complete module content with theory and explanations
- **[examples/](examples/)** - Practical code examples and exercises

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support OR AMD GPU with ROCm support
- CUDA Toolkit 12.0+ or ROCm 6.0+ (Docker images provide CUDA 12.9.1 and ROCm latest)
- C/C++ compiler (GCC, Clang, or MSVC)

Tip: You can skip native installs by using our Docker environment (recommended):
```
./docker/scripts/run.sh --auto
```

### Running Examples

Navigate to the examples directory:
```bash
cd examples/
```

Build and run examples (binaries are written to `build/`):
```bash
# Build all examples for your detected GPU
make

# Run specific examples (CUDA)
./build/01_vector_addition_cuda
./build/04_device_info_cuda
./build/05_performance_comparison_cuda || ./build/05_performance_comparison

# Or HIP versions (cross-platform)
./build/02_vector_addition_hip
./build/04_device_info_hip
```

## Examples Overview

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `01_vector_addition_cuda.cu` | Basic CUDA vector addition | Kernels, memory management, error handling |
| `02_vector_addition_hip.cpp` | Cross-platform HIP version | HIP API, portability |
| `03_matrix_addition_cuda.cu` | 2D matrix operations | 2D threading, indexing |
| `03_matrix_addition_hip.cpp` | HIP 2D matrix operations | HIP indexing, portability |
| `04_device_info_cuda.cu` | GPU properties and capabilities | Device queries, system info |
| `04_device_info_hip.cpp` | HIP device and platform info | HIP device queries |
| `05_performance_comparison_cuda.cu` | CPU vs GPU benchmarking (CUDA) | Performance analysis, timing |
| `05_performance_comparison_hip.cpp` | Benchmarking (HIP) | HIP performance, memory bandwidth |
| `06_debug_example_cuda.cu` | Debugging and optimization (CUDA) | Error checking, occupancy |
| `06_debug_example_hip.cpp` | Debugging and optimization (HIP) | HIP debugging |
| `07_cross_platform_comparison.cpp` | AMD vs NVIDIA comparison | Portability, tuning |

## Topics Covered

### 1. GPU Architecture
- SIMT (Single Instruction, Multiple Thread) execution model
- Memory hierarchy (global, shared, registers, constant)
- Streaming multiprocessors and warps
- Thread hierarchy (threads → blocks → grids)

### 2. Programming Models
- **CUDA**: NVIDIA's proprietary platform
- **HIP**: Cross-platform alternative for AMD and NVIDIA

### 3. Memory Management
- Host and device memory allocation
- Data transfers between CPU and GPU
- Memory access patterns and optimization

### 4. Parallel Execution
- Thread indexing and coordination
- Block and grid configuration
- Avoiding thread divergence

### 5. Debugging and Optimization
- Error handling best practices
- Performance profiling tools
- Occupancy optimization
- Memory bandwidth utilization

## Learning Path

1. **Start Here**: Read through [content.md](content.md) for comprehensive theory
2. **Setup**: Follow environment setup instructions
3. **Practice**: Work through examples in numerical order
4. **Experiment**: Modify examples with different parameters
5. **Debug**: Use debugging example to learn troubleshooting
6. **Optimize**: Apply performance analysis techniques

## Common Issues and Solutions

### Setup Problems
- **CUDA not found**: Add CUDA to PATH and LD_LIBRARY_PATH
- **No GPU detected**: Check drivers with `nvidia-smi` or `rocm-smi`
- **Compilation errors**: Verify toolkit installation

### Runtime Issues  
- **Out of memory**: Check available GPU memory, reduce problem size
- **Invalid configuration**: Verify block sizes within GPU limits
- **Kernel errors**: Use proper error checking macros

## Next Steps
After completing this module:
- Proceed to Module 2: Multi-Dimensional Data Processing
- Explore advanced memory patterns
- Learn about performance optimization techniques
- Practice with larger, real-world problems

## Additional Resources
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [HIP Programming Guide](https://rocmdocs.amd.com/projects/HIP/en/latest/)
- [GPU Performance Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---
**Duration**: 4-6 hours  
**Difficulty**: Beginner  
**Prerequisites**: Basic C/C++ programming knowledge