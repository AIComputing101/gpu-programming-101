# Module 1 Examples: GPU Programming Fundamentals

This directory contains practical examples that accompany Module 1 of the GPU Programming 101 course. These examples demonstrate the core concepts of CUDA and HIP programming.

## Files Overview

### CUDA Examples (NVIDIA)
| File | Description | Key Concepts |
|------|-------------||--------------|
| `01_vector_addition_cuda.cu` | Basic CUDA vector addition with error handling | Kernels, memory management, error checking |
| `03_matrix_addition_cuda.cu` | 2D matrix addition with thread indexing | 2D threading, grid configuration |
| `04_device_info_cuda.cu` | Query and display GPU properties | Device queries, capability checking |
| `05_performance_comparison.cu` | CPU vs GPU performance benchmark | Performance analysis, timing |
| `06_debug_example.cu` | Debugging techniques and occupancy analysis | Error handling, profiling, optimization |

### HIP Examples (AMD/NVIDIA Cross-Platform)
| File | Description | Key Concepts |
|------|-------------||--------------|
| `02_vector_addition_hip.cpp` | Cross-platform vector addition using HIP | HIP API, portability |
| `03_matrix_addition_hip.cpp` | 2D matrix addition with HIP | Cross-platform 2D threading |
| `04_device_info_hip.cpp` | HIP device properties and platform detection | HIP device queries, platform abstraction |
| `05_performance_comparison_hip.cpp` | HIP performance analysis with bandwidth testing | HIP performance, memory bandwidth |
| `06_debug_example_hip.cpp` | HIP debugging and optimization techniques | HIP debugging, occupancy analysis |
| `07_cross_platform_comparison.cpp` | AMD vs NVIDIA optimization comparison | Platform-specific optimizations, portability |

## Prerequisites

### For CUDA Examples
- NVIDIA GPU with compute capability 3.5+
- NVIDIA drivers (version 450+)
- CUDA Toolkit 11.0+
- GCC/Clang compiler

### For HIP Examples
- AMD GPU with ROCm support OR NVIDIA GPU
- ROCm 4.0+ (for AMD) or CUDA 11.0+ (for NVIDIA backend)
- HIP compiler (hipcc)

## Quick Start

### Using the Makefile

```bash
# Build all CUDA examples
make cuda

# Build all HIP examples
make hip

# Run tests
make test

# Clean build files
make clean

# Show help
make help
```

### Manual Compilation

**CUDA Examples:**
```bash
nvcc -o vector_add 01_vector_addition_cuda.cu
nvcc -o matrix_add 03_matrix_addition_cuda.cu
nvcc -o device_info 04_device_info_cuda.cu
nvcc -o performance 05_performance_comparison.cu
nvcc -o debug 06_debug_example.cu
```

**HIP Examples:**
```bash
hipcc -o vector_add_hip 02_vector_addition_hip.cpp
hipcc -o matrix_add_hip 03_matrix_addition_hip.cpp
hipcc -o device_info_hip 04_device_info_hip.cpp
hipcc -o performance_hip 05_performance_comparison_hip.cpp
hipcc -o debug_hip 06_debug_example_hip.cpp
hipcc -o cross_platform 07_cross_platform_comparison.cpp
```

## Example Descriptions

### 1. Vector Addition (CUDA)
**File:** `01_vector_addition_cuda.cu`

Demonstrates:
- Basic kernel structure
- Memory allocation and transfer
- Thread indexing
- Error handling with macros

**Usage:**
```bash
make vector_add_cuda
./vector_add_cuda
```

**Expected Output:**
```
Launching kernel with 4 blocks of 256 threads each
Verification (first 5 elements):
0.000 + 1.000 = 1.000
0.708 + 0.293 = 1.000
...
Vector addition completed successfully!
```

### 2. Vector Addition (HIP)
**File:** `02_vector_addition_hip.cpp`

Demonstrates:
- Cross-platform GPU programming
- HIP API usage
- Device property queries
- Portability between AMD and NVIDIA

**Usage:**
```bash
make vector_add_hip
./vector_add_hip
```

### 3. Matrix Addition (CUDA)
**File:** `03_matrix_addition_cuda.cu`

Demonstrates:
- 2D thread indexing
- 2D grid and block configuration
- Boundary checking for matrices
- Performance with larger datasets

**Usage:**
```bash
make matrix_add_cuda
./matrix_add_cuda
```

### 3b. Matrix Addition (HIP)
**File:** `03_matrix_addition_hip.cpp`

Demonstrates:
- Cross-platform 2D matrix operations
- HIP-specific device queries
- Memory usage reporting
- Platform-agnostic thread indexing

**Usage:**
```bash
make matrix_add_hip
./matrix_add_hip
```

### 4. Device Information (CUDA)
**File:** `04_device_info_cuda.cu`

Demonstrates:
- Querying GPU capabilities
- Memory information
- Compute capability checking
- Multi-GPU systems

**Usage:**
```bash
make device_info_cuda
./device_info_cuda
```

### 4b. Device Information (HIP)
**File:** `04_device_info_hip.cpp`

Demonstrates:
- Cross-platform device queries
- AMD vs NVIDIA feature detection
- HIP runtime and driver versions
- Platform-specific properties

**Usage:**
```bash
make device_info_hip
./device_info_hip
```

### 5. Performance Comparison (CUDA)
**File:** `05_performance_comparison.cu`

Demonstrates:
- CPU vs GPU benchmarking
- Memory bandwidth analysis
- Performance scaling with problem size
- Timing with CUDA events

**Usage:**
```bash
make performance_cuda
./performance_cuda
```

### 5b. Performance Comparison (HIP)
**File:** `05_performance_comparison_hip.cpp`

Demonstrates:
- Cross-platform performance analysis
- HIP event-based timing
- Memory bandwidth efficiency
- Block size optimization analysis
- Platform-specific performance characteristics

**Usage:**
```bash
make performance_hip
./performance_hip
```

### 6. Debug Example (CUDA)
**File:** `06_debug_example.cu`

Demonstrates:
- Debug printf in kernels
- Occupancy analysis
- Shared memory usage
- Error checking techniques

**Usage:**
```bash
make debug_cuda
./debug_cuda
```

### 6b. Debug Example (HIP)
**File:** `06_debug_example_hip.cpp`

Demonstrates:
- HIP debugging techniques
- Cross-platform occupancy analysis
- HIP event timing
- Platform-specific feature detection
- Warp-level operations

**Usage:**
```bash
make debug_hip
./debug_hip
```

### 7. Cross-Platform Comparison
**File:** `07_cross_platform_comparison.cpp`

Demonstrates:
- Writing portable HIP code
- AMD vs NVIDIA optimizations
- Platform-specific feature detection
- Performance comparison across platforms
- Memory bandwidth analysis

**Usage:**
```bash
make cross_platform
./cross_platform
```

## Common Issues and Solutions

### Compilation Errors

**"nvcc: command not found"**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**"No CUDA-capable device"**
- Check `nvidia-smi` output
- Verify driver installation
- Ensure GPU is not in exclusive mode

**HIP compilation issues**
```bash
# For AMD GPUs
export HIP_PLATFORM=amd

# For NVIDIA GPUs  
export HIP_PLATFORM=nvidia
```

### Runtime Errors

**"CUDA error: out of memory"**
- Reduce problem size
- Check available memory with device_info example
- Use memory profiling tools

**"CUDA error: invalid configuration"**
- Check block size limits with device_info
- Ensure block size ≤ 1024 threads
- Verify grid dimensions are within limits

## Performance Tips

1. **Block Size Selection**
   - Use multiples of 32 (warp size)
   - Common choices: 128, 256, 512
   - Use occupancy calculator for optimization

2. **Memory Access**
   - Prefer coalesced memory access patterns
   - Minimize CPU-GPU data transfers
   - Use appropriate memory types (global, shared, constant)

3. **Thread Divergence**
   - Avoid conditional branches within warps
   - Restructure algorithms to minimize divergence

## Learning Paths

### CUDA-First Path (NVIDIA GPUs)
1. Start with `01_vector_addition_cuda.cu` to understand basics
2. Explore `04_device_info_cuda.cu` to learn about your GPU
3. Try `03_matrix_addition_cuda.cu` for 2D indexing
4. Run `05_performance_comparison.cu` to see GPU advantages
5. Use `06_debug_example.cu` to learn debugging techniques
6. Experiment with `02_vector_addition_hip.cpp` for portability

### HIP-First Path (AMD GPUs or Cross-Platform)
1. Start with `02_vector_addition_hip.cpp` for HIP basics
2. Explore `04_device_info_hip.cpp` to understand your platform
3. Try `03_matrix_addition_hip.cpp` for 2D operations
4. Run `05_performance_comparison_hip.cpp` for benchmarking
5. Use `06_debug_example_hip.cpp` for debugging techniques
6. Test `07_cross_platform_comparison.cpp` for optimization

### Comparison Path (Both Platforms Available)
1. Run corresponding CUDA and HIP examples side by side
2. Compare performance characteristics
3. Test portability with `07_cross_platform_comparison.cpp`
4. Experiment with platform-specific optimizations

## Additional Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
- [GPU Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## Exercises

Try these modifications to deepen your understanding:

1. **Modify vector addition** to use different block sizes (64, 512, 1024)
2. **Add timing** to measure kernel execution time
3. **Implement element-wise** mathematical operations (sin, cos, exp)
4. **Create a 2D matrix multiplication** kernel
5. **Add input validation** and better error handling
6. **Port CUDA examples** to HIP using hipify tools

Happy GPU programming! 🚀