# Module 1: Foundations of GPU Programming with CUDA and HIP
*Heterogeneous Data Parallel Computing*

> Environment note: Examples are validated in containers using CUDA 13.0.1 (Ubuntu 24.04) and ROCm 7.0.1 (Ubuntu 24.04). The advanced build system automatically detects your GPU vendor and optimizes accordingly. Using Docker is recommended for a consistent setup.

## Learning Objectives
After completing this module, you will be able to:
- Understand the fundamental differences between CPU and GPU architectures
- Set up CUDA and HIP development environments
- Write, compile, and execute basic GPU kernels
- Manage GPU memory allocation and data transfers
- Debug common GPU programming issues
- Apply basic optimization techniques for parallel execution

## 1.1 Introduction to GPU Programming

Welcome to the world of GPU programming! This module introduces the fundamentals of programming GPUs using NVIDIA's CUDA and AMD's HIP within the ROCm framework. We'll use the NVIDIA H100 and AMD MI300X as our reference platforms - representing the cutting edge of parallel computing in 2025.

### Why GPU Programming?

Traditional CPUs excel at sequential processing with complex instruction sets and large caches, while GPUs are designed for massively parallel workloads. Consider these key differences:

**CPU Architecture:**
- Few cores (8-64 typically)
- Complex out-of-order execution
- Large caches (MB per core)
- Optimized for single-threaded performance

**GPU Architecture:**
- Thousands of cores (10,000+ on H100)
- Simple in-order execution
- Small caches shared among many cores
- Optimized for throughput, not latency

This makes GPUs ideal for:
- Machine learning training and inference
- Scientific simulations (weather, molecular dynamics)
- Image and video processing
- Cryptocurrency mining
- Real-time graphics rendering

### Target Hardware Platforms

**NVIDIA H100:**
- 141GB HBM3 memory
- 3,000+ TFLOPS of performance
- 14,592 CUDA cores
- Memory bandwidth: 3.35 TB/s

**AMD MI300X:**
- 192GB HBM3e memory
- 5.2 TB/s memory bandwidth
- 304 Compute Units
- Advanced packaging with CPU and GPU on same die

## 1.2 GPU Architecture Deep Dive

### SIMT (Single Instruction, Multiple Thread) Model

GPUs use the SIMT execution model, where groups of threads execute the same instruction on different data elements. This is different from SIMD (Single Instruction, Multiple Data) because threads can diverge and follow different execution paths.

**Key Concepts:**
- **Warp (NVIDIA)** or **Wavefront (AMD)**: Group of 32 threads that execute together
- **Streaming Multiprocessor (SM)**: Contains multiple warp schedulers and execution units
- **Thread Hierarchy**: Individual threads → Warps → Thread Blocks → Grid

### Memory Hierarchy

Understanding GPU memory is crucial for performance:

```
┌─────────────────────────────────────┐
│        Host (CPU) Memory            │ ← Slowest, largest
├─────────────────────────────────────┤
│        Global GPU Memory            │ ← Large, moderate speed
├─────────────────────────────────────┤
│        Shared Memory               │ ← Fast, small, per-block
├─────────────────────────────────────┤
│        Registers                   │ ← Fastest, smallest, per-thread
└─────────────────────────────────────┘
```

**Memory Types:**
1. **Global Memory**: Large but slow, accessible by all threads
2. **Shared Memory**: Fast, on-chip memory shared within a thread block
3. **Registers**: Fastest memory, private to each thread
4. **Constant Memory**: Read-only, cached, good for frequently accessed data
5. **Texture Memory**: Specialized for spatial locality access patterns

### Programming Frameworks

**CUDA (Compute Unified Device Architecture)**
- NVIDIA's proprietary platform
- Mature ecosystem with extensive libraries (cuBLAS, cuDNN, cuFFT)
- C/C++ extensions with device-specific optimizations
- Comprehensive debugging and profiling tools

**HIP (Heterogeneous-Compute Interface for Portability)**
- AMD's open-source alternative
- Designed for portability between AMD and NVIDIA GPUs
- Can compile to both ROCm (AMD) and CUDA (NVIDIA) backends
- Growing ecosystem with ROCm libraries

## 1.3 Setting Up Your Development Environment

### Prerequisites
- A compatible GPU (NVIDIA or AMD)
- Linux, Windows, or macOS (Linux recommended for production)
- C/C++ compiler (gcc, clang, or MSVC)
- Basic command-line knowledge

### CUDA Development Setup (NVIDIA)

**Step 1: Install NVIDIA Drivers**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-550

# Check installation
nvidia-smi
```

**Step 2: Install CUDA Toolkit**
```bash
# Download from developer.nvidia.com
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

**Step 3: Verify Installation**
```bash
nvcc --version
nvidia-smi
```

### HIP Development Setup (AMD ROCm)

**Step 1: Install ROCm**
```bash
# Ubuntu 22.04/24.04
wget https://repo.radeon.com/amdgpu-install/7.0/ubuntu/jammy/amdgpu-install_7.0.60000-1_all.deb
sudo apt install ./amdgpu-install_7.0.60000-1_all.deb
sudo amdgpu-install --usecase=hiplibsdk,rocm

# Add user to video group
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
```

**Step 2: Verify Installation**
```bash
hipcc --version
rocminfo
/opt/rocm/bin/rocm-smi
```

### Development Tools

**Recommended IDEs:**
- Visual Studio Code with CUDA/HIP extensions
- CLion with GPU debugging support
- Nsight Eclipse Edition (NVIDIA)

**Essential Tools:**
- `nvprof`/`ncu` (NVIDIA profiling)
- `rocprof` (AMD profiling)
- `cuda-gdb`/`rocgdb` (debugging)

### Quick System Check

Create this test script to verify your setup:

```bash
#!/bin/bash
echo "=== GPU Programming Environment Check ==="

# Check for NVIDIA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
fi

# Check for AMD
if command -v rocm-smi &> /dev/null; then
    echo "✓ AMD GPU detected"
    rocm-smi --showproductname --showmeminfo vram
fi

# Check compilers
if command -v nvcc &> /dev/null; then
    echo "✓ NVCC available: $(nvcc --version | grep release)"
fi

if command -v hipcc &> /dev/null; then
    echo "✓ HIPCC available: $(hipcc --version | head -1)"
fi
```

## 1.4 Your First CUDA Program: Vector Addition

### Understanding CUDA Program Structure

A CUDA program consists of:
1. **Host code**: Runs on CPU, manages GPU
2. **Device code**: Runs on GPU, performs parallel computation
3. **Memory management**: Allocate/transfer data between CPU and GPU
4. **Kernel launch**: Execute parallel function on GPU

### Complete Vector Addition Example

```cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA kernel - runs on GPU
__global__ void addVectors(float *a, float *b, float *c, int n) {
    // Calculate global thread index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Boundary check to prevent out-of-bounds access
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    const int N = 1024;  // Vector size
    const int bytes = N * sizeof(float);
    
    // Host vectors
    float *h_a, *h_b, *h_c;
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel: 4 blocks of 256 threads each
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;  // Ceiling division
    
    addVectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Verify result (first few elements)
    printf("Verification (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("%.3f + %.3f = %.3f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Clean up memory
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    printf("Vector addition completed successfully!\n");
    return 0;
}
```

### Key CUDA Concepts Explained

**1. Kernel Function (`__global__`)**
- Executed on GPU by many threads in parallel
- Cannot return values (use void)
- Accessed from host code

**2. Thread Indexing**
```cuda
int i = threadIdx.x + blockIdx.x * blockDim.x;
```
- `threadIdx.x`: Thread ID within block (0-255 for blockSize=256)
- `blockIdx.x`: Block ID within grid
- `blockDim.x`: Number of threads per block

**3. Memory Management**
- `cudaMalloc()`: Allocate GPU memory
- `cudaMemcpy()`: Transfer data between CPU and GPU
- `cudaFree()`: Free GPU memory

**4. Kernel Launch Syntax**
```cuda
kernelName<<<gridSize, blockSize>>>(parameters);
```

### Compilation and Execution

```bash
# Compile
nvcc -o 01_vector_addition_cuda 01_vector_addition_cuda.cu

# Run
./01_vector_addition_cuda
```

Expected output:
```
Verification (first 5 elements):
0.000 + 1.000 = 1.000
0.708 + 0.293 = 1.000
0.841 + 0.159 = 1.000
0.990 + 0.010 = 1.000
0.654 + 0.346 = 1.000
Vector addition completed successfully!
```

## 1.5 Your First HIP Program: Cross-Platform Vector Addition

### HIP: Write Once, Run Anywhere

HIP provides a single-source C++ API that works on both AMD and NVIDIA GPUs. You can write HIP code that compiles for:
- AMD GPUs using ROCm backend
- NVIDIA GPUs using CUDA backend

### Complete HIP Vector Addition Example

```cpp
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// HIP kernel - runs on GPU (AMD or NVIDIA)
__global__ void addVectors(float *a, float *b, float *c, int n) {
    // HIP provides both hipThreadIdx_x and threadIdx.x syntax
    int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    
    // Boundary check
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// HIP error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    const int N = 1024;
    const int bytes = N * sizeof(float);
    
    // Host vectors
    float *h_a, *h_b, *h_c;
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Get device information
    int device;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    printf("Running on: %s\n", props.name);
    
    // Allocate host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }
    
    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));
    
    // Launch configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Method 1: Modern HIP kernel launch (recommended)
    addVectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    // Alternative Method 2: Legacy HIP launch syntax
    // hipLaunchKernelGGL(addVectors, dim3(gridSize), dim3(blockSize), 0, 0, d_a, d_b, d_c, N);
    
    // Check for kernel launch errors
    HIP_CHECK(hipGetLastError());
    
    // Wait for GPU to finish
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));
    
    // Verify result
    printf("Verification (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("%.3f + %.3f = %.3f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Clean up memory
    free(h_a); free(h_b); free(h_c);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    
    printf("HIP vector addition completed successfully!\n");
    return 0;
}
```

### Key HIP Features

**1. Platform Abstraction**
- Same source code works on AMD and NVIDIA
- Runtime determines which backend to use
- Automatic optimization for target architecture

**2. API Compatibility**
```cpp
// HIP provides both syntaxes:
hipThreadIdx_x  // HIP-specific
threadIdx.x     // CUDA-compatible
```

**3. Memory Management**
- `hipMalloc()`: Allocate GPU memory
- `hipMemcpy()`: Transfer data
- `hipFree()`: Free GPU memory
- Same semantics as CUDA equivalents

### Compilation Options

**For AMD GPUs (ROCm):**
```bash
hipcc -o 01_vector_addition_hip 01_vector_addition_hip.cpp
./01_vector_addition_hip
```

**For NVIDIA GPUs (CUDA backend):**
```bash
hipcc --amdgpu-target=none -o 01_vector_addition_hip 01_vector_addition_hip.cpp
./01_vector_addition_hip
```

### Converting CUDA to HIP

HIP provides automated conversion tools:

```bash
# Convert CUDA file to HIP
hipify-perl 01_vector_addition_cuda.cu > 01_vector_addition_hip.cpp

# Or use the Python version
hipify-python 01_vector_addition_cuda.cu --output 01_vector_addition_hip.cpp
```

### Performance Comparison Framework

```cpp
#include <hip/hip_runtime.h>
#include <chrono>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start, end;
public:
    void startTimer() { start = std::chrono::high_resolution_clock::now(); }
    void stopTimer() { end = std::chrono::high_resolution_clock::now(); }
    double getElapsedMs() {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// Add this to your main function:
Timer timer;
timer.startTimer();
addVectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
HIP_CHECK(hipDeviceSynchronize());
timer.stopTimer();
printf("Kernel execution time: %.3f ms\n", timer.getElapsedMs());
```

## 1.6 Understanding GPU Parallel Execution Model

### Thread Hierarchy Breakdown

GPU threads are organized in a three-level hierarchy:

```
Grid
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread N-1
├── Block 1
│   ├── Thread 0
│   ├── ...
└── Block M-1
```

**Key Concepts:**

1. **Thread**: Smallest execution unit, runs one kernel instance
2. **Block**: Group of threads that can cooperate and share memory
3. **Grid**: Collection of blocks executing the same kernel

### Thread Indexing in Detail

Understanding thread indexing is crucial for parallel algorithms:

```cuda
// 1D indexing (most common)
int tid = threadIdx.x + blockIdx.x * blockDim.x;

// 2D indexing (for matrix operations)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int tid = row * width + col;

// 3D indexing (for volume processing)
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

### Practical Example: Matrix Addition

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixAdd(float *A, float *B, float *C, int width, int height) {
    // 2D thread indexing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check
    if (row < height && col < width) {
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize matrices
    for (int i = 0; i < width * height; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Define block and grid sizes
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    printf("Grid size: (%d, %d)\n", gridSize.x, gridSize.y);
    printf("Block size: (%d, %d)\n", blockSize.x, blockSize.y);
    printf("Total threads: %d\n", gridSize.x * gridSize.y * blockSize.x * blockSize.y);
    
    // Launch kernel
    matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    bool success = true;
    for (int i = 0; i < width * height; i++) {
        if (h_C[i] != 3.0f) {
            success = false;
            break;
        }
    }
    
    printf("Matrix addition %s\n", success ? "PASSED" : "FAILED");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}
```

### Optimizing Block Size

Block size significantly affects performance. Consider these factors:

**Occupancy**: How well you utilize GPU resources
```cuda
// Check occupancy for your kernel
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixAdd, 0, 0);
printf("Suggested block size: %d\n", blockSize);
```

**Common Block Sizes:**
- **32, 64, 128, 256, 512**: Power of 2 sizes work well
- **16x16, 32x32**: Good for 2D problems
- **Avoid**: Very small (<32) or very large (>1024) blocks

### Practical Exercise: Thread Divergence

Create a kernel to demonstrate thread divergence:

```cuda
__global__ void divergenceExample(int *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        // This causes thread divergence within a warp
        if (tid % 2 == 0) {
            // Even threads do this
            data[tid] = tid * 2;
        } else {
            // Odd threads do this
            data[tid] = tid * 3;
        }
    }
}
```

**Performance Impact:**
- Threads in the same warp that take different paths reduce efficiency
- GPU must execute both paths serially
- Avoid divergence when possible by restructuring algorithms

### Memory Access Patterns

Efficient memory access is crucial for performance:

```cuda
// Coalesced access (efficient)
__global__ void coalescedAccess(float *data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[tid] = tid;  // Each thread accesses consecutive memory
}

// Strided access (inefficient)
__global__ void stridedAccess(float *data, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[tid * stride] = tid;  // Threads access memory with gaps
}
```

### Hands-On Exercises

1. **Modify vector addition to use different block sizes (64, 128, 512)**
2. **Create a 2D matrix multiplication kernel**
3. **Implement element-wise operations (sin, cos, exp) on large arrays**
4. **Compare performance with CPU implementations**

## 1.7 Troubleshooting and Optimization Guide

### Common Compilation Errors

**Error: `nvcc not found`**
```bash
# Fix: Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Error: `cuda.h: No such file`**
```bash
# Fix: Install CUDA toolkit properly
sudo apt install nvidia-cuda-toolkit
# Or specify include path
nvcc -I/usr/local/cuda/include program.cu
```

**Error: `undefined reference to cudaMalloc`**
```bash
# Fix: Link CUDA runtime library
nvcc -lcudart program.cu
```

### Common Runtime Errors

**1. "CUDA error: invalid device"**
```cuda
// Check device availability
int deviceCount;
cudaGetDeviceCount(&deviceCount);
if (deviceCount == 0) {
    printf("No CUDA-capable devices found\n");
    return -1;
}

// Set specific device
cudaSetDevice(0);
```

**2. "CUDA error: out of memory"**
```cuda
// Check available memory
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
printf("GPU memory: %zu MB free of %zu MB total\n", 
       free_mem/1024/1024, total_mem/1024/1024);

// Use smaller data sizes or process in chunks
const int MAX_SIZE = free_mem / sizeof(float) / 4;  // Reserve 25% buffer
```

**3. "CUDA error: invalid configuration"**
```cuda
// Check device limits
cudaDeviceProp props;
cudaGetDeviceProperties(&props, 0);
printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
printf("Max block dimensions: (%d, %d, %d)\n", 
       props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
printf("Max grid dimensions: (%d, %d, %d)\n",
       props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
```

### Debug Tools and Techniques

**1. Error Checking Macro**
```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Use consistently
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
```

**2. Kernel Launch Error Checking**
```cuda
kernel<<<grid, block>>>(args);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}
CUDA_CHECK(cudaDeviceSynchronize());
```

**3. Debug Prints in Kernels**
```cuda
__global__ void debugKernel(float *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Only print from a few threads to avoid spam
    if (tid < 5) {
        printf("Thread %d: processing data[%d] = %f\n", tid, tid, data[tid]);
    }
}
```

### Performance Optimization Strategies

**1. Memory Bandwidth Optimization**
```cuda
// Use appropriate data types
float4 vec = make_float4(1.0f, 2.0f, 3.0f, 4.0f);  // Vectorized load

// Coalesced memory access
__global__ void efficientAccess(float *data, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Good: consecutive access
    data[tid] = tid;
    
    // Bad: strided access
    // data[tid * stride] = tid;
}
```

**2. Occupancy Optimization**
```cuda
// Check theoretical occupancy
int maxActiveBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, kernel, blockSize, 0);

// Get suggested block size
int minGridSize, optimalBlockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, kernel, 0, 0);
printf("Optimal block size: %d\n", optimalBlockSize);
```

**3. Shared Memory Usage**
```cuda
__global__ void sharedMemoryExample(float *data, int n) {
    // Allocate shared memory
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load data to shared memory
    if (gid < n) {
        shared_data[tid] = data[gid];
    }
    __syncthreads();  // Synchronize within block
    
    // Process data in shared memory (faster)
    if (gid < n) {
        data[gid] = shared_data[tid] * 2.0f;
    }
}
```

### Profiling Tools

**NVIDIA Tools:**
```bash
# Legacy profiler (deprecated but still useful)
nvprof ./your_program

# Modern profiler
ncu --metrics gpu__time_duration.avg ./your_program

# System-wide profiling
nsys profile --trace=cuda,nvtx ./your_program
```

**AMD ROCm Tools:**
```bash
# Profile HIP applications
rocprof --hip-trace ./your_program

# Detailed metrics
rocprof --stats ./your_program

# System profiling
roctx-trace ./your_program
```

### Performance Comparison Template

```cuda
#include <chrono>
#include <iostream>

class GpuTimer {
    cudaEvent_t start, stop;
    float elapsedTime;
public:
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    void startTimer() {
        cudaEventRecord(start, 0);
    }
    
    void stopTimer() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
    }
    
    float getElapsedMs() { return elapsedTime; }
    
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

// Usage example
GpuTimer timer;
timer.startTimer();
kernel<<<grid, block>>>(args);
timer.stopTimer();
printf("Kernel execution time: %.3f ms\n", timer.getElapsedMs());
```

### Optimization Checklist

✅ **Memory Access:**
- Use coalesced memory access patterns
- Minimize CPU-GPU data transfers
- Consider using pinned memory for large transfers

✅ **Thread Organization:**
- Choose appropriate block sizes (multiples of 32)
- Maximize occupancy
- Minimize thread divergence

✅ **Algorithm Design:**
- Use shared memory for frequently accessed data
- Consider memory bank conflicts
- Implement efficient reduction patterns

✅ **Platform-Specific:**
- **H100**: Utilize Tensor Cores for AI workloads
- **MI300X**: Leverage unified memory architecture
- Use vendor-optimized libraries (cuBLAS, rocBLAS)

### Practice Exercises

1. **Debug a broken kernel with out-of-bounds access**
2. **Optimize memory bandwidth using vectorized loads**  
3. **Profile a kernel and identify bottlenecks**
4. **Implement shared memory optimization for matrix multiplication**
5. **Compare performance across different block sizes**

## Summary

This module covered the fundamentals of GPU programming:

- **Architecture**: Understanding SIMT execution and memory hierarchy
- **Programming Models**: CUDA for NVIDIA, HIP for cross-platform development
- **Thread Management**: Organizing parallel work efficiently
- **Memory Management**: Allocating and transferring data between CPU and GPU
- **Optimization**: Basic techniques for improving performance
- **Debugging**: Tools and techniques for troubleshooting GPU programs

**Next Steps:**
- Practice with the provided examples
- Experiment with different problem sizes and block configurations
- Learn about advanced memory patterns in Module 2
- Explore GPU-accelerated libraries for your domain
