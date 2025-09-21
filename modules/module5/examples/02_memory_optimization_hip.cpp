/**
 * Module 5: Performance Engineering and Optimization - Memory Optimization (HIP)
 * 
 * Comprehensive memory optimization techniques for AMD GPU architectures using ROCm/HIP.
 * This example demonstrates LDS optimization, memory coalescing patterns, and AMD-specific
 * memory hierarchy utilization strategies.
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>

#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

constexpr int WAVEFRONT_SIZE = 64;

// Memory coalescing demonstration
__global__ void memory_coalescing_test_good(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // Coalesced access
    }
}

__global__ void memory_coalescing_test_bad(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int strided_idx = (idx * 32) % n;  // Non-coalesced, strided access
        output[idx] = input[strided_idx] * 2.0f;
    }
}

// LDS optimization for matrix transpose
__global__ void matrix_transpose_lds_optimized(float* input, float* output, int rows, int cols) {
    __shared__ float lds_tile[32][32 + 1];  // +1 to avoid LDS bank conflicts
    
    int block_x = blockIdx.x * 32;
    int block_y = blockIdx.y * 32;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    
    // Load tile into LDS with coalesced reads
    if (block_x + thread_x < cols && block_y + thread_y < rows) {
        lds_tile[thread_y][thread_x] = input[(block_y + thread_y) * cols + (block_x + thread_x)];
    }
    
    __syncthreads();
    
    // Write transposed tile with coalesced writes
    if (block_y + thread_x < rows && block_x + thread_y < cols) {
        output[(block_x + thread_y) * rows + (block_y + thread_x)] = lds_tile[thread_x][thread_y];
    }
}

// Memory bandwidth optimization test
__global__ void bandwidth_test_kernel(float4* input, float4* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 data = input[idx];
        data.x *= 2.0f;
        data.y *= 2.0f;
        data.z *= 2.0f;
        data.w *= 2.0f;
        output[idx] = data;
    }
}

class PerformanceTimer {
private:
    hipEvent_t start_event, stop_event;
    
public:
    PerformanceTimer() {
        HIP_CHECK(hipEventCreate(&start_event));
        HIP_CHECK(hipEventCreate(&stop_event));
    }
    
    ~PerformanceTimer() {
        HIP_CHECK(hipEventDestroy(start_event));
        HIP_CHECK(hipEventDestroy(stop_event));
    }
    
    void start() {
        HIP_CHECK(hipEventRecord(start_event));
    }
    
    float stop() {
        HIP_CHECK(hipEventRecord(stop_event));
        HIP_CHECK(hipEventSynchronize(stop_event));
        
        float elapsed_ms;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start_event, stop_event));
        return elapsed_ms;
    }
};

void test_memory_coalescing() {
    std::cout << "\n=== Memory Coalescing Test ===\n";
    
    const int n = 1024 * 1024;
    const int bytes = n * sizeof(float);
    
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    
    // Initialize with random data
    std::vector<float> h_input(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::generate(h_input.begin(), h_input.end(), [&]() { return dis(gen); });
    
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), bytes, hipMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    PerformanceTimer timer;
    
    // Test coalesced access
    timer.start();
    hipLaunchKernelGGL(memory_coalescing_test_good, grid, block, 0, 0, d_input, d_output, n);
    HIP_CHECK(hipDeviceSynchronize());
    float coalesced_time = timer.stop();
    
    // Test non-coalesced access
    timer.start();
    hipLaunchKernelGGL(memory_coalescing_test_bad, grid, block, 0, 0, d_input, d_output, n);
    HIP_CHECK(hipDeviceSynchronize());
    float non_coalesced_time = timer.stop();
    
    double coalesced_bandwidth = (2.0 * bytes) / (coalesced_time * 1e6);
    double non_coalesced_bandwidth = (2.0 * bytes) / (non_coalesced_time * 1e6);
    
    std::cout << "Coalesced access:     " << std::fixed << std::setprecision(3) << coalesced_time << " ms"
              << " (Bandwidth: " << std::setprecision(1) << coalesced_bandwidth << " GB/s)\n";
    std::cout << "Non-coalesced access: " << std::fixed << std::setprecision(3) << non_coalesced_time << " ms"
              << " (Bandwidth: " << std::setprecision(1) << non_coalesced_bandwidth << " GB/s)\n";
    std::cout << "Performance ratio: " << std::setprecision(2) << non_coalesced_time / coalesced_time << "x\n";
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

void test_matrix_transpose() {
    std::cout << "\n=== Matrix Transpose LDS Optimization ===\n";
    
    const int rows = 2048;
    const int cols = 2048;
    const size_t bytes = rows * cols * sizeof(float);
    
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    
    // Initialize input matrix
    std::vector<float> h_input(rows * cols);
    std::iota(h_input.begin(), h_input.end(), 0);
    
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), bytes, hipMemcpyHostToDevice));
    
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    PerformanceTimer timer;
    timer.start();
    hipLaunchKernelGGL(matrix_transpose_lds_optimized, grid, block, 0, 0, d_input, d_output, rows, cols);
    HIP_CHECK(hipDeviceSynchronize());
    float transpose_time = timer.stop();
    
    double bandwidth = (2.0 * bytes) / (transpose_time * 1e6);
    
    std::cout << "Matrix transpose (" << rows << "x" << cols << "): "
              << std::fixed << std::setprecision(3) << transpose_time << " ms"
              << " (Bandwidth: " << std::setprecision(1) << bandwidth << " GB/s)\n";
    
    // Verify correctness
    std::vector<float> h_output(rows * cols);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, bytes, hipMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < 10 && correct; ++i) {
        for (int j = 0; j < 10 && correct; ++j) {
            if (h_output[j * rows + i] != h_input[i * cols + j]) {
                correct = false;
            }
        }
    }
    
    std::cout << "Correctness: " << (correct ? "PASS" : "FAIL") << "\n";
    
    hipFree(d_input);
    hipFree(d_output);
}

void test_memory_bandwidth() {
    std::cout << "\n=== Memory Bandwidth Optimization ===\n";
    
    const int n = 1024 * 1024;
    const size_t bytes = n * sizeof(float4);
    
    float4 *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    PerformanceTimer timer;
    timer.start();
    hipLaunchKernelGGL(bandwidth_test_kernel, grid, block, 0, 0, d_input, d_output, n);
    HIP_CHECK(hipDeviceSynchronize());
    float kernel_time = timer.stop();
    
    double bandwidth = (2.0 * bytes) / (kernel_time * 1e6);
    
    std::cout << "Vectorized memory access (float4): "
              << std::fixed << std::setprecision(3) << kernel_time << " ms"
              << " (Bandwidth: " << std::setprecision(1) << bandwidth << " GB/s)\n";
    
    hipFree(d_input);
    hipFree(d_output);
}

int main() {
    std::cout << "HIP Memory Optimization Techniques\n";
    std::cout << "==================================\n";
    
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device);
    
    std::cout << "GPU: " << props.name << "\n";
    std::cout << "Memory: " << props.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "Memory Clock: " << props.memoryClockRate / 1000 << " MHz\n";
    std::cout << "Memory Bus Width: " << props.memoryBusWidth << " bits\n";
    
    // Calculate theoretical bandwidth
    double theoretical_bandwidth = 2.0 * (props.memoryClockRate * 1000.0) * (props.memoryBusWidth / 8.0) / 1e9;
    std::cout << "Theoretical Bandwidth: " << std::fixed << std::setprecision(1) << theoretical_bandwidth << " GB/s\n";
    
    try {
        test_memory_coalescing();
        test_matrix_transpose();
        test_memory_bandwidth();
        
        std::cout << "\n=== Memory Optimization Summary ===\n";
        std::cout << "1. Memory coalescing is critical for bandwidth utilization\n";
        std::cout << "2. LDS can effectively reduce global memory traffic\n";
        std::cout << "3. Vectorized memory access improves bandwidth efficiency\n";
        std::cout << "4. AMD GPUs benefit from LDS bank conflict avoidance\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}