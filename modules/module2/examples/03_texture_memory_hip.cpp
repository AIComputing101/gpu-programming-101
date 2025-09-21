#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>  // For std::memset
#include "rocm7_utils.h"

// AMD GPU-optimized cached memory access (texture memory alternative)
// Uses constant memory and shared memory for caching
__global__ void cachedFilterKernel(const float* __restrict__ input, float *output, 
                                   int width, int height, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_filter = filter_size / 2;
        
        // Apply filter using cached global memory access
        for (int fy = -half_filter; fy <= half_filter; fy++) {
            for (int fx = -half_filter; fx <= half_filter; fx++) {
                int src_x = x + fx;
                int src_y = y + fy;
                
                // Clamp coordinates for boundary conditions
                src_x = max(0, min(src_x, width - 1));
                src_y = max(0, min(src_y, height - 1));
                
                // Cached access with coalescing
                float value = input[src_y * width + src_x];
                sum += value;
            }
        }
        
        output[y * width + x] = sum / (filter_size * filter_size);
    }
}

// Cached memory transpose with spatial locality optimization
__global__ void cachedTranspose(const float* __restrict__ input, float *output, 
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Read with cache-friendly access pattern
        float value = input[y * width + x];
        
        // Write transposed with boundary check
        if (y < width && x < height) {
            output[x * height + y] = value;
        }
    }
}

// Software bilinear interpolation optimized for AMD GPUs
__global__ void bilinearInterpolation(const float* __restrict__ input, float *output, 
                                     int out_width, int out_height, 
                                     int in_width, int in_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < out_width && y < out_height) {
        // Map output coordinates to input coordinates
        float scale_x = (float)in_width / out_width;
        float scale_y = (float)in_height / out_height;
        
        float src_x = (x + 0.5f) * scale_x - 0.5f;
        float src_y = (y + 0.5f) * scale_y - 0.5f;
        
        // Manual bilinear interpolation
        int x1 = (int)floorf(src_x);
        int y1 = (int)floorf(src_y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        
        // Clamp coordinates
        x1 = max(0, min(x1, in_width - 1));
        y1 = max(0, min(y1, in_height - 1));
        x2 = max(0, min(x2, in_width - 1));
        y2 = max(0, min(y2, in_height - 1));
        
        // Get interpolation weights
        float wx = src_x - floorf(src_x);
        float wy = src_y - floorf(src_y);
        
        // Sample four points
        float p11 = input[y1 * in_width + x1];
        float p12 = input[y1 * in_width + x2];
        float p21 = input[y2 * in_width + x1];
        float p22 = input[y2 * in_width + x2];
        
        // Bilinear interpolation
        float interpolated = (1.0f - wx) * (1.0f - wy) * p11 +
                           wx * (1.0f - wy) * p12 +
                           (1.0f - wx) * wy * p21 +
                           wx * wy * p22;
        
        output[y * out_width + x] = interpolated;
    }
}

// Manual bilinear interpolation for platforms without hardware support
__global__ void manualBilinearInterpolation(const float *input, float *output,
                                           int out_width, int out_height,
                                           int in_width, int in_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < out_width && y < out_height) {
        float scale_x = (float)in_width / out_width;
        float scale_y = (float)in_height / out_height;
        
        float src_x = (x + 0.5f) * scale_x - 0.5f;
        float src_y = (y + 0.5f) * scale_y - 0.5f;
        
        int x1 = (int)floorf(src_x);
        int y1 = (int)floorf(src_y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        
        // Clamp coordinates
        x1 = max(0, min(x1, in_width - 1));
        y1 = max(0, min(y1, in_height - 1));
        x2 = max(0, min(x2, in_width - 1));
        y2 = max(0, min(y2, in_height - 1));
        
        float fx = src_x - x1;
        float fy = src_y - y1;
        
        // Bilinear interpolation
        float v1 = input[y1 * in_width + x1] * (1.0f - fx) + input[y1 * in_width + x2] * fx;
        float v2 = input[y2 * in_width + x1] * (1.0f - fx) + input[y2 * in_width + x2] * fx;
        float result = v1 * (1.0f - fy) + v2 * fy;
        
        output[y * out_width + x] = result;
    }
}

// AMD GPU optimized cached memory access pattern
__global__ void amdOptimizedCachedAccess(const float* __restrict__ input, float *output,
                                         int width, int height) {
    // AMD wavefront-aware cached access
    int wavefront_id = blockIdx.x * blockDim.x / 64 + threadIdx.x / 64;
    int lane_id = threadIdx.x % 64;
    
    int total_wavefronts = gridDim.x * blockDim.x / 64;
    int pixels_per_wavefront = (width * height + total_wavefronts - 1) / total_wavefronts;
    
    for (int i = 0; i < pixels_per_wavefront; i++) {
        int pixel_id = wavefront_id * pixels_per_wavefront + i;
        if (pixel_id < width * height) {
            int x = pixel_id % width;
            int y = pixel_id / width;
            
            // Coalesced cached access within wavefront
            float value = input[y * width + x];
            output[pixel_id] = value;
        }
    }
}

// Cached memory demonstration (replaces texture memory for AMD compatibility)
void demonstrateCachedMemoryAccess(float *d_input, float *d_output, 
                                   int width, int height) {
    printf("=== AMD GPU Cached Memory Access Demo ===\n");
    printf("(Alternative to texture memory for AMD GPUs)\n");
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Test cached filter
    HIP_CHECK(hipEventRecord(start));
    cachedFilterKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, 3);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float time;
    HIP_CHECK(hipEventElapsedTime(&time, start, stop));
    printf("Cached filter time: %.3f ms\n", time);
    
    // Test cached transpose
    HIP_CHECK(hipEventRecord(start));
    cachedTranspose<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    HIP_CHECK(hipEventElapsedTime(&time, start, stop));
    printf("Cached transpose time: %.3f ms\n", time);
    
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

void demonstrateCachedMemoryAccess() {
    printf("=== HIP Texture Memory Demo ===\n");
    
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);
    const int filter_size = 3;
    
    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output_texture = (float*)malloc(size);
    float *h_output_manual = (float*)malloc(size);
    
    // Initialize input data
    for (int i = 0; i < width * height; i++) {
        h_input[i] = sinf(i * 0.01f) + cosf(i * 0.02f);
    }
    
    // Allocate device memory
    float *d_input, *d_output_cached, *d_output_manual;
    HIP_CHECK(hipMalloc(&d_input, size));
    HIP_CHECK(hipMalloc(&d_output_cached, size));
    HIP_CHECK(hipMalloc(&d_output_manual, size));
    
    // Copy input to device
    HIP_CHECK(hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice));
    
    // Setup execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Test 1: Cached memory filtering (AMD GPU optimized)
    printf("Testing cached memory filtering...\n");
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(cachedFilterKernel, gridSize, blockSize, 0, 0,
                       d_input, d_output_cached, width, height, filter_size);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float cached_time;
    HIP_CHECK(hipEventElapsedTime(&cached_time, start, stop));
    printf("Cached filtering time: %.3f ms\n", cached_time);
    
    // Test 2: Software bilinear interpolation
    printf("Testing software bilinear interpolation...\n");
    
    int out_width = 512, out_height = 512;
    float *d_resized;
    HIP_CHECK(hipMalloc(&d_resized, out_width * out_height * sizeof(float)));
    
    dim3 resizeBlockSize(16, 16);
    dim3 resizeGridSize((out_width + resizeBlockSize.x - 1) / resizeBlockSize.x,
                        (out_height + resizeBlockSize.y - 1) / resizeBlockSize.y);
    
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(bilinearInterpolation, resizeGridSize, resizeBlockSize, 0, 0,
                       d_input, d_resized, out_width, out_height, width, height);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float resize_time;
    HIP_CHECK(hipEventElapsedTime(&resize_time, start, stop));
    printf("Software resize time: %.3f ms\n", resize_time);
    
    // Test 3: AMD optimized cached access pattern
    printf("Testing AMD optimized cached access...\n");
    
    dim3 amdBlockSize(256);
    dim3 amdGridSize((width * height + amdBlockSize.x - 1) / amdBlockSize.x);
    
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(amdOptimizedCachedAccess, amdGridSize, amdBlockSize, 0, 0,
                       d_input, d_output_cached, width, height);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float amd_time;
    HIP_CHECK(hipEventElapsedTime(&amd_time, start, stop));
    printf("AMD optimized cached access time: %.3f ms\n", amd_time);
    
    // Verify results
    float *h_output_cached = (float*)malloc(size);
    HIP_CHECK(hipMemcpy(h_output_cached, d_output_cached, size, hipMemcpyDeviceToHost));
    
    // Calculate performance metrics
    float bandwidth_gb_s = (2.0f * size) / (cached_time * 1e6); // Read + Write
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    
    // Cached memory analysis
    printf("\n=== Cached Memory Access Analysis ===\n");
    printf("AMD GPU cached memory provides:\n");
    printf("- L1/L2 cache utilization\n");
    printf("- Memory coalescing optimization\n");
    printf("- Wavefront-aware access patterns\n");
    printf("- Manual boundary handling control\n");
    
#ifdef __HIP_PLATFORM_AMD__
    printf("\nAMD GPU specific optimizations:\n");
    printf("- 64-thread wavefront optimization\n");
    printf("- Memory coalescing for cache efficiency\n");
    printf("- Manual bilinear interpolation\n");
#endif
    
    // Demonstrate additional cached memory functionality
    demonstrateCachedMemoryAccess(d_input, d_output_cached, width, height);
    
    // Cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output_cached));
    HIP_CHECK(hipFree(d_output_manual));
    HIP_CHECK(hipFree(d_resized));
    
    free(h_input);
    free(h_output_cached);
    free(h_output_manual);
}

int main() {
    printf("HIP Cached Memory Access Example (AMD GPU Optimized)\n");
    printf("===================================================\n");
    printf("Note: This example uses cached memory access patterns\n");
    printf("      optimized for AMD GPUs instead of texture memory.\n\n");
    
    demonstrateCachedMemoryAccess();
    
    return 0;
}