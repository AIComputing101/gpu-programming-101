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
    
    printf("Launching kernel with %d blocks of %d threads each\n", gridSize, blockSize);
    
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