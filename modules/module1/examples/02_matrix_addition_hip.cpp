#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matrixAdd(float *A, float *B, float *C, int width, int height) {
    // 2D thread indexing using HIP built-ins
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    
    // Boundary check
    if (row < height && col < width) {
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

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
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);
    
    // Get device information
    int device;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    printf("Running on: %s\n", props.name);
    printf("Compute capability: %d.%d\n", props.major, props.minor);
    
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
    HIP_CHECK(hipMalloc(&d_A, size));
    HIP_CHECK(hipMalloc(&d_B, size));
    HIP_CHECK(hipMalloc(&d_C, size));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));
    
    // Define block and grid sizes
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    printf("Matrix dimensions: %d x %d\n", width, height);
    printf("Grid size: (%d, %d)\n", gridSize.x, gridSize.y);
    printf("Block size: (%d, %d)\n", blockSize.x, blockSize.y);
    printf("Total threads: %d\n", gridSize.x * gridSize.y * blockSize.x * blockSize.y);
    
    // Launch kernel using HIP syntax
    matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    
    // Alternative HIP launch syntax (legacy)
    // hipLaunchKernelGGL(matrixAdd, gridSize, blockSize, 0, 0, d_A, d_B, d_C, width, height);
    
    // Check for errors
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy result back
    HIP_CHECK(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost));
    
    // Verify result
    bool success = true;
    for (int i = 0; i < width * height; i++) {
        if (h_C[i] != 3.0f) {
            success = false;
            printf("Error at element %d: expected 3.0, got %f\n", i, h_C[i]);
            break;
        }
    }
    
    printf("Matrix addition %s\n", success ? "PASSED" : "FAILED");
    
    // Show memory usage
    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    printf("GPU memory usage: %.2f MB used of %.2f MB total\n", 
           (total_mem - free_mem) / (1024.0 * 1024.0), 
           total_mem / (1024.0 * 1024.0));
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    
    return 0;
}