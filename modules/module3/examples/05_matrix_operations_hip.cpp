#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "rocm7_utils.h"

// Try to include ROCBlas if available
#ifdef __has_include
    #if __has_include(<rocblas/rocblas.h>)
        #include <rocblas/rocblas.h>
        #define HAS_ROCBLAS 1
    #else
        #define HAS_ROCBLAS 0
    #endif
#else
    #define HAS_ROCBLAS 0
#endif

#define TILE_SIZE 16
#define BLOCK_SIZE 256

// Matrix multiplication with shared memory tiling
__global__ void matrixMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;
    
    float Cvalue = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (Row < N && t * TILE_SIZE + tx < N)
            tileA[ty][tx] = A[Row * N + t * TILE_SIZE + tx];
        else
            tileA[ty][tx] = 0.0f;
            
        if (t * TILE_SIZE + ty < N && Col < N)
            tileB[ty][tx] = B[(t * TILE_SIZE + ty) * N + Col];
        else
            tileB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (Row < N && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

// AMD-optimized matrix multiplication with LDS optimization
__global__ void matrixMulAMDOptimized(float *A, float *B, float *C, int N) {
    // AMD GPU optimization: Use larger tiles for better LDS utilization
    __shared__ float tileA[32][32 + 1]; // +1 to avoid LDS bank conflicts
    __shared__ float tileB[32][32 + 1];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int Row = by * 32 + ty;
    int Col = bx * 32 + tx;
    
    float Cvalue = 0.0f;
    
    for (int t = 0; t < (N + 31) / 32; ++t) {
        // Coalesced loading optimized for AMD memory hierarchy
        if (Row < N && t * 32 + tx < N)
            tileA[ty][tx] = A[Row * N + t * 32 + tx];
        else
            tileA[ty][tx] = 0.0f;
            
        if (t * 32 + ty < N && Col < N)
            tileB[ty][tx] = B[(t * 32 + ty) * N + Col];
        else
            tileB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute with loop unrolling for AMD architecture
        #pragma unroll 8
        for (int k = 0; k < 32; ++k) {
            Cvalue += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (Row < N && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

// Matrix transpose with shared memory to avoid bank conflicts
__global__ void transposeSharedMem(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    
    int xIndex = blockIdx.x * TILE_SIZE + threadIdx.x;
    int yIndex = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load data into shared memory
    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = input[yIndex * width + xIndex];
    }
    
    __syncthreads();
    
    // Write transposed data to output
    xIndex = blockIdx.y * TILE_SIZE + threadIdx.x;
    yIndex = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (xIndex < height && yIndex < width) {
        output[yIndex * height + xIndex] = tile[threadIdx.x][threadIdx.y];
    }
}

// AMD-optimized transpose with LDS bank conflict avoidance
__global__ void transposeAMDOptimized(float *input, float *output, int width, int height) {
    // AMD LDS has 32 banks, optimize accordingly
    __shared__ float tile[32][32 + 1];
    
    int xIndex = blockIdx.x * 32 + threadIdx.x;
    int yIndex = blockIdx.y * 32 + threadIdx.y;
    
    // Coalesced load optimized for AMD 64-byte cache lines
    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = input[yIndex * width + xIndex];
    }
    
    __syncthreads();
    
    // Coalesced write with proper indexing
    xIndex = blockIdx.y * 32 + threadIdx.x;
    yIndex = blockIdx.x * 32 + threadIdx.y;
    
    if (xIndex < height && yIndex < width) {
        output[yIndex * height + xIndex] = tile[threadIdx.x][threadIdx.y];
    }
}

// Matrix-vector multiplication using wavefront-level operations
__global__ void matrixVectorMul(float *matrix, float *vector, float *result, int N) {
    __shared__ float shared_vector[BLOCK_SIZE];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row < N) {
        // Load vector into shared memory
        if (tid < N) {
            shared_vector[tid] = vector[tid];
        } else {
            shared_vector[tid] = 0.0f;
        }
        __syncthreads();
        
        float sum = 0.0f;
        
        // Each thread computes multiple elements
        for (int col = tid; col < N; col += blockDim.x) {
            sum += matrix[row * N + col] * shared_vector[col];
        }
        
        // Reduce within block using AMD wavefront primitives
        __shared__ float sdata[BLOCK_SIZE];
        sdata[tid] = sum;
        __syncthreads();
        
        // AMD-optimized reduction
        for (int s = blockDim.x / 2; s > 32; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        // Final wavefront reduction
        if (tid < 32) {
            volatile float* vdata = sdata;
            vdata[tid] += vdata[tid + 32];
            vdata[tid] += vdata[tid + 16];
            vdata[tid] += vdata[tid + 8];
            vdata[tid] += vdata[tid + 4];
            vdata[tid] += vdata[tid + 2];
            vdata[tid] += vdata[tid + 1];
        }
        
        if (tid == 0) {
            result[row] = sdata[0];
        }
    }
}

// AMD wavefront-aware matrix-vector multiplication
__global__ void matrixVectorMulWavefront(float *matrix, float *vector, float *result, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 64; // AMD wavefront size
    int warp_id = tid / 64;
    
    if (row < N) {
        float sum = 0.0f;
        
        // Each thread processes multiple elements
        for (int col = tid; col < N; col += blockDim.x) {
            sum += matrix[row * N + col] * vector[col];
        }
        
        // Wavefront reduction using shuffle operations
        #pragma unroll
        for (int offset = 32; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset);
        }
        
        // Store wavefront results
        __shared__ float warp_sums[4]; // Support up to 256 threads (4 wavefronts)
        if (lane_id == 0) {
            warp_sums[warp_id] = sum;
        }
        
        __syncthreads();
        
        // Final reduction of wavefront results
        if (warp_id == 0) {
            sum = (lane_id < (blockDim.x / 64)) ? warp_sums[lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 32; offset > 0; offset /= 2) {
                sum += __shfl_down(sum, offset);
            }
            
            if (lane_id == 0) {
                result[row] = sum;
            }
        }
    }
}

// LU decomposition kernel
__global__ void luDecomposition(float *matrix, int N, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    int col = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
    
    if (row < N && col < N) {
        if (col == k + 1) {
            // L matrix update
            matrix[row * N + col] = matrix[row * N + col] / matrix[k * N + k];
        } else if (row == k + 1) {
            // U matrix already computed
        } else {
            // Update remaining elements
            matrix[row * N + col] -= matrix[row * N + k] * matrix[k * N + col];
        }
    }
}

// Strassen matrix multiplication (recursive approach for large matrices)
__global__ void strassenMatrixMul(float *A, float *B, float *C, int N, int level) {
    // Simplified Strassen implementation for educational purposes
    // In practice, would need recursive kernel launches or iterative approach
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N && level == 0) {
        // Base case: regular matrix multiplication
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Matrix multiplication demonstration

class MatrixOperations {
private:
#if HAS_ROCBLAS
    rocblas_handle handle;
#endif
    
public:
    MatrixOperations() {
#if HAS_ROCBLAS
        rocblas_create_handle(&handle);
#endif
    }
    
    ~MatrixOperations() {
#if HAS_ROCBLAS
        rocblas_destroy_handle(handle);
#endif
    }
    
    void testMatrixMultiplication() {
        printf("=== Matrix Multiplication Performance Test ===\n");
        
        const int N = 1024;
        const size_t size = N * N * sizeof(float);
        
        // Allocate host memory
        float *h_A = (float*)malloc(size);
        float *h_B = (float*)malloc(size);
        float *h_C_custom = (float*)malloc(size);
#if HAS_ROCBLAS
        float *h_C_rocblas = (float*)malloc(size);
#endif
        
        // Initialize matrices
        for (int i = 0; i < N * N; i++) {
            h_A[i] = (float)(rand() % 100) / 100.0f;
            h_B[i] = (float)(rand() % 100) / 100.0f;
        }
        
        // Allocate device memory
        float *d_A, *d_B, *d_C_custom;
        HIP_CHECK(hipMalloc(&d_A, size));
        HIP_CHECK(hipMalloc(&d_B, size));
        HIP_CHECK(hipMalloc(&d_C_custom, size));
#if HAS_ROCBLAS
        float *d_C_rocblas;
        HIP_CHECK(hipMalloc(&d_C_rocblas, size));
#endif
        
        // Copy data to device
        HIP_CHECK(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));
        
        // Test custom implementation
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
        
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        HIP_CHECK(hipEventRecord(start));
        matrixMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C_custom, N);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float custom_time;
        HIP_CHECK(hipEventElapsedTime(&custom_time, start, stop));
        
        // Test AMD optimized implementation
        dim3 amdBlockSize(32, 32);
        dim3 amdGridSize((N + 31) / 32, (N + 31) / 32);
        
        HIP_CHECK(hipEventRecord(start));
        matrixMulAMDOptimized<<<amdGridSize, amdBlockSize>>>(d_A, d_B, d_C_custom, N);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float amd_time;
        HIP_CHECK(hipEventElapsedTime(&amd_time, start, stop));
        
#if HAS_ROCBLAS
        // Test rocBLAS implementation
        const float alpha = 1.0f, beta = 0.0f;
        
        HIP_CHECK(hipEventRecord(start));
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                     N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C_rocblas, N);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float rocblas_time;
        HIP_CHECK(hipEventElapsedTime(&rocblas_time, start, stop));
#endif
        
        // Performance analysis
        double flops = 2.0 * N * N * N; // Multiply-add operations
        
        printf("Matrix size: %dx%d\n", N, N);
        printf("Custom tiled GEMM:     %8.3f ms (%8.2f GFLOPS)\n", 
               custom_time, flops / (custom_time * 1e6));
        printf("AMD optimized GEMM:    %8.3f ms (%8.2f GFLOPS)\n", 
               amd_time, flops / (amd_time * 1e6));
#if HAS_ROCBLAS
        printf("rocBLAS GEMM:          %8.3f ms (%8.2f GFLOPS)\n", 
               rocblas_time, flops / (rocblas_time * 1e6));
#else
        printf("rocBLAS GEMM:          Not available (rocBLAS not found)\n");
#endif
        
        // Verify correctness
        HIP_CHECK(hipMemcpy(h_C_custom, d_C_custom, size, hipMemcpyDeviceToHost));
#if HAS_ROCBLAS
        HIP_CHECK(hipMemcpy(h_C_rocblas, d_C_rocblas, size, hipMemcpyDeviceToHost));
        
        double max_error = 0.0;
        for (int i = 0; i < N * N; i++) {
            double error = fabs(h_C_custom[i] - h_C_rocblas[i]);
            max_error = fmax(max_error, error);
        }
        printf("Max error vs rocBLAS: %e\n", max_error);
#else
        printf("Correctness verification: rocBLAS not available\n");
#endif
        
        // Cleanup
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        
        free(h_A); free(h_B); free(h_C_custom);
        HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C_custom));
#if HAS_ROCBLAS
        free(h_C_rocblas);
        HIP_CHECK(hipFree(d_C_rocblas));
#endif
    }
    
    void testMatrixTranspose() {
        printf("\n=== Matrix Transpose Performance Test ===\n");
        
        const int width = 2048, height = 2048;
        const size_t size = width * height * sizeof(float);
        
        // Allocate memory
        float *h_input = (float*)malloc(size);
        float *h_output = (float*)malloc(size);
        float *d_input, *d_output;
        
        HIP_CHECK(hipMalloc(&d_input, size));
        HIP_CHECK(hipMalloc(&d_output, size));
        
        // Initialize input
        for (int i = 0; i < width * height; i++) {
            h_input[i] = (float)i;
        }
        HIP_CHECK(hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice));
        
        // Test transpose implementations
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, 
                      (height + TILE_SIZE - 1) / TILE_SIZE);
        
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Standard transpose
        HIP_CHECK(hipEventRecord(start));
        transposeSharedMem<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float transpose_time;
        HIP_CHECK(hipEventElapsedTime(&transpose_time, start, stop));
        
        // AMD optimized transpose
        dim3 amdBlockSize(32, 32);
        dim3 amdGridSize((width + 31) / 32, (height + 31) / 32);
        
        HIP_CHECK(hipEventRecord(start));
        transposeAMDOptimized<<<amdGridSize, amdBlockSize>>>(d_input, d_output, width, height);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float amd_transpose_time;
        HIP_CHECK(hipEventElapsedTime(&amd_transpose_time, start, stop));
        
        // Performance analysis
        double bandwidth = (2.0 * size) / (transpose_time * 1e6); // Read + Write
        double amd_bandwidth = (2.0 * size) / (amd_transpose_time * 1e6);
        
        printf("Matrix size: %dx%d\n", width, height);
        printf("Standard transpose:  %8.3f ms (%8.2f GB/s)\n", 
               transpose_time, bandwidth);
        printf("AMD optimized:       %8.3f ms (%8.2f GB/s)\n", 
               amd_transpose_time, amd_bandwidth);
        
        // Cleanup
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        free(h_input); free(h_output);
        HIP_CHECK(hipFree(d_input)); HIP_CHECK(hipFree(d_output));
    }
    
    void testMatrixVectorMultiplication() {
        printf("\n=== Matrix-Vector Multiplication Test ===\n");
        
        const int N = 4096;
        const size_t matrix_size = N * N * sizeof(float);
        const size_t vector_size = N * sizeof(float);
        
        // Allocate memory
        float *h_matrix = (float*)malloc(matrix_size);
        float *h_vector = (float*)malloc(vector_size);
        float *h_result = (float*)malloc(vector_size);
        
        float *d_matrix, *d_vector, *d_result;
        HIP_CHECK(hipMalloc(&d_matrix, matrix_size));
        HIP_CHECK(hipMalloc(&d_vector, vector_size));
        HIP_CHECK(hipMalloc(&d_result, vector_size));
        
        // Initialize data
        for (int i = 0; i < N * N; i++) {
            h_matrix[i] = (float)(rand() % 100) / 100.0f;
        }
        for (int i = 0; i < N; i++) {
            h_vector[i] = (float)(rand() % 100) / 100.0f;
        }
        
        HIP_CHECK(hipMemcpy(d_matrix, h_matrix, matrix_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vector, h_vector, vector_size, hipMemcpyHostToDevice));
        
        // Test implementations
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Standard implementation
        HIP_CHECK(hipEventRecord(start));
        matrixVectorMul<<<N, BLOCK_SIZE>>>(d_matrix, d_vector, d_result, N);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float standard_time;
        HIP_CHECK(hipEventElapsedTime(&standard_time, start, stop));
        
        // Wavefront-optimized implementation
        HIP_CHECK(hipEventRecord(start));
        matrixVectorMulWavefront<<<N, BLOCK_SIZE>>>(d_matrix, d_vector, d_result, N);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float wavefront_time;
        HIP_CHECK(hipEventElapsedTime(&wavefront_time, start, stop));
        
        // Performance analysis
        double flops = 2.0 * N * N; // Multiply-add operations
        
        printf("Matrix size: %dx%d\n", N, N);
        printf("Standard implementation:   %8.3f ms (%8.2f GFLOPS)\n", 
               standard_time, flops / (standard_time * 1e6));
        printf("Wavefront optimized:       %8.3f ms (%8.2f GFLOPS)\n", 
               wavefront_time, flops / (wavefront_time * 1e6));
        
        // Cleanup
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        free(h_matrix); free(h_vector); free(h_result);
        HIP_CHECK(hipFree(d_matrix)); HIP_CHECK(hipFree(d_vector)); HIP_CHECK(hipFree(d_result));
    }
};

void demonstrateMatrixOperations() {
    printf("=== HIP Matrix Operations Demo ===\n");
    
    MatrixOperations demo;
    
    demo.testMatrixMultiplication();
    demo.testMatrixTranspose();
    demo.testMatrixVectorMultiplication();
    
    printf("\n=== AMD GPU Matrix Operation Optimizations ===\n");
    printf("- Use 32x32 tiles for better LDS utilization\n");
    printf("- Optimize for 64-thread wavefronts\n");
    printf("- Avoid LDS bank conflicts with proper padding\n");
    printf("- Use wavefront shuffle operations for reductions\n");
    printf("- Leverage rocBLAS for production workloads\n");
    printf("- Consider mixed precision for improved performance\n");
}

int main() {
    printf("HIP Matrix Operations Example\n");
    printf("============================\n");
    
    demonstrateMatrixOperations();
    
    return 0;
}