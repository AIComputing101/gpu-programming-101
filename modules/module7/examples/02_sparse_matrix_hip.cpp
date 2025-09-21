#include <hip/hip_runtime.h>
#include "rocm7_utils.h"  // ROCm 7.0 enhanced utilities

// Conditional rocsparse support - comment out if not available
#ifdef HAS_ROCSPARSE
#include <rocsparse/rocsparse.h>
#include <rocblas/rocblas.h>
#endif
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <memory>

#define CHECK_HIP(call) do { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#ifdef HAS_ROCSPARSE
#define CHECK_ROCSPARSE(call) do { \
    rocsparse_status status = call; \
    if (status != rocsparse_status_success) { \
        std::cerr << "rocSPARSE Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)
#endif

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Return milliseconds
    }
};

struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<float> values;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    
    CSRMatrix(int r, int c, int nz) : rows(r), cols(c), nnz(nz) {
        values.resize(nnz);
        row_ptr.resize(rows + 1);
        col_idx.resize(nnz);
    }
};

CSRMatrix generateRandomSparseMatrix(int rows, int cols, double sparsity) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    
    int target_nnz = static_cast<int>(rows * cols * sparsity);
    CSRMatrix matrix(rows, cols, target_nnz);
    
    int current_nnz = 0;
    matrix.row_ptr[0] = 0;
    
    for (int i = 0; i < rows && current_nnz < target_nnz; ++i) {
        int nnz_this_row = std::min(target_nnz - current_nnz, 
                                   static_cast<int>((target_nnz / rows) * (1.0 + val_dist(gen) * 0.5)));
        
        std::vector<int> cols_in_row;
        for (int j = 0; j < nnz_this_row; ++j) {
            int col;
            do {
                col = col_dist(gen);
            } while (std::find(cols_in_row.begin(), cols_in_row.end(), col) != cols_in_row.end());
            cols_in_row.push_back(col);
        }
        
        std::sort(cols_in_row.begin(), cols_in_row.end());
        
        for (int col : cols_in_row) {
            matrix.values[current_nnz] = val_dist(gen);
            matrix.col_idx[current_nnz] = col;
            current_nnz++;
        }
        
        matrix.row_ptr[i + 1] = current_nnz;
    }
    
    // Fill remaining rows
    for (int i = rows; i >= 0; --i) {
        if (matrix.row_ptr[i] == 0) {
            matrix.row_ptr[i] = current_nnz;
        } else {
            break;
        }
    }
    
    matrix.nnz = current_nnz;
    matrix.values.resize(current_nnz);
    matrix.col_idx.resize(current_nnz);
    
    return matrix;
}

__global__ void spmv_csr_kernel(const float* values, const int* row_ptr, const int* col_idx,
                                const float* x, float* y, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        
        for (int j = start; j < end; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        
        y[row] = sum;
    }
}

__global__ void spmv_csr_wavefront_kernel(const float* values, const int* row_ptr, const int* col_idx,
                                          const float* x, float* y, int rows) {
    int wavefront_id = blockIdx.x * blockDim.x / 64 + threadIdx.x / 64;
    int lane_id = threadIdx.x % 64; // AMD wavefront size is 64
    int total_wavefronts = gridDim.x * blockDim.x / 64;
    
    for (int row = wavefront_id; row < rows; row += total_wavefronts) {
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        
        float sum = 0.0f;
        
        // Process elements in chunks of 64
        for (int j = start + lane_id; j < end; j += 64) {
            sum += values[j] * x[col_idx[j]];
        }
        
        // Wavefront reduction using shuffle operations
        #pragma unroll
        for (int offset = 32; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset);
        }
        
        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

__global__ void spmv_csr_lds_optimized_kernel(const float* values, const int* row_ptr, const int* col_idx,
                                              const float* x, float* y, int rows) {
    __shared__ float lds_data[256]; // Local Data Share for AMD GPUs
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    
    float sum = 0.0f;
    
    // Each thread processes multiple elements
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
        sum += values[j] * x[col_idx[j]];
    }
    
    lds_data[threadIdx.x] = sum;
    __syncthreads();
    
    // Block reduction optimized for AMD architecture
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            lds_data[threadIdx.x] += lds_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Final warp reduction
    if (threadIdx.x < 32) {
        volatile float* vlds = lds_data;
        vlds[threadIdx.x] += vlds[threadIdx.x + 32];
        vlds[threadIdx.x] += vlds[threadIdx.x + 16];
        vlds[threadIdx.x] += vlds[threadIdx.x + 8];
        vlds[threadIdx.x] += vlds[threadIdx.x + 4];
        vlds[threadIdx.x] += vlds[threadIdx.x + 2];
        vlds[threadIdx.x] += vlds[threadIdx.x + 1];
    }
    
    if (threadIdx.x == 0) {
        y[row] = lds_data[0];
    }
}

class SparseMatrixMultiplier {
private:
    rocsparse_handle rocsparse_handle;
    hipStream_t stream;
    
public:
    SparseMatrixMultiplier() {
        CHECK_ROCSPARSE(rocsparse_create_handle(&rocsparse_handle));
        CHECK_HIP(hipStreamCreate(&stream));
        CHECK_ROCSPARSE(rocsparse_set_stream(rocsparse_handle, stream));
    }
    
    ~SparseMatrixMultiplier() {
        rocsparse_destroy_handle(rocsparse_handle);
        hipStreamDestroy(stream);
    }
    
    void spmv_rocsparse(const CSRMatrix& matrix, const float* d_x, float* d_y) {
        const float alpha = 1.0f, beta = 0.0f;
        
        rocsparse_mat_descr descr;
        CHECK_ROCSPARSE(rocsparse_create_mat_descr(&descr));
        CHECK_ROCSPARSE(rocsparse_set_mat_index_base(descr, rocsparse_index_base_zero));
        CHECK_ROCSPARSE(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));
        
        // Allocate device memory for matrix
        float* d_values;
        int* d_row_ptr;
        int* d_col_idx;
        
        CHECK_HIP(hipMalloc(&d_values, matrix.nnz * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_row_ptr, (matrix.rows + 1) * sizeof(int)));
        CHECK_HIP(hipMalloc(&d_col_idx, matrix.nnz * sizeof(int)));
        
        CHECK_HIP(hipMemcpy(d_values, matrix.values.data(), matrix.nnz * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_row_ptr, matrix.row_ptr.data(), (matrix.rows + 1) * sizeof(int), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_col_idx, matrix.col_idx.data(), matrix.nnz * sizeof(int), hipMemcpyHostToDevice));
        
        // Perform SpMV
        CHECK_ROCSPARSE(rocsparse_scsrmv(rocsparse_handle, rocsparse_operation_none,
                                        matrix.rows, matrix.cols, matrix.nnz,
                                        &alpha, descr,
                                        d_values, d_row_ptr, d_col_idx,
                                        nullptr, // info
                                        d_x, &beta, d_y));
        
        CHECK_HIP(hipStreamSynchronize(stream));
        
        // Cleanup
        CHECK_HIP(HIP_CHECK(hipFree(d_values));
        CHECK_HIP(HIP_CHECK(hipFree(d_row_ptr));
        CHECK_HIP(HIP_CHECK(hipFree(d_col_idx));
        rocsparse_destroy_mat_descr(descr);
    }
    
    void spmv_custom(const CSRMatrix& matrix, const float* d_x, float* d_y, int kernel_type = 0) {
        // Allocate device memory for matrix
        float* d_values;
        int* d_row_ptr;
        int* d_col_idx;
        
        CHECK_HIP(hipMalloc(&d_values, matrix.nnz * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_row_ptr, (matrix.rows + 1) * sizeof(int)));
        CHECK_HIP(hipMalloc(&d_col_idx, matrix.nnz * sizeof(int)));
        
        CHECK_HIP(hipMemcpy(d_values, matrix.values.data(), matrix.nnz * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_row_ptr, matrix.row_ptr.data(), (matrix.rows + 1) * sizeof(int), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_col_idx, matrix.col_idx.data(), matrix.nnz * sizeof(int), hipMemcpyHostToDevice));
        
        dim3 block_size(256);
        dim3 grid_size;
        
        switch (kernel_type) {
            case 0: // Thread-per-row
                grid_size = dim3((matrix.rows + block_size.x - 1) / block_size.x);
                hipLaunchKernelGGL(spmv_csr_kernel, grid_size, block_size, 0, stream,
                                  d_values, d_row_ptr, d_col_idx, d_x, d_y, matrix.rows);
                break;
                
            case 1: // Wavefront-per-row (AMD-specific)
                grid_size = dim3((matrix.rows * 64 + block_size.x - 1) / block_size.x);
                hipLaunchKernelGGL(spmv_csr_wavefront_kernel, grid_size, block_size, 0, stream,
                                  d_values, d_row_ptr, d_col_idx, d_x, d_y, matrix.rows);
                break;
                
            case 2: // Block-per-row with LDS optimization
                grid_size = dim3(matrix.rows);
                hipLaunchKernelGGL(spmv_csr_lds_optimized_kernel, grid_size, block_size, 0, stream,
                                  d_values, d_row_ptr, d_col_idx, d_x, d_y, matrix.rows);
                break;
        }
        
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipStreamSynchronize(stream));
        
        // Cleanup
        CHECK_HIP(HIP_CHECK(hipFree(d_values));
        CHECK_HIP(HIP_CHECK(hipFree(d_row_ptr));
        CHECK_HIP(HIP_CHECK(hipFree(d_col_idx));
    }
};

__global__ void sparse_matrix_add_amd_kernel(const float* A_vals, const int* A_row_ptr, const int* A_col_idx,
                                             const float* B_vals, const int* B_row_ptr, const int* B_col_idx,
                                             float* C_vals, int* C_row_ptr, int* C_col_idx,
                                             int rows, float alpha, float beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= rows) return;
    
    int c_start = C_row_ptr[row];
    int a_start = A_row_ptr[row];
    int a_end = A_row_ptr[row + 1];
    int b_start = B_row_ptr[row];
    int b_end = B_row_ptr[row + 1];
    
    int c_idx = c_start;
    int a_idx = a_start;
    int b_idx = b_start;
    
    // AMD GPU optimization: process multiple elements per thread
    while (a_idx < a_end && b_idx < b_end) {
        int a_col = A_col_idx[a_idx];
        int b_col = B_col_idx[b_idx];
        
        if (a_col < b_col) {
            C_col_idx[c_idx] = a_col;
            C_vals[c_idx] = alpha * A_vals[a_idx];
            a_idx++;
        } else if (b_col < a_col) {
            C_col_idx[c_idx] = b_col;
            C_vals[c_idx] = beta * B_vals[b_idx];
            b_idx++;
        } else {
            C_col_idx[c_idx] = a_col;
            C_vals[c_idx] = alpha * A_vals[a_idx] + beta * B_vals[b_idx];
            a_idx++;
            b_idx++;
        }
        c_idx++;
    }
    
    while (a_idx < a_end) {
        C_col_idx[c_idx] = A_col_idx[a_idx];
        C_vals[c_idx] = alpha * A_vals[a_idx];
        a_idx++;
        c_idx++;
    }
    
    while (b_idx < b_end) {
        C_col_idx[c_idx] = B_col_idx[b_idx];
        C_vals[c_idx] = beta * B_vals[b_idx];
        b_idx++;
        c_idx++;
    }
}

void demonstrateSparseOperations() {
    std::cout << "\n=== HIP/ROCm Sparse Matrix Operations Demo ===" << std::endl;
    
    const int rows = 4096;
    const int cols = 4096;
    const double sparsity = 0.01; // 1% non-zero elements
    
    std::cout << "Generating sparse matrix (" << rows << "x" << cols 
              << ", sparsity=" << sparsity << ")..." << std::endl;
    
    CSRMatrix matrix = generateRandomSparseMatrix(rows, cols, sparsity);
    std::cout << "Generated matrix with " << matrix.nnz << " non-zero elements" << std::endl;
    
    // Allocate device memory
    float *d_x, *d_y1, *d_y2;
    
    CHECK_HIP(hipMalloc(&d_x, matrix.cols * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_y1, matrix.rows * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_y2, matrix.rows * sizeof(float)));
    
    // Initialize input vector
    std::vector<float> x(matrix.cols, 1.0f);
    CHECK_HIP(hipMemcpy(d_x, x.data(), matrix.cols * sizeof(float), hipMemcpyHostToDevice));
    
    Timer timer;
    SparseMatrixMultiplier multiplier;
    
    // Test rocSPARSE SpMV
    std::cout << "\n--- rocSPARSE SpMV ---" << std::endl;
    timer.start();
    multiplier.spmv_rocsparse(matrix, d_x, d_y1);
    double rocsparse_time = timer.elapsed();
    std::cout << "rocSPARSE time: " << rocsparse_time << " ms" << std::endl;
    
    // Test custom kernels optimized for AMD
    const char* kernel_names[] = {"Thread-per-row", "Wavefront-per-row (AMD)", "Block-per-row with LDS"};
    
    for (int kernel_type = 0; kernel_type < 3; ++kernel_type) {
        std::cout << "\n--- Custom " << kernel_names[kernel_type] << " ---" << std::endl;
        
        CHECK_HIP(hipMemset(d_y2, 0, matrix.rows * sizeof(float)));
        
        timer.start();
        multiplier.spmv_custom(matrix, d_x, d_y2, kernel_type);
        double custom_time = timer.elapsed();
        
        std::cout << "Custom kernel time: " << custom_time << " ms" << std::endl;
        std::cout << "Speedup vs rocSPARSE: " << rocsparse_time / custom_time << "x" << std::endl;
        
        // Verify correctness
        std::vector<float> y1(matrix.rows), y2(matrix.rows);
        CHECK_HIP(hipMemcpy(y1.data(), d_y1, matrix.rows * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(y2.data(), d_y2, matrix.rows * sizeof(float), hipMemcpyDeviceToHost));
        
        double max_error = 0.0;
        for (int i = 0; i < matrix.rows; ++i) {
            double error = std::abs(y1[i] - y2[i]);
            max_error = std::max(max_error, error);
        }
        std::cout << "Max error vs rocSPARSE: " << max_error << std::endl;
    }
    
    // AMD GPU-specific performance analysis
    std::cout << "\n=== AMD GPU Performance Analysis ===" << std::endl;
    double flops = 2.0 * matrix.nnz; // One multiply and one add per non-zero
    double gflops_rocsparse = flops / (rocsparse_time * 1e6);
    std::cout << "rocSPARSE GFLOPS: " << gflops_rocsparse << std::endl;
    
    double bandwidth_rocsparse = (matrix.nnz * sizeof(float) + matrix.nnz * sizeof(int) + 
                                 matrix.cols * sizeof(float) + matrix.rows * sizeof(float)) / 
                                (rocsparse_time * 1e6);
    std::cout << "rocSPARSE Bandwidth: " << bandwidth_rocsparse << " GB/s" << std::endl;
    
    // AMD GPU memory hierarchy efficiency
    double memory_intensity = flops / (matrix.nnz * sizeof(float) + matrix.nnz * sizeof(int) + 
                                      matrix.cols * sizeof(float) + matrix.rows * sizeof(float));
    std::cout << "Memory Intensity (FLOPS/Byte): " << memory_intensity << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y1));
    HIP_CHECK(hipFree(d_y2));
}

void demonstrateAdvancedSparseOperations() {
    std::cout << "\n=== Advanced AMD Sparse Operations ===" << std::endl;
    
    // Demonstrate sparse matrix-matrix multiplication using rocSPARSE
    rocsparse_handle handle;
    CHECK_ROCSPARSE(rocsparse_create_handle(&handle));
    
    const int rows = 1024;
    const int cols = 1024;
    
    CSRMatrix A = generateRandomSparseMatrix(rows, cols, 0.05);
    CSRMatrix B = generateRandomSparseMatrix(rows, cols, 0.05);
    
    std::cout << "Matrix A: " << A.rows << "x" << A.cols << " with " << A.nnz << " non-zeros" << std::endl;
    std::cout << "Matrix B: " << B.rows << "x" << B.cols << " with " << B.nnz << " non-zeros" << std::endl;
    
    // Allocate device memory for matrices
    float *d_A_vals, *d_B_vals;
    int *d_A_row_ptr, *d_A_col_idx, *d_B_row_ptr, *d_B_col_idx;
    
    CHECK_HIP(hipMalloc(&d_A_vals, A.nnz * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_A_row_ptr, (A.rows + 1) * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_A_col_idx, A.nnz * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_B_vals, B.nnz * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_B_row_ptr, (B.rows + 1) * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_B_col_idx, B.nnz * sizeof(int)));
    
    // Copy data to device
    CHECK_HIP(hipMemcpy(d_A_vals, A.values.data(), A.nnz * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_A_row_ptr, A.row_ptr.data(), (A.rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_A_col_idx, A.col_idx.data(), A.nnz * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B_vals, B.values.data(), B.nnz * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B_row_ptr, B.row_ptr.data(), (B.rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B_col_idx, B.col_idx.data(), B.nnz * sizeof(int), hipMemcpyHostToDevice));
    
    Timer timer;
    timer.start();
    
    // Create matrix descriptors
    rocsparse_mat_descr descr_A, descr_B;
    CHECK_ROCSPARSE(rocsparse_create_mat_descr(&descr_A));
    CHECK_ROCSPARSE(rocsparse_create_mat_descr(&descr_B));
    CHECK_ROCSPARSE(rocsparse_set_mat_index_base(descr_A, rocsparse_index_base_zero));
    CHECK_ROCSPARSE(rocsparse_set_mat_index_base(descr_B, rocsparse_index_base_zero));
    
    // Estimate memory requirements for C = A * B
    const float alpha = 1.0f;
    size_t buffer_size = 0;
    
    CHECK_ROCSPARSE(rocsparse_scsrgemm_buffer_size(handle, rocsparse_operation_none, rocsparse_operation_none,
                                                  A.rows, B.cols, A.cols, &alpha,
                                                  descr_A, A.nnz, d_A_row_ptr, d_A_col_idx,
                                                  descr_B, B.nnz, d_B_row_ptr, d_B_col_idx,
                                                  nullptr, nullptr, 0, nullptr, nullptr,
                                                  nullptr, &buffer_size));
    
    void* temp_buffer = nullptr;
    if (buffer_size > 0) {
        CHECK_HIP(hipMalloc(&temp_buffer, buffer_size));
    }
    
    double spgemm_setup_time = timer.elapsed();
    std::cout << "AMD sparse matrix-matrix multiplication setup time: " << spgemm_setup_time << " ms" << std::endl;
    std::cout << "Required buffer size: " << buffer_size / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // Demonstrate AMD-specific optimizations
    std::cout << "\n=== AMD GPU Architecture Optimizations ===" << std::endl;
    std::cout << "- Wavefront size: 64 threads (vs 32 for NVIDIA)" << std::endl;
    std::cout << "- Local Data Share (LDS): 64 KB per compute unit" << std::endl;
    std::cout << "- Memory coalescing: 64-byte cache lines" << std::endl;
    std::cout << "- Bank conflict avoidance: 32 banks in LDS" << std::endl;
    
    // Cleanup
    if (temp_buffer) HIP_CHECK(hipFree(temp_buffer));
    rocsparse_destroy_mat_descr(descr_A);
    rocsparse_destroy_mat_descr(descr_B);
    rocsparse_destroy_handle(handle);
    
    HIP_CHECK(hipFree(d_A_vals));
    HIP_CHECK(hipFree(d_A_row_ptr));
    HIP_CHECK(hipFree(d_A_col_idx));
    HIP_CHECK(hipFree(d_B_vals));
    HIP_CHECK(hipFree(d_B_row_ptr));
    HIP_CHECK(hipFree(d_B_col_idx));
}

int main() {
    std::cout << "HIP/ROCm Sparse Matrix Operations Demo" << std::endl;
    std::cout << "======================================" << std::endl;
    
#ifdef HAS_ROCSPARSE
    demonstrateSparseOperations();
    demonstrateAdvancedSparseOperations();
    
    return 0;
#else
    std::cout << "Note: This example requires rocSPARSE library which is not available." << std::endl;
    std::cout << "To enable this example:" << std::endl;
    std::cout << "1. Install rocSPARSE: sudo apt install rocsparse-dev" << std::endl;
    std::cout << "2. Compile with -DHAS_ROCSPARSE flag" << std::endl;
    std::cout << "3. Link with -lrocsparse -lrocblas" << std::endl;
    std::cout << std::endl;
    std::cout << "Skipping sparse matrix operations..." << std::endl;
    return 0;
#endif
}