#include <cuda_runtime.h>
#include <texture_fetch_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Modern texture object approach (CUDA 5.0+)
__global__ void textureFilterKernel(cudaTextureObject_t texObj, float *output, 
                                   int width, int height, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_filter = filter_size / 2;
        
        // Apply filter using texture memory
        for (int fy = -half_filter; fy <= half_filter; fy++) {
            for (int fx = -half_filter; fx <= half_filter; fx++) {
                // Normalize coordinates to [0,1] range
                float u = (float)(x + fx + 0.5f) / width;
                float v = (float)(y + fy + 0.5f) / height;
                
                // Texture automatically handles boundary conditions and interpolation
                float value = tex2D<float>(texObj, u, v);
                sum += value;
            }
        }
        
        output[y * width + x] = sum / (filter_size * filter_size);
    }
}

// Texture-based matrix transpose with spatial locality
__global__ void textureTranspose(cudaTextureObject_t texObj, float *output, 
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Normalized coordinates
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;
        
        // Fetch using texture cache
        float value = tex2D<float>(texObj, u, v);
        
        // Write transposed
        if (y < width && x < height) {
            output[x * height + y] = value;
        }
    }
}

// Bilinear interpolation example
__global__ void bilinearInterpolation(cudaTextureObject_t texObj, float *output, 
                                     int out_width, int out_height, 
                                     int in_width, int in_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < out_width && y < out_height) {
        // Map output coordinates to input coordinates
        float scale_x = (float)in_width / out_width;
        float scale_y = (float)in_height / out_height;
        
        float src_x = (x + 0.5f) * scale_x;
        float src_y = (y + 0.5f) * scale_y;
        
        // Normalize coordinates
        float u = src_x / in_width;
        float v = src_y / in_height;
        
        // Hardware bilinear interpolation
        float interpolated = tex2D<float>(texObj, u, v);
        
        output[y * out_width + x] = interpolated;
    }
}

// Compare texture vs global memory performance
__global__ void globalMemoryFilter(float *input, float *output, 
                                  int width, int height, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_filter = filter_size / 2;
        
        for (int fy = -half_filter; fy <= half_filter; fy++) {
            for (int fx = -half_filter; fx <= half_filter; fx++) {
                int px = min(max(x + fx, 0), width - 1);   // Clamp to bounds
                int py = min(max(y + fy, 0), height - 1);
                
                sum += input[py * width + px];
            }
        }
        
        output[y * width + x] = sum / (filter_size * filter_size);
    }
}

// Texture-based convolution with different filter kernels
__constant__ float const_filter[25];  // Max 5x5 filter

__global__ void textureConvolution(cudaTextureObject_t texObj, float *output,
                                  int width, int height, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_filter = filter_size / 2;
        
        for (int fy = -half_filter; fy <= half_filter; fy++) {
            for (int fx = -half_filter; fx <= half_filter; fx++) {
                float u = (float)(x + fx + 0.5f) / width;
                float v = (float)(y + fy + 0.5f) / height;
                
                float pixel = tex2D<float>(texObj, u, v);
                int filter_idx = (fy + half_filter) * filter_size + (fx + half_filter);
                
                sum += pixel * const_filter[filter_idx];
            }
        }
        
        output[y * width + x] = sum;
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Helper function to create 2D texture object
cudaTextureObject_t createTexture2D(float *data, int width, int height) {
    // Allocate CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
    
    // Copy data to CUDA array
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, data, 
                                  width * sizeof(float), width * sizeof(float), 
                                  height, cudaMemcpyHostToDevice));
    
    // Specify texture resource
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    // Specify texture object parameters
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;  // Clamp to edge
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;      // Bilinear filtering
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;                   // Normalize coordinates [0,1]
    
    // Create texture object
    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    
    return texObj;
}

// Helper function to destroy texture and free resources
void destroyTexture2D(cudaTextureObject_t texObj) {
    // Get resource descriptor to access the array
    cudaResourceDesc resDesc;
    CUDA_CHECK(cudaGetTextureObjectResourceDesc(&resDesc, texObj));
    
    // Destroy texture object
    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    
    // Free CUDA array
    CUDA_CHECK(cudaFreeArray(resDesc.res.array.array));
}

void runFilterBenchmark() {
    printf("=== Texture vs Global Memory Filter Benchmark ===\n");
    
    const int width = 1024;
    const int height = 1024;
    const int filter_size = 5;
    const int size = width * height * sizeof(float);
    
    // Host data
    float *h_input = (float*)malloc(size);
    float *h_output_texture = (float*)malloc(size);
    float *h_output_global = (float*)malloc(size);
    
    // Initialize input with test pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_input[y * width + x] = sinf(x * 0.1f) * cosf(y * 0.1f);
        }
    }
    
    // Device memory
    float *d_input, *d_output_texture, *d_output_global;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output_texture, size));
    CUDA_CHECK(cudaMalloc(&d_output_global, size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Create texture
    cudaTextureObject_t texObj = createTexture2D(h_input, width, height);
    
    // Grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Benchmark texture memory
    CUDA_CHECK(cudaEventRecord(start));
    textureFilterKernel<<<gridSize, blockSize>>>(texObj, d_output_texture, 
                                                 width, height, filter_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float texture_time;
    CUDA_CHECK(cudaEventElapsedTime(&texture_time, start, stop));
    
    // Benchmark global memory
    CUDA_CHECK(cudaEventRecord(start));
    globalMemoryFilter<<<gridSize, blockSize>>>(d_input, d_output_global, 
                                               width, height, filter_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float global_time;
    CUDA_CHECK(cudaEventElapsedTime(&global_time, start, stop));
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_output_texture, d_output_texture, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_global, d_output_global, size, cudaMemcpyDeviceToHost));
    
    // Verify results are similar
    float max_diff = 0.0f;
    for (int i = 0; i < width * height; i++) {
        float diff = fabs(h_output_texture[i] - h_output_global[i]);
        max_diff = fmax(max_diff, diff);
    }
    
    printf("Image size: %dx%d\n", width, height);
    printf("Filter size: %dx%d\n", filter_size, filter_size);
    printf("Texture memory time: %.3f ms\n", texture_time);
    printf("Global memory time: %.3f ms\n", global_time);
    printf("Texture speedup: %.2fx\n", global_time / texture_time);
    printf("Max difference: %.6f\n", max_diff);
    printf("Results match: %s\n", (max_diff < 1e-4) ? "YES" : "NO");
    
    // Cleanup
    free(h_input); free(h_output_texture); free(h_output_global);
    cudaFree(d_input); cudaFree(d_output_texture); cudaFree(d_output_global);
    destroyTexture2D(texObj);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

void runInterpolationExample() {
    printf("\n=== Bilinear Interpolation Example ===\n");
    
    const int in_width = 256, in_height = 256;
    const int out_width = 512, out_height = 512;  // Upscale 2x
    
    const int in_size = in_width * in_height * sizeof(float);
    const int out_size = out_width * out_height * sizeof(float);
    
    // Create test pattern
    float *h_input = (float*)malloc(in_size);
    float *h_output = (float*)malloc(out_size);
    
    for (int y = 0; y < in_height; y++) {
        for (int x = 0; x < in_width; x++) {
            // Create a checkerboard pattern
            h_input[y * in_width + x] = ((x/16) + (y/16)) % 2;
        }
    }
    
    // Create texture
    cudaTextureObject_t texObj = createTexture2D(h_input, in_width, in_height);
    
    // Device output
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, out_size));
    
    // Interpolation
    dim3 blockSize(16, 16);
    dim3 gridSize((out_width + blockSize.x - 1) / blockSize.x,
                  (out_height + blockSize.y - 1) / blockSize.y);
    
    bilinearInterpolation<<<gridSize, blockSize>>>(texObj, d_output, 
                                                   out_width, out_height,
                                                   in_width, in_height);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost));
    
    printf("Input size: %dx%d\n", in_width, in_height);
    printf("Output size: %dx%d\n", out_width, out_height);
    printf("Scale factor: %.2fx\n", (float)out_width / in_width);
    
    // Sample a few values to show interpolation
    printf("Sample interpolated values:\n");
    for (int i = 0; i < 5; i++) {
        int x = i * out_width / 5;
        int y = out_height / 2;
        printf("  (%d, %d) = %.3f\n", x, y, h_output[y * out_width + x]);
    }
    
    free(h_input); free(h_output);
    cudaFree(d_output);
    destroyTexture2D(texObj);
}

void runConvolutionExample() {
    printf("\n=== Texture-Based Convolution Example ===\n");
    
    const int width = 512, height = 512;
    const int filter_size = 3;
    const int size = width * height * sizeof(float);
    
    // Define edge detection filter (Sobel X)
    float h_filter[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    
    // Copy filter to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(const_filter, h_filter, 
                                 filter_size * filter_size * sizeof(float)));
    
    // Create test image (gradient)
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_input[y * width + x] = (float)x / width;  // Horizontal gradient
        }
    }
    
    // Create texture
    cudaTextureObject_t texObj = createTexture2D(h_input, width, height);
    
    // Device output
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    // Apply convolution
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    textureConvolution<<<gridSize, blockSize>>>(texObj, d_output, 
                                               width, height, filter_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float conv_time;
    CUDA_CHECK(cudaEventElapsedTime(&conv_time, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    printf("Image size: %dx%d\n", width, height);
    printf("Filter: Sobel X (edge detection)\n");
    printf("Convolution time: %.3f ms\n", conv_time);
    
    // Find min/max for normalization info
    float min_val = h_output[0], max_val = h_output[0];
    for (int i = 1; i < width * height; i++) {
        min_val = fmin(min_val, h_output[i]);
        max_val = fmax(max_val, h_output[i]);
    }
    
    printf("Output range: [%.3f, %.3f]\n", min_val, max_val);
    
    free(h_input); free(h_output);
    cudaFree(d_output);
    destroyTexture2D(texObj);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

void runTextureTransposeExample() {
    printf("\n=== Texture-Based Matrix Transpose ===\n");
    
    const int width = 1024, height = 1024;
    const int size = width * height * sizeof(float);
    
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    
    // Initialize matrix
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)i;
    }
    
    // Create texture
    cudaTextureObject_t texObj = createTexture2D(h_input, width, height);
    
    // Device output
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    // Transpose using texture
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    textureTranspose<<<gridSize, blockSize>>>(texObj, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float transpose_time;
    CUDA_CHECK(cudaEventElapsedTime(&transpose_time, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Verify transpose is correct
    bool correct = true;
    for (int y = 0; y < height && correct; y++) {
        for (int x = 0; x < width && correct; x++) {
            float expected = y * width + x;  // Original: h_input[y][x] = y*w+x
            float actual = h_output[x * height + y];  // Transposed: h_output[x][y]
            if (fabs(actual - expected) > 1e-5) {
                correct = false;
            }
        }
    }
    
    printf("Matrix size: %dx%d\n", width, height);
    printf("Transpose time: %.3f ms\n", transpose_time);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    double bandwidth = (2.0 * size / (1024.0 * 1024.0 * 1024.0)) / (transpose_time / 1000.0);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);
    
    free(h_input); free(h_output);
    cudaFree(d_output);
    destroyTexture2D(texObj);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("CUDA Texture Memory Examples\n");
    printf("============================\n");
    
    // Check device properties
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("Running on: %s\n", props.name);
    printf("Texture alignment: %zu bytes\n", props.textureAlignment);
    printf("Max 2D texture dimensions: %dx%d\n", 
           props.maxTexture2D[0], props.maxTexture2D[1]);
    printf("Max 2D layered texture dimensions: %dx%dx%d\n",
           props.maxTexture2DLayered[0], props.maxTexture2DLayered[1], 
           props.maxTexture2DLayered[2]);
    
    // Run examples
    runFilterBenchmark();
    runInterpolationExample();
    runConvolutionExample();
    runTextureTransposeExample();
    
    // Educational summary
    printf("\n=== Texture Memory Benefits ===\n");
    printf("✓ ADVANTAGES:\n");
    printf("  - Automatic caching with spatial locality optimization\n");
    printf("  - Hardware bilinear/trilinear interpolation\n");
    printf("  - Boundary handling (clamp, wrap, mirror)\n");
    printf("  - Read-only access with optimized cache behavior\n");
    printf("  - Reduces memory bandwidth pressure\n");
    
    printf("\n✓ BEST USE CASES:\n");
    printf("  - Image processing and filtering\n");
    printf("  - Interpolation and resampling\n");
    printf("  - Lookup tables with spatial coherence\n");
    printf("  - Scientific simulations with stencil patterns\n");
    printf("  - Computer graphics and rendering\n");
    
    printf("\n⚠ LIMITATIONS:\n");
    printf("  - Read-only access from kernels\n");
    printf("  - Additional memory overhead for CUDA arrays\n");
    printf("  - Limited by texture cache size\n");
    printf("  - Coordinate normalization overhead\n");
    
    printf("\n💡 OPTIMIZATION TIPS:\n");
    printf("  - Use for algorithms with spatial locality\n");
    printf("  - Consider texture cache behavior in access patterns\n");
    printf("  - Leverage hardware interpolation when appropriate\n");
    printf("  - Use appropriate addressing modes for boundary conditions\n");
    printf("  - Profile to ensure texture cache hits\n");
    
    printf("\nTexture memory examples completed successfully!\n");
    return 0;
}