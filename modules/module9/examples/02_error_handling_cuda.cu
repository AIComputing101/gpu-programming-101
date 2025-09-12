#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvml.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <exception>
#include <map>
#include <mutex>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw GPUException("CUDA Error", cudaGetErrorString(error), __FILE__, __LINE__); \
    } \
} while(0)

#define CHECK_NVML(call) do { \
    nvmlReturn_t result = call; \
    if (result != NVML_SUCCESS) { \
        throw GPUException("NVML Error", nvmlErrorString(result), __FILE__, __LINE__); \
    } \
} while(0)

class GPUException : public std::exception {
private:
    std::string message_;
    std::string file_;
    int line_;
    
public:
    GPUException(const std::string& type, const std::string& msg, const char* file, int line)
        : message_(type + ": " + msg), file_(file), line_(line) {}
    
    const char* what() const noexcept override {
        return message_.c_str();
    }
    
    std::string getDetails() const {
        return message_ + " at " + file_ + ":" + std::to_string(line_);
    }
};

class ErrorLogger {
private:
    std::ofstream log_file_;
    std::mutex log_mutex_;
    
public:
    ErrorLogger(const std::string& filename = "gpu_errors.log") 
        : log_file_(filename, std::ios::app) {
        if (!log_file_.is_open()) {
            throw std::runtime_error("Failed to open log file: " + filename);
        }
    }
    
    ~ErrorLogger() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }
    
    void log(const std::string& level, const std::string& message) {
        std::lock_guard<std::mutex> lock(log_mutex_);
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        log_file_ << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
                  << "] [" << level << "] " << message << std::endl;
        log_file_.flush();
    }
    
    void logError(const GPUException& e) {
        log("ERROR", e.getDetails());
    }
    
    void logWarning(const std::string& message) {
        log("WARNING", message);
    }
    
    void logInfo(const std::string& message) {
        log("INFO", message);
    }
};

class GPUHealthChecker {
private:
    nvmlDevice_t device_;
    unsigned int device_count_;
    ErrorLogger& logger_;
    
public:
    GPUHealthChecker(ErrorLogger& logger) : logger_(logger) {
        CHECK_NVML(nvmlInit());
        CHECK_NVML(nvmlDeviceGetCount(&device_count_));
        
        if (device_count_ == 0) {
            throw GPUException("System Error", "No NVIDIA GPUs found", __FILE__, __LINE__);
        }
        
        CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &device_));
        logger_.logInfo("GPU Health Checker initialized for " + std::to_string(device_count_) + " devices");
    }
    
    ~GPUHealthChecker() {
        nvmlShutdown();
    }
    
    struct HealthStatus {
        unsigned int temperature;
        unsigned int power_usage;
        unsigned int memory_used;
        unsigned int memory_total;
        unsigned int gpu_utilization;
        unsigned int memory_utilization;
        bool is_healthy;
        std::string warnings;
    };
    
    HealthStatus checkHealth() {
        HealthStatus status = {};
        
        try {
            CHECK_NVML(nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &status.temperature));
            CHECK_NVML(nvmlDeviceGetPowerUsage(device_, &status.power_usage));
            
            nvmlMemory_t memory_info;
            CHECK_NVML(nvmlDeviceGetMemoryInfo(device_, &memory_info));
            status.memory_used = memory_info.used / (1024 * 1024); // MB
            status.memory_total = memory_info.total / (1024 * 1024); // MB
            
            nvmlUtilization_t utilization;
            CHECK_NVML(nvmlDeviceGetUtilizationRates(device_, &utilization));
            status.gpu_utilization = utilization.gpu;
            status.memory_utilization = utilization.memory;
            
            status.is_healthy = true;
            
            // Check temperature
            if (status.temperature > 85) {
                status.is_healthy = false;
                status.warnings += "High temperature (" + std::to_string(status.temperature) + "C); ";
                logger_.logWarning("GPU temperature critical: " + std::to_string(status.temperature) + "C");
            } else if (status.temperature > 75) {
                status.warnings += "Elevated temperature (" + std::to_string(status.temperature) + "C); ";
                logger_.logWarning("GPU temperature elevated: " + std::to_string(status.temperature) + "C");
            }
            
            // Check memory usage
            double memory_usage_percent = (double(status.memory_used) / status.memory_total) * 100;
            if (memory_usage_percent > 90) {
                status.is_healthy = false;
                status.warnings += "Critical memory usage (" + std::to_string(int(memory_usage_percent)) + "%); ";
                logger_.logWarning("GPU memory usage critical: " + std::to_string(int(memory_usage_percent)) + "%");
            } else if (memory_usage_percent > 80) {
                status.warnings += "High memory usage (" + std::to_string(int(memory_usage_percent)) + "%); ";
                logger_.logWarning("GPU memory usage high: " + std::to_string(int(memory_usage_percent)) + "%");
            }
            
        } catch (const GPUException& e) {
            status.is_healthy = false;
            status.warnings = "Health check failed: " + std::string(e.what());
            logger_.logError(e);
        }
        
        return status;
    }
    
    void printHealthStatus(const HealthStatus& status) {
        std::cout << "\n=== GPU Health Status ===" << std::endl;
        std::cout << "Temperature: " << status.temperature << "°C" << std::endl;
        std::cout << "Power Usage: " << status.power_usage / 1000.0 << "W" << std::endl;
        std::cout << "Memory: " << status.memory_used << "/" << status.memory_total << " MB ("
                  << (double(status.memory_used) / status.memory_total) * 100 << "%)" << std::endl;
        std::cout << "GPU Utilization: " << status.gpu_utilization << "%" << std::endl;
        std::cout << "Memory Utilization: " << status.memory_utilization << "%" << std::endl;
        std::cout << "Status: " << (status.is_healthy ? "HEALTHY" : "WARNING") << std::endl;
        if (!status.warnings.empty()) {
            std::cout << "Warnings: " << status.warnings << std::endl;
        }
        std::cout << "========================" << std::endl;
    }
};

class SafeMemoryManager {
private:
    std::map<void*, size_t> allocated_ptrs_;
    std::mutex alloc_mutex_;
    size_t total_allocated_;
    ErrorLogger& logger_;
    
public:
    SafeMemoryManager(ErrorLogger& logger) : total_allocated_(0), logger_(logger) {}
    
    ~SafeMemoryManager() {
        cleanup();
    }
    
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(alloc_mutex_);
        
        void* ptr = nullptr;
        try {
            CHECK_CUDA(cudaMalloc(&ptr, size));
            allocated_ptrs_[ptr] = size;
            total_allocated_ += size;
            
            logger_.logInfo("Allocated " + std::to_string(size) + " bytes. Total: " + 
                           std::to_string(total_allocated_) + " bytes");
            
        } catch (const GPUException& e) {
            logger_.logError(e);
            throw;
        }
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(alloc_mutex_);
        
        auto it = allocated_ptrs_.find(ptr);
        if (it != allocated_ptrs_.end()) {
            try {
                CHECK_CUDA(cudaFree(ptr));
                total_allocated_ -= it->second;
                allocated_ptrs_.erase(it);
                
                logger_.logInfo("Deallocated memory. Total remaining: " + 
                               std::to_string(total_allocated_) + " bytes");
                
            } catch (const GPUException& e) {
                logger_.logError(e);
                throw;
            }
        } else {
            logger_.logWarning("Attempted to deallocate unknown pointer");
        }
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(alloc_mutex_);
        
        for (auto& pair : allocated_ptrs_) {
            try {
                CHECK_CUDA(cudaFree(pair.first));
                logger_.logInfo("Cleaned up " + std::to_string(pair.second) + " bytes");
            } catch (const GPUException& e) {
                logger_.logError(e);
            }
        }
        
        allocated_ptrs_.clear();
        total_allocated_ = 0;
    }
    
    size_t getTotalAllocated() const {
        std::lock_guard<std::mutex> lock(alloc_mutex_);
        return total_allocated_;
    }
    
    size_t getActiveAllocations() const {
        std::lock_guard<std::mutex> lock(alloc_mutex_);
        return allocated_ptrs_.size();
    }
};

__global__ void error_prone_kernel(float* data, int n, bool trigger_error) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (trigger_error && idx == 0) {
        // Simulate various error conditions
        int* invalid_ptr = nullptr;
        *invalid_ptr = 42; // This will cause a segmentation fault
    }
    
    if (idx < n) {
        data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
    }
}

__global__ void recovery_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Simple recovery operation
        data[idx] = (data[idx] >= 0) ? data[idx] : 0.0f;
    }
}

class RobustKernelExecutor {
private:
    SafeMemoryManager& memory_manager_;
    ErrorLogger& logger_;
    GPUHealthChecker& health_checker_;
    
public:
    RobustKernelExecutor(SafeMemoryManager& mem_mgr, ErrorLogger& logger, GPUHealthChecker& health)
        : memory_manager_(mem_mgr), logger_(logger), health_checker_(health) {}
    
    bool executeWithRecovery(float* d_data, int n, bool trigger_error = false) {
        const int max_retries = 3;
        int retry_count = 0;
        
        while (retry_count < max_retries) {
            try {
                // Check GPU health before execution
                auto health_status = health_checker_.checkHealth();
                if (!health_status.is_healthy) {
                    logger_.logWarning("GPU health check failed, but attempting execution");
                }
                
                dim3 block_size(256);
                dim3 grid_size((n + block_size.x - 1) / block_size.x);
                
                // Execute kernel with error checking
                error_prone_kernel<<<grid_size, block_size>>>(d_data, n, trigger_error && retry_count == 0);
                
                // Check for kernel launch errors
                CHECK_CUDA(cudaGetLastError());
                
                // Synchronize and check for execution errors
                CHECK_CUDA(cudaDeviceSynchronize());
                
                logger_.logInfo("Kernel executed successfully on attempt " + std::to_string(retry_count + 1));
                return true;
                
            } catch (const GPUException& e) {
                retry_count++;
                logger_.logError(e);
                logger_.logWarning("Kernel execution failed, attempt " + std::to_string(retry_count) + "/" + std::to_string(max_retries));
                
                if (retry_count < max_retries) {
                    // Attempt recovery
                    try {
                        // Reset CUDA context
                        CHECK_CUDA(cudaDeviceReset());
                        
                        // Run recovery kernel
                        dim3 block_size(256);
                        dim3 grid_size((n + block_size.x - 1) / block_size.x);
                        recovery_kernel<<<grid_size, block_size>>>(d_data, n);
                        CHECK_CUDA(cudaDeviceSynchronize());
                        
                        logger_.logInfo("Recovery kernel executed successfully");
                        
                        // Wait before retry
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                        
                    } catch (const GPUException& recovery_error) {
                        logger_.logError(recovery_error);
                        logger_.logWarning("Recovery attempt failed");
                    }
                }
            }
        }
        
        logger_.logWarning("All retry attempts failed");
        return false;
    }
};

void demonstrateErrorHandling() {
    try {
        ErrorLogger logger("cuda_error_demo.log");
        logger.logInfo("=== Starting CUDA Error Handling Demo ===");
        
        GPUHealthChecker health_checker(logger);
        SafeMemoryManager memory_manager(logger);
        RobustKernelExecutor executor(memory_manager, logger, health_checker);
        
        const int n = 1024 * 1024;
        const size_t size = n * sizeof(float);
        
        // Check initial GPU health
        auto initial_health = health_checker.checkHealth();
        health_checker.printHealthStatus(initial_health);
        
        // Allocate memory safely
        float* d_data = static_cast<float*>(memory_manager.allocate(size));
        std::vector<float> h_data(n, 1.0f);
        
        // Copy data to device
        CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
        
        std::cout << "\n=== Test 1: Normal Execution ===" << std::endl;
        bool success = executor.executeWithRecovery(d_data, n, false);
        std::cout << "Normal execution: " << (success ? "SUCCESS" : "FAILED") << std::endl;
        
        std::cout << "\n=== Test 2: Error Recovery ===" << std::endl;
        success = executor.executeWithRecovery(d_data, n, true);
        std::cout << "Error recovery: " << (success ? "SUCCESS" : "FAILED") << std::endl;
        
        // Copy results back
        CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, size, cudaMemcpyDeviceToHost));
        
        // Check final GPU health
        auto final_health = health_checker.checkHealth();
        health_checker.printHealthStatus(final_health);
        
        // Cleanup is automatic via RAII
        memory_manager.deallocate(d_data);
        
        logger.logInfo("=== Demo completed successfully ===");
        
    } catch (const GPUException& e) {
        std::cerr << "GPU Exception: " << e.getDetails() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
    }
}

void demonstrateAdvancedErrorHandling() {
    try {
        ErrorLogger logger("cuda_advanced_error_demo.log");
        logger.logInfo("=== Starting Advanced CUDA Error Handling Demo ===");
        
        // Test memory allocation limits
        std::cout << "\n=== Testing Memory Allocation Limits ===" << std::endl;
        SafeMemoryManager memory_manager(logger);
        
        try {
            // Try to allocate an unreasonably large amount of memory
            size_t huge_size = 100LL * 1024 * 1024 * 1024; // 100 GB
            void* huge_ptr = memory_manager.allocate(huge_size);
            memory_manager.deallocate(huge_ptr);
        } catch (const GPUException& e) {
            std::cout << "Expected memory allocation failure: " << e.what() << std::endl;
        }
        
        // Test multiple small allocations
        std::vector<void*> ptrs;
        for (int i = 0; i < 100; ++i) {
            try {
                void* ptr = memory_manager.allocate(1024 * 1024); // 1 MB each
                ptrs.push_back(ptr);
            } catch (const GPUException& e) {
                logger.logWarning("Allocation failed at iteration " + std::to_string(i) + ": " + e.what());
                break;
            }
        }
        
        std::cout << "Successfully allocated " << ptrs.size() << " memory blocks" << std::endl;
        std::cout << "Total allocated: " << memory_manager.getTotalAllocated() / (1024 * 1024) << " MB" << std::endl;
        
        // Cleanup all allocations
        for (void* ptr : ptrs) {
            memory_manager.deallocate(ptr);
        }
        
        logger.logInfo("=== Advanced demo completed ===");
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in advanced demo: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "CUDA Error Handling and Recovery Demo" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    demonstrateErrorHandling();
    
    std::cout << "\n" << std::endl;
    
    demonstrateAdvancedErrorHandling();
    
    return 0;
}