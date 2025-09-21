#include <hip/hip_runtime.h>
#include "rocm7_utils.h"  // ROCm 7.0 enhanced utilities
#include <rocm_smi/rocm_smi.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <exception>
#include <map>
#include <mutex>
#include <thread>

#define CHECK_HIP(call) do { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        throw GPUException("HIP Error", hipGetErrorString(error), __FILE__, __LINE__); \
    } \
} while(0)

#define CHECK_RSMI(call) do { \
    rsmi_status_t result = call; \
    if (result != RSMI_STATUS_SUCCESS) { \
        throw GPUException("ROCm SMI Error", "Error code: " + std::to_string(result), __FILE__, __LINE__); \
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
    uint32_t device_id_;
    uint32_t device_count_;
    ErrorLogger& logger_;
    
public:
    GPUHealthChecker(ErrorLogger& logger) : device_id_(0), logger_(logger) {
        CHECK_RSMI(rsmi_init(0));
        CHECK_RSMI(rsmi_num_monitor_devices(&device_count_));
        
        if (device_count_ == 0) {
            throw GPUException("System Error", "No AMD GPUs found", __FILE__, __LINE__);
        }
        
        logger_.logInfo("GPU Health Checker initialized for " + std::to_string(device_count_) + " AMD devices");
    }
    
    ~GPUHealthChecker() {
        rsmi_shut_down();
    }
    
    struct HealthStatus {
        uint64_t temperature;
        uint64_t power_usage;
        uint64_t memory_used;
        uint64_t memory_total;
        uint32_t gpu_utilization;
        bool is_healthy;
        std::string warnings;
    };
    
    HealthStatus checkHealth() {
        HealthStatus status = {};
        
        try {
            // Get temperature
            int64_t temp_millidegrees;
            CHECK_RSMI(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_JUNCTION, 
                      RSMI_TEMP_CURRENT, &temp_millidegrees));
            status.temperature = temp_millidegrees / 1000; // Convert to degrees C
            
            // Get power usage
            uint64_t power_microwatts;
            CHECK_RSMI(rsmi_dev_power_ave_get(device_id_, 0, &power_microwatts));
            status.power_usage = power_microwatts; // Keep in microwatts
            
            // Get memory info
            uint64_t memory_total, memory_used;
            CHECK_RSMI(rsmi_dev_memory_total_get(device_id_, RSMI_MEM_TYPE_VRAM, &memory_total));
            CHECK_RSMI(rsmi_dev_memory_usage_get(device_id_, RSMI_MEM_TYPE_VRAM, &memory_used));
            status.memory_used = memory_used / (1024 * 1024); // MB
            status.memory_total = memory_total / (1024 * 1024); // MB
            
            // Get GPU utilization
            uint32_t utilization_percent;
            CHECK_RSMI(rsmi_dev_busy_percent_get(device_id_, &utilization_percent));
            status.gpu_utilization = utilization_percent;
            
            status.is_healthy = true;
            
            // Check temperature (AMD GPUs typically run hotter)
            if (status.temperature > 90) {
                status.is_healthy = false;
                status.warnings += "Critical temperature (" + std::to_string(status.temperature) + "C); ";
                logger_.logWarning("GPU temperature critical: " + std::to_string(status.temperature) + "C");
            } else if (status.temperature > 80) {
                status.warnings += "High temperature (" + std::to_string(status.temperature) + "C); ";
                logger_.logWarning("GPU temperature high: " + std::to_string(status.temperature) + "C");
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
        std::cout << "\n=== AMD GPU Health Status ===" << std::endl;
        std::cout << "Temperature: " << status.temperature << "°C" << std::endl;
        std::cout << "Power Usage: " << status.power_usage / 1000000.0 << "W" << std::endl;
        std::cout << "Memory: " << status.memory_used << "/" << status.memory_total << " MB ("
                  << (double(status.memory_used) / status.memory_total) * 100 << "%)" << std::endl;
        std::cout << "GPU Utilization: " << status.gpu_utilization << "%" << std::endl;
        std::cout << "Status: " << (status.is_healthy ? "HEALTHY" : "WARNING") << std::endl;
        if (!status.warnings.empty()) {
            std::cout << "Warnings: " << status.warnings << std::endl;
        }
        std::cout << "============================" << std::endl;
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
            CHECK_HIP(hipMalloc(&ptr, size));
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
                CHECK_HIP(HIP_CHECK(hipFree(ptr));
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
                CHECK_HIP(HIP_CHECK(hipFree(pair.first));
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
        // Simulate various error conditions for AMD GPUs
        int* invalid_ptr = nullptr;
        *invalid_ptr = 42; // This will cause a segmentation fault
    }
    
    if (idx < n) {
        // AMD GPUs have native support for sqrt, use it efficiently
        data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
    }
}

__global__ void recovery_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Simple recovery operation optimized for AMD architecture
        data[idx] = (data[idx] >= 0) ? data[idx] : 0.0f;
    }
}

__global__ void wavefront_aware_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = idx & 63; // AMD wavefront size is 64
    
    if (idx < n) {
        // Wavefront-aware processing
        float value = data[idx];
        
        // Use AMD-specific intrinsics if available
        if (lane_id == 0) {
            // Wavefront leader does additional work
            value *= 1.1f;
        }
        
        data[idx] = value;
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
                
                // Use AMD-optimized block size (multiple of 64 for wavefront efficiency)
                dim3 block_size(256);
                dim3 grid_size((n + block_size.x - 1) / block_size.x);
                
                // Execute kernel with error checking
                hipLaunchKernelGGL(error_prone_kernel, grid_size, block_size, 0, 0, 
                                  d_data, n, trigger_error && retry_count == 0);
                
                // Check for kernel launch errors
                CHECK_HIP(hipGetLastError());
                
                // Synchronize and check for execution errors
                CHECK_HIP(hipDeviceSynchronize());
                
                logger_.logInfo("Kernel executed successfully on attempt " + std::to_string(retry_count + 1));
                return true;
                
            } catch (const GPUException& e) {
                retry_count++;
                logger_.logError(e);
                logger_.logWarning("Kernel execution failed, attempt " + std::to_string(retry_count) + "/" + std::to_string(max_retries));
                
                if (retry_count < max_retries) {
                    // Attempt recovery
                    try {
                        // Reset HIP context
                        CHECK_HIP(hipDeviceReset());
                        
                        // Run recovery kernel
                        dim3 block_size(256);
                        dim3 grid_size((n + block_size.x - 1) / block_size.x);
                        hipLaunchKernelGGL(recovery_kernel, grid_size, block_size, 0, 0, d_data, n);
                        CHECK_HIP(hipDeviceSynchronize());
                        
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
    
    bool executeWavefrontAware(float* d_data, int n) {
        try {
            // Use wavefront-aware kernel for AMD GPUs
            dim3 block_size(256); // Multiple of 64 for wavefront efficiency
            dim3 grid_size((n + block_size.x - 1) / block_size.x);
            
            hipLaunchKernelGGL(wavefront_aware_kernel, grid_size, block_size, 0, 0, d_data, n);
            CHECK_HIP(hipGetLastError());
            CHECK_HIP(hipDeviceSynchronize());
            
            logger_.logInfo("Wavefront-aware kernel executed successfully");
            return true;
            
        } catch (const GPUException& e) {
            logger_.logError(e);
            return false;
        }
    }
};

void demonstrateErrorHandling() {
    try {
        ErrorLogger logger("hip_error_demo.log");
        logger.logInfo("=== Starting HIP Error Handling Demo ===");
        
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
        CHECK_HIP(hipMemcpy(d_data, h_data.data(), size, hipMemcpyHostToDevice));
        
        std::cout << "\n=== Test 1: Normal Execution ===" << std::endl;
        bool success = executor.executeWithRecovery(d_data, n, false);
        std::cout << "Normal execution: " << (success ? "SUCCESS" : "FAILED") << std::endl;
        
        std::cout << "\n=== Test 2: AMD Wavefront-Aware Execution ===" << std::endl;
        success = executor.executeWavefrontAware(d_data, n);
        std::cout << "Wavefront-aware execution: " << (success ? "SUCCESS" : "FAILED") << std::endl;
        
        std::cout << "\n=== Test 3: Error Recovery ===" << std::endl;
        success = executor.executeWithRecovery(d_data, n, true);
        std::cout << "Error recovery: " << (success ? "SUCCESS" : "FAILED") << std::endl;
        
        // Copy results back
        CHECK_HIP(hipMemcpy(h_data.data(), d_data, size, hipMemcpyDeviceToHost));
        
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
        ErrorLogger logger("hip_advanced_error_demo.log");
        logger.logInfo("=== Starting Advanced HIP Error Handling Demo ===");
        
        // Test memory allocation limits
        std::cout << "\n=== Testing AMD GPU Memory Allocation Limits ===" << std::endl;
        SafeMemoryManager memory_manager(logger);
        
        try {
            // Try to allocate an unreasonably large amount of memory
            size_t huge_size = 100LL * 1024 * 1024 * 1024; // 100 GB
            void* huge_ptr = memory_manager.allocate(huge_size);
            memory_manager.deallocate(huge_ptr);
        } catch (const GPUException& e) {
            std::cout << "Expected memory allocation failure: " << e.what() << std::endl;
        }
        
        // Test multiple small allocations optimized for AMD memory hierarchy
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
        
        logger.logInfo("=== Advanced AMD GPU demo completed ===");
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in advanced demo: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "HIP Error Handling and Recovery Demo" << std::endl;
    std::cout << "====================================" << std::endl;
    
    demonstrateErrorHandling();
    
    std::cout << "\n" << std::endl;
    
    demonstrateAdvancedErrorHandling();
    
    return 0;
}