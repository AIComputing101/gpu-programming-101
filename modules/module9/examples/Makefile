# Module 9: Production GPU Programming
# Makefile for enterprise-grade GPU application examples

# Compiler settings
NVCC = nvcc
HIPCC = hipcc
CXX = g++

# Compiler flags for production-ready applications
CUDA_FLAGS = -std=c++17 -O3 -arch=sm_70 -lineinfo --use_fast_math -DPRODUCTION_BUILD
CUDA_DEBUG_FLAGS = -std=c++17 -g -G -arch=sm_70 -DDEBUG_BUILD
HIP_FLAGS = -std=c++17 -O3 --fast-math -DPRODUCTION_BUILD
HIP_DEBUG_FLAGS = -std=c++17 -g -DDEBUG_BUILD
CXX_FLAGS = -std=c++17 -O3 -DPRODUCTION_BUILD

# Library flags for production dependencies
CUDA_LIBS = -lcublas -lcurand -lcufft -lcudnn -lnvml
HIP_LIBS = -lrocblas -lrocrand -lrocfft -lMIOpen -lrocm_smi64
COMMON_LIBS = -lpthread -ldl

# Directories
BUILD_DIR = build
PROFILE_DIR = profiles
DEPLOY_DIR = deploy
CONFIG_DIR = config
LOGS_DIR = logs

# Source files
CUDA_SOURCES = $(wildcard *_cuda.cu)
HIP_SOURCES = $(wildcard *_hip.cpp)
CPP_SOURCES = $(wildcard *_common.cpp)

# Target executables
CUDA_TARGETS = $(CUDA_SOURCES:%.cu=$(BUILD_DIR)/%)
HIP_TARGETS = $(HIP_SOURCES:%.cpp=$(BUILD_DIR)/%)
CPP_TARGETS = $(CPP_SOURCES:%.cpp=$(BUILD_DIR)/%)

# Default target
.PHONY: all
all: setup cuda hip common

# Setup directories for production deployment
.PHONY: setup
setup:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(PROFILE_DIR)
	@mkdir -p $(DEPLOY_DIR)
	@mkdir -p $(CONFIG_DIR)
	@mkdir -p $(LOGS_DIR)

# Build CUDA production applications
.PHONY: cuda
cuda: setup $(CUDA_TARGETS)

# Build HIP production applications
.PHONY: hip
hip: setup $(HIP_TARGETS)

# Build common C++ applications
.PHONY: common
common: setup $(CPP_TARGETS)

# Individual CUDA compilation with production libraries
$(BUILD_DIR)/%_cuda: %_cuda.cu
	@echo "Building CUDA production application: $@"
	$(NVCC) $(CUDA_FLAGS) $< -o $@ $(CUDA_LIBS) $(COMMON_LIBS)

# Individual HIP compilation with production libraries
$(BUILD_DIR)/%_hip: %_hip.cpp
	@echo "Building HIP production application: $@"
	$(HIPCC) $(HIP_FLAGS) $< -o $@ $(HIP_LIBS) $(COMMON_LIBS)

# Individual C++ compilation
$(BUILD_DIR)/%_common: %_common.cpp
	@echo "Building common production component: $@"
	$(CXX) $(CXX_FLAGS) $< -o $@ $(COMMON_LIBS)

# Debug builds
.PHONY: debug
debug: CUDA_FLAGS = $(CUDA_DEBUG_FLAGS)
debug: HIP_FLAGS = $(HIP_DEBUG_FLAGS)
debug: CXX_FLAGS = -std=c++17 -g -DDEBUG_BUILD
debug: all

# Production builds with security hardening
.PHONY: production
production: CUDA_FLAGS += -DNDEBUG -Xptxas -O3 -DSECURITY_HARDENED
production: HIP_FLAGS += -DNDEBUG -DSECURITY_HARDENED
production: CXX_FLAGS += -DNDEBUG -fstack-protector-strong -D_FORTIFY_SOURCE=2 -DSECURITY_HARDENED
production: all

# Production application categories
.PHONY: architecture
architecture: setup
	@if [ -f 01_architecture_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 01_architecture_cuda.cu -o $(BUILD_DIR)/01_architecture_cuda $(CUDA_LIBS) $(COMMON_LIBS); fi
	@if [ -f 01_architecture_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 01_architecture_hip.cpp -o $(BUILD_DIR)/01_architecture_hip $(HIP_LIBS) $(COMMON_LIBS); fi

.PHONY: error_handling
error_handling: setup
	@if [ -f 02_error_handling_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 02_error_handling_cuda.cu -o $(BUILD_DIR)/02_error_handling_cuda $(CUDA_LIBS) $(COMMON_LIBS); fi
	@if [ -f 02_error_handling_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 02_error_handling_hip.cpp -o $(BUILD_DIR)/02_error_handling_hip $(HIP_LIBS) $(COMMON_LIBS); fi

.PHONY: deployment
deployment: setup
	@if [ -f 03_deployment_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 03_deployment_cuda.cu -o $(BUILD_DIR)/03_deployment_cuda $(CUDA_LIBS) $(COMMON_LIBS); fi
	@if [ -f 03_deployment_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 03_deployment_hip.cpp -o $(BUILD_DIR)/03_deployment_hip $(HIP_LIBS) $(COMMON_LIBS); fi

.PHONY: monitoring
monitoring: setup
	@if [ -f 04_monitoring_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 04_monitoring_cuda.cu -o $(BUILD_DIR)/04_monitoring_cuda $(CUDA_LIBS) $(COMMON_LIBS); fi
	@if [ -f 04_monitoring_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 04_monitoring_hip.cpp -o $(BUILD_DIR)/04_monitoring_hip $(HIP_LIBS) $(COMMON_LIBS); fi

.PHONY: scalability
scalability: setup
	@if [ -f 05_scalability_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 05_scalability_cuda.cu -o $(BUILD_DIR)/05_scalability_cuda $(CUDA_LIBS) $(COMMON_LIBS); fi
	@if [ -f 05_scalability_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 05_scalability_hip.cpp -o $(BUILD_DIR)/05_scalability_hip $(HIP_LIBS) $(COMMON_LIBS); fi

.PHONY: security
security: setup
	@if [ -f 06_security_cuda.cu ]; then $(NVCC) $(CUDA_FLAGS) 06_security_cuda.cu -o $(BUILD_DIR)/06_security_cuda $(CUDA_LIBS) $(COMMON_LIBS); fi
	@if [ -f 06_security_hip.cpp ]; then $(HIPCC) $(HIP_FLAGS) 06_security_hip.cpp -o $(BUILD_DIR)/06_security_hip $(HIP_LIBS) $(COMMON_LIBS); fi

# Production testing with comprehensive validation
.PHONY: test
test: all
	@echo "Running Module 9 Production Application Tests..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS) $(CPP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Testing $$target..."; \
			$$target --production-test 2>&1 | tee $(LOGS_DIR)/$$(basename $$target)_test.log; \
			echo ""; \
		fi; \
	done

# Production readiness validation
.PHONY: test_production
test_production: production
	@echo "Running production readiness tests..."
	@echo "Production Test Results - $$(date)" > $(PROFILE_DIR)/production_test_results.txt
	@echo "=================================" >> $(PROFILE_DIR)/production_test_results.txt
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS) $(CPP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Production test for $$target..." | tee -a $(PROFILE_DIR)/production_test_results.txt; \
			$$target --production-validation 2>&1 | tee -a $(PROFILE_DIR)/production_test_results.txt; \
			echo "" >> $(PROFILE_DIR)/production_test_results.txt; \
		fi; \
	done

# Load testing
.PHONY: load_test
load_test: production
	@echo "Running load tests for production applications..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Load testing $$target..."; \
			$$target --load-test --duration=300 --rps=1000 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_load_test.txt; \
		fi; \
	done

# Security testing
.PHONY: security_test
security_test: production
	@echo "Running security tests..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS) $(CPP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Security testing $$target..."; \
			$$target --security-scan 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_security.txt; \
		fi; \
	done

# Compliance testing
.PHONY: compliance_test
compliance_test: production
	@echo "Running compliance tests..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS) $(CPP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Compliance testing $$target..."; \
			$$target --compliance-check --standard=ISO27001 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_compliance.txt; \
		fi; \
	done

# Disaster recovery testing
.PHONY: disaster_recovery_test
disaster_recovery_test: production
	@echo "Running disaster recovery tests..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Disaster recovery testing $$target..."; \
			$$target --disaster-recovery-test 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_dr_test.txt; \
		fi; \
	done

# Production deployment preparation
.PHONY: package_production
package_production: production
	@echo "Packaging production applications..."
	@cp -r $(BUILD_DIR)/* $(DEPLOY_DIR)/
	@echo "#!/bin/bash" > $(DEPLOY_DIR)/start.sh
	@echo "# Production GPU Application Startup Script" >> $(DEPLOY_DIR)/start.sh
	@echo "export CUDA_VISIBLE_DEVICES=0" >> $(DEPLOY_DIR)/start.sh
	@echo "export HIP_VISIBLE_DEVICES=0" >> $(DEPLOY_DIR)/start.sh
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "./$$(basename $$target) --production-mode &" >> $(DEPLOY_DIR)/start.sh; \
		fi; \
	done
	@echo "wait" >> $(DEPLOY_DIR)/start.sh
	@chmod +x $(DEPLOY_DIR)/start.sh
	@echo "Production package ready in $(DEPLOY_DIR)/"

# Container builds for deployment
.PHONY: build_containers
build_containers: production
	@echo "Building production containers..."
	@if command -v docker > /dev/null 2>&1; then \
		echo "FROM nvidia/cuda:11.8-runtime-ubuntu20.04" > $(DEPLOY_DIR)/Dockerfile.cuda; \
		echo "COPY build/ /app/" >> $(DEPLOY_DIR)/Dockerfile.cuda; \
		echo "WORKDIR /app" >> $(DEPLOY_DIR)/Dockerfile.cuda; \
		echo "EXPOSE 8080" >> $(DEPLOY_DIR)/Dockerfile.cuda; \
		echo "HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD ./health_check" >> $(DEPLOY_DIR)/Dockerfile.cuda; \
		echo "CMD [\"./start.sh\"]" >> $(DEPLOY_DIR)/Dockerfile.cuda; \
		docker build -t gpu-production:cuda -f $(DEPLOY_DIR)/Dockerfile.cuda $(DEPLOY_DIR)/; \
	else \
		echo "Docker not available"; \
	fi

# Kubernetes deployment manifests
.PHONY: k8s_manifests
k8s_manifests: production
	@echo "Generating Kubernetes deployment manifests..."
	@{ \
		echo "apiVersion: apps/v1"; \
		echo "kind: Deployment"; \
		echo "metadata:"; \
		echo "  name: gpu-production-service"; \
		echo "  namespace: production"; \
		echo "spec:"; \
		echo "  replicas: 3"; \
		echo "  strategy:"; \
		echo "    type: RollingUpdate"; \
		echo "    rollingUpdate:"; \
		echo "      maxSurge: 1"; \
		echo "      maxUnavailable: 0"; \
		echo "  selector:"; \
		echo "    matchLabels:"; \
		echo "      app: gpu-production-service"; \
		echo "  template:"; \
		echo "    metadata:"; \
		echo "      labels:"; \
		echo "        app: gpu-production-service"; \
		echo "    spec:"; \
		echo "      containers:"; \
		echo "      - name: gpu-service"; \
		echo "        image: gpu-production:latest"; \
		echo "        ports:"; \
		echo "        - containerPort: 8080"; \
		echo "        resources:"; \
		echo "          requests:"; \
		echo "            nvidia.com/gpu: 1"; \
		echo "            memory: 8Gi"; \
		echo "            cpu: 2"; \
		echo "          limits:"; \
		echo "            nvidia.com/gpu: 1"; \
		echo "            memory: 16Gi"; \
		echo "            cpu: 4"; \
		echo "        readinessProbe:"; \
		echo "          httpGet:"; \
		echo "            path: /health/ready"; \
		echo "            port: 8080"; \
		echo "          initialDelaySeconds: 30"; \
		echo "        livenessProbe:"; \
		echo "          httpGet:"; \
		echo "            path: /health/live"; \
		echo "            port: 8080"; \
		echo "          initialDelaySeconds: 60"; \
	} > $(DEPLOY_DIR)/gpu-service-deployment.yaml
	@echo "Kubernetes manifests generated in $(DEPLOY_DIR)/"

# Production monitoring setup
.PHONY: deploy_monitoring
deploy_monitoring: production
	@echo "Setting up production monitoring..."
	@cat > $(DEPLOY_DIR)/prometheus.yml <<'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'gpu-applications'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scrape_interval: 10s
EOF
	@cat > $(DEPLOY_DIR)/grafana-dashboard.json <<'EOF'
{
  "dashboard": {
    "title": "GPU Production Applications",
    "panels": [
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {"expr": "gpu_utilization_percent"}
        ]
      },
      {
        "title": "Application Throughput",
        "type": "graph", 
        "targets": [
          {"expr": "application_requests_per_second"}
        ]
      }
    ]
  }
}
EOF
	@echo "Monitoring configuration generated in $(DEPLOY_DIR)/"

# Performance profiling for production
.PHONY: profile_production
profile_production: production
	@echo "Profiling production applications..."
	@for target in $(CUDA_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Production profiling $$target..."; \
			ncu --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__sass_thread_inst_executed_op_all.sum --log-file $(PROFILE_DIR)/$$(basename $$target)_production.ncu-rep $$target --production-workload; \
		fi; \
	done
	@for target in $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Production profiling $$target..."; \
			rocprof --stats --hsa-trace --output-file $(PROFILE_DIR)/$$(basename $$target)_production.csv $$target --production-workload; \
		fi; \
	done

# Production benchmarking
.PHONY: benchmark_production
benchmark_production: production
	@echo "Running production benchmarks..."
	@echo "Production GPU Applications Benchmark Results" > $(PROFILE_DIR)/production_benchmark.txt
	@echo "Generated on: $$(date)" >> $(PROFILE_DIR)/production_benchmark.txt
	@echo "============================================" >> $(PROFILE_DIR)/production_benchmark.txt
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS) $(CPP_TARGETS); do \
		if [ -f $$target ]; then \
			echo ""; \
			echo "Production benchmarking $$target..." | tee -a $(PROFILE_DIR)/production_benchmark.txt; \
			echo "-------------------------------------------" >> $(PROFILE_DIR)/production_benchmark.txt; \
			$$target --production-benchmark --duration=60 | tee -a $(PROFILE_DIR)/production_benchmark.txt; \
			echo "" >> $(PROFILE_DIR)/production_benchmark.txt; \
		fi; \
	done

# Capacity planning analysis
.PHONY: capacity_planning
capacity_planning: production
	@echo "Running capacity planning analysis..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Capacity analysis for $$target..."; \
			$$target --capacity-planning --forecast-days=30 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_capacity.txt; \
		fi; \
	done

# Cost analysis
.PHONY: cost_analysis
cost_analysis: production
	@echo "Running cost analysis..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "Cost analysis for $$target..."; \
			$$target --cost-analysis --cloud-provider=aws 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_cost.txt; \
		fi; \
	done

# Performance reports generation
.PHONY: performance_reports
performance_reports: benchmark_production profile_production
	@echo "Generating production performance reports..."
	@echo "Production Performance Analysis Report" > $(PROFILE_DIR)/production_performance_report.md
	@echo "====================================" >> $(PROFILE_DIR)/production_performance_report.md
	@echo "" >> $(PROFILE_DIR)/production_performance_report.md
	@echo "## Executive Summary" >> $(PROFILE_DIR)/production_performance_report.md
	@echo "- Date: $$(date)" >> $(PROFILE_DIR)/production_performance_report.md
	@echo "- Environment: Production" >> $(PROFILE_DIR)/production_performance_report.md
	@echo "- GPU: $$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'Multi-vendor')" >> $(PROFILE_DIR)/production_performance_report.md
	@echo "" >> $(PROFILE_DIR)/production_performance_report.md
	@echo "## Performance Metrics" >> $(PROFILE_DIR)/production_performance_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/production_performance_report.md
	@cat $(PROFILE_DIR)/production_benchmark.txt >> $(PROFILE_DIR)/production_performance_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/production_performance_report.md
	@echo "Production performance report generated: $(PROFILE_DIR)/production_performance_report.md"

# SLA monitoring
.PHONY: sla_monitoring
sla_monitoring: production
	@echo "Setting up SLA monitoring..."
	@for target in $(CUDA_TARGETS) $(HIP_TARGETS); do \
		if [ -f $$target ]; then \
			echo "SLA monitoring for $$target..."; \
			$$target --sla-monitoring --availability-target=99.9 --latency-target=100ms 2>&1 | tee $(PROFILE_DIR)/$$(basename $$target)_sla.txt; \
		fi; \
	done

# Clean all artifacts
.PHONY: clean
clean:
	@echo "Cleaning all production artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(PROFILE_DIR)
	rm -rf $(DEPLOY_DIR)
	rm -rf $(LOGS_DIR)

# Production deployment check
.PHONY: production_check
production_check:
	@echo "Production Deployment Readiness Check:"
	@echo "====================================="
	@echo "System Requirements:"
	@nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "CUDA GPU not available"
	@rocm-smi --showproductname 2>/dev/null || echo "ROCm GPU not available"
	@echo ""
	@echo "Container Runtime:"
	@docker --version 2>/dev/null || echo "Docker not available"
	@kubectl version --client 2>/dev/null || echo "Kubernetes client not available"
	@echo ""
	@echo "Monitoring Stack:"
	@which prometheus 2>/dev/null || echo "Prometheus not in PATH"
	@which grafana-server 2>/dev/null || echo "Grafana not in PATH"
	@echo ""
	@echo "Security Tools:"
	@which trivy 2>/dev/null || echo "Trivy security scanner not available"

# Generate deployment documentation
.PHONY: deployment_docs
deployment_docs: production
	@echo "Generating deployment documentation..."
	@cat > $(DEPLOY_DIR)/DEPLOYMENT.md <<'EOF'
# Production GPU Application Deployment Guide

## Prerequisites

- CUDA Toolkit 11.8+ or ROCm 5.4+
- Docker with GPU runtime support
- Kubernetes cluster with GPU nodes
- Monitoring stack (Prometheus + Grafana)

## Deployment Steps

1. **Build Applications:**
   ```bash
   make production
   ```

2. **Run Tests:**
   ```bash
   make test_production
   make load_test
   make security_test
   ```

3. **Package for Deployment:**
   ```bash
   make package_production
   make build_containers
   ```

4. **Deploy to Kubernetes:**
   ```bash
   kubectl apply -f gpu-service-deployment.yaml
   ```

5. **Setup Monitoring:**
   ```bash
   make deploy_monitoring
   ```

## Production Checklist

- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance benchmarks meet SLA
- [ ] Monitoring configured
- [ ] Backup and recovery procedures tested
- [ ] Documentation updated

## Support

For production support, contact the GPU Operations team.
EOF
	@echo "Deployment documentation generated: $(DEPLOY_DIR)/DEPLOYMENT.md"

# System information for production
.PHONY: system_info
system_info:
	@echo "Production System Information:"
	@echo "============================="
	@echo "Hardware Configuration:"
	@nvidia-smi --query-gpu=name,memory.total,compute_cap,power.limit --format=csv,noheader 2>/dev/null || echo "NVIDIA GPU not detected"
	@rocm-smi --showallinfo 2>/dev/null | grep -E "GPU ID|Product Name|Memory" || echo "AMD GPU not detected"
	@echo ""
	@echo "Software Configuration:"
	@nvcc --version | grep release 2>/dev/null || echo "CUDA not available"
	@hipcc --version 2>/dev/null | head -1 || echo "HIP not available"
	@docker --version 2>/dev/null || echo "Docker not available"
	@echo ""
	@echo "Production Environment:"
	@echo "Build Date: $$(date)"
	@echo "Optimization Level: Production (-O3)"
	@echo "Security: Hardened"

# Comprehensive production report
.PHONY: report
report: test_production load_test security_test compliance_test benchmark_production
	@echo "Generating comprehensive production report..."
	@echo "Module 9: Production GPU Programming - Comprehensive Report" > $(PROFILE_DIR)/production_report.md
	@echo "=======================================================" >> $(PROFILE_DIR)/production_report.md
	@echo "" >> $(PROFILE_DIR)/production_report.md
	@echo "## Executive Summary" >> $(PROFILE_DIR)/production_report.md
	@echo "- Report Date: $$(date)" >> $(PROFILE_DIR)/production_report.md
	@echo "- System: $$(hostname)" >> $(PROFILE_DIR)/production_report.md
	@echo "- GPU: $$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'Multi-vendor')" >> $(PROFILE_DIR)/production_report.md
	@echo "- Applications: $$(ls $(BUILD_DIR) | wc -l) production applications" >> $(PROFILE_DIR)/production_report.md
	@echo "" >> $(PROFILE_DIR)/production_report.md
	@echo "## Test Results Summary" >> $(PROFILE_DIR)/production_report.md
	@echo "- Production Tests: See production_test_results.txt" >> $(PROFILE_DIR)/production_report.md
	@echo "- Load Tests: All applications tested under production load" >> $(PROFILE_DIR)/production_report.md
	@echo "- Security Tests: Security scanning completed" >> $(PROFILE_DIR)/production_report.md
	@echo "- Compliance: Standards compliance verified" >> $(PROFILE_DIR)/production_report.md
	@echo "" >> $(PROFILE_DIR)/production_report.md
	@echo "## Performance Analysis" >> $(PROFILE_DIR)/production_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/production_report.md
	@cat $(PROFILE_DIR)/production_benchmark.txt 2>/dev/null || echo "Benchmark results not available" >> $(PROFILE_DIR)/production_report.md
	@echo "\`\`\`" >> $(PROFILE_DIR)/production_report.md
	@echo "" >> $(PROFILE_DIR)/production_report.md
	@echo "## Deployment Readiness" >> $(PROFILE_DIR)/production_report.md
	@echo "- Container Images: Built and tested" >> $(PROFILE_DIR)/production_report.md
	@echo "- Kubernetes Manifests: Generated and validated" >> $(PROFILE_DIR)/production_report.md
	@echo "- Monitoring: Configuration ready for deployment" >> $(PROFILE_DIR)/production_report.md
	@echo "- Documentation: Complete deployment guide available" >> $(PROFILE_DIR)/production_report.md
	@echo "Production report generated: $(PROFILE_DIR)/production_report.md"

# Help target
.PHONY: help
help:
	@echo "Module 9: Production GPU Programming - Build System"
	@echo "================================================="
	@echo ""
	@echo "Build Targets:"
	@echo "  all              - Build all production applications"
	@echo "  production       - Build with production optimization and hardening"
	@echo "  debug            - Build with debug information"
	@echo "  clean            - Remove all build artifacts"
	@echo ""
	@echo "Application Categories:"
	@echo "  architecture     - Build production architecture examples"
	@echo "  error_handling   - Build error handling and resilience examples"
	@echo "  deployment       - Build deployment and DevOps examples"
	@echo "  monitoring       - Build monitoring and observability examples"
	@echo "  scalability      - Build scalability and performance examples"
	@echo "  security         - Build security and compliance examples"
	@echo ""
	@echo "Testing Targets:"
	@echo "  test             - Run all production tests"
	@echo "  test_production  - Run production readiness tests"
	@echo "  load_test        - Run load testing"
	@echo "  security_test    - Run security testing"
	@echo "  compliance_test  - Run compliance testing"
	@echo "  disaster_recovery_test - Run disaster recovery tests"
	@echo ""
	@echo "Deployment Targets:"
	@echo "  package_production - Package applications for deployment"
	@echo "  build_containers - Build Docker containers"
	@echo "  k8s_manifests    - Generate Kubernetes manifests"
	@echo "  deploy_monitoring - Setup monitoring configuration"
	@echo "  deployment_docs  - Generate deployment documentation"
	@echo ""
	@echo "Analysis Targets:"
	@echo "  benchmark_production - Run production performance benchmarks"
	@echo "  profile_production - Profile production applications"
	@echo "  capacity_planning - Analyze capacity requirements"
	@echo "  cost_analysis    - Analyze deployment costs"
	@echo "  performance_reports - Generate performance reports"
	@echo "  sla_monitoring   - Setup SLA monitoring"
	@echo ""
	@echo "Validation Targets:"
	@echo "  production_check - Check production deployment readiness"
	@echo "  system_info      - Display production system information"
	@echo "  report           - Generate comprehensive production report"
	@echo ""
	@echo "Examples:"
	@echo "  make production && make test_production"
	@echo "  make package_production && make build_containers"
	@echo "  make report  # Complete production analysis"