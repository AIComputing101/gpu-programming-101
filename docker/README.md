# GPU Programming 101 - Docker Development Environment

This directory contains Docker configurations for comprehensive GPU programming development environments supporting both NVIDIA CUDA and AMD ROCm platforms.

## 🚀 Latest Versions (2025)

- **CUDA**: 12.9.1 (Latest stable release)
- **ROCm**: 7.0 (Latest stable release) 
- **Ubuntu**: 22.04 LTS
- **Nsight Tools**: 2025.1.1

## 🚀 Quick Start

### Prerequisites
- Docker 20.10+ with GPU support
- Docker Compose 2.0+
- **For NVIDIA**: NVIDIA drivers + nvidia-container-toolkit
- **For AMD**: AMD drivers + ROCm support

### 1. Build Containers
```bash
# Build CUDA container
./docker/scripts/build.sh cuda

# Build ROCm container  
./docker/scripts/build.sh rocm

# Build both
./docker/scripts/build.sh --all
```

### 2. Run Development Environment
```bash
# Auto-detect GPU and run appropriate container
./docker/scripts/run.sh --auto

# Or specify platform
./docker/scripts/run.sh cuda    # For NVIDIA GPUs
./docker/scripts/run.sh rocm    # For AMD GPUs
```

## 📁 Directory Structure

```
docker/
├── README.md                 # This file
├── docker-compose.yml        # Multi-platform orchestration
├── cuda/
│   └── Dockerfile           # NVIDIA CUDA development image
├── rocm/
│   └── Dockerfile           # AMD ROCm development image
└── scripts/
    ├── build.sh             # Build automation script
    └── run.sh               # Container run script
```

## 🐳 Available Containers

### CUDA Development Container
**Image**: `gpu-programming-101:cuda`  
**Base**: `nvidia/cuda:12.9.1-devel-ubuntu22.04`

**Features**:
- CUDA 12.9.1 with development tools
- NVIDIA Nsight Systems & Compute profilers
- Python 3 with scientific libraries
- GPU monitoring and debugging tools

**GPU Requirements**:
- NVIDIA GPU with compute capability 3.5+
- NVIDIA drivers 535+
- nvidia-container-toolkit

### ROCm Development Container
**Image**: `gpu-programming-101:rocm`  
**Base**: `rocm/dev-ubuntu-22.04:7.0`

**Features**:
- ROCm 7.0 with HIP development environment
- Cross-platform GPU programming (AMD/NVIDIA)
- ROCm profiling tools (rocprof, roctracer)
- Python 3 with scientific libraries

**GPU Requirements**:
- AMD GPU with ROCm support (RX 580+, MI series)
- AMD drivers with ROCm 7.0+

## 🔧 Container Usage

### Interactive Development
```bash
# Start CUDA container
./docker/scripts/run.sh cuda
[CUDA-DEV] /workspace/gpu-programming-101 $ 

# Test GPU access
[CUDA-DEV] /workspace/gpu-programming-101 $ nvidia-smi
[CUDA-DEV] /workspace/gpu-programming-101 $ /workspace/test-gpu.sh

# Build and run examples
[CUDA-DEV] /workspace/gpu-programming-101 $ cd modules/module1/examples
[CUDA-DEV] /workspace/gpu-programming-101/modules/module1/examples $ make
[CUDA-DEV] /workspace/gpu-programming-101/modules/module1/examples $ ./01_vector_addition_cuda
```

### Background Services
```bash
# Run in detached mode
./docker/scripts/run.sh cuda --detach

# Check status
docker ps

# Access running container
docker exec -it gpu101-cuda-dev bash

# View logs
docker logs gpu101-cuda-dev
```

## 🛠️ Build Options

### Basic Build
```bash
./docker/scripts/build.sh cuda
./docker/scripts/build.sh rocm
./docker/scripts/build.sh --all
```

### Advanced Build Options
```bash
# Clean build (remove existing images)
./docker/scripts/build.sh --clean --all

# Build without cache
./docker/scripts/build.sh --no-cache cuda

# Pull latest base images
./docker/scripts/build.sh --pull --all

# Use docker-compose
./docker/scripts/build.sh --compose cuda
```

## 🚀 Docker Compose Usage

### Start Services
```bash
cd docker/

# Start CUDA development environment
docker-compose up cuda-dev

# Start ROCm development environment  
docker-compose up rocm-dev

# Start in background
docker-compose up -d cuda-dev

# Start both platforms
docker-compose up cuda-dev rocm-dev
```

### Manage Services
```bash
# Stop services
docker-compose down

# View logs
docker-compose logs cuda-dev

# Execute commands in running service
docker-compose exec cuda-dev bash

# Rebuild services
docker-compose build cuda-dev
```

## 🔍 Testing and Debugging

### Container Health Checks
Each container includes built-in health checks and test scripts:

```bash
# Inside container - test GPU environment
/workspace/test-gpu.sh

# Test specific modules
cd modules/module1/examples
make test

# GPU info commands
nvidia-smi          # NVIDIA
rocm-smi           # AMD  
nvcc --version     # CUDA compiler
hipcc --version    # HIP compiler
```

### Debugging Tools

**CUDA Debugging**:
```bash
# Inside CUDA container
cuda-gdb ./program              # GPU debugger
ncu --metrics gpu__time_duration.avg ./program  # Profiler
nsys profile --trace=cuda ./program            # System tracer
```

**ROCm Debugging**:
```bash
# Inside ROCm container
rocgdb ./program               # GPU debugger
rocprof --hip-trace ./program  # HIP tracer
rocprof --stats ./program      # Performance stats
```

### Performance Profiling
```bash
# CUDA performance analysis
ncu --set full ./05_performance_comparison
nsys profile -t cuda ./05_performance_comparison

# ROCm performance analysis
rocprof --hip-trace --stats ./02_vector_addition_hip
```

## 💾 Volume Mounts

### Project Files
- **Host**: `./` (project root)
- **Container**: `/workspace/gpu-programming-101`
- **Access**: Read-Write

### Persistent Home Directories
- **CUDA**: `cuda-home` → `/root`
- **ROCm**: `rocm-home` → `/root`
- **Cache**: `cuda-cache`, `rocm-cache` → `/root/.cache`

## 🔧 Environment Variables

### CUDA Container
```bash
CUDA_HOME=/usr/local/cuda
PATH=/usr/local/cuda/bin:$PATH
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
NVIDIA_VISIBLE_DEVICES=all
```

### ROCm Container
```bash
ROCM_PATH=/opt/rocm
HIP_PATH=/opt/rocm/hip
HIP_PLATFORM=amd
HSA_OVERRIDE_GFX_VERSION=11.0.0
```

## 🛡️ Security Considerations

### GPU Device Access
```bash
# NVIDIA containers require GPU runtime
--runtime=nvidia --gpus all

# AMD containers require device access
--device=/dev/kfd --device=/dev/dri
--security-opt seccomp=unconfined
```

### File Permissions
- Containers run as root for GPU access
- Project files maintain host permissions via bind mounts
- Use named volumes for persistent container data

## 🐛 Troubleshooting

### Common Issues

**"No GPU detected"**
```bash
# Check host GPU access
nvidia-smi  # For NVIDIA
rocm-smi   # For AMD

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.9.1-base nvidia-smi

# Check container runtime
docker run --rm --device=/dev/kfd rocm/dev-ubuntu-22.04:7.0 rocminfo
```

**"Container build fails"**
```bash
# Clean build cache
docker system prune -a

# Update Docker
sudo apt update && sudo apt upgrade docker-ce docker-compose

# Check base image availability
docker pull nvidia/cuda:12.9.1-devel-ubuntu22.04
docker pull rocm/dev-ubuntu-22.04:7.0
```

**"Permission denied errors"**
```bash
# Fix script permissions
chmod +x docker/scripts/*.sh

# Check Docker group membership
sudo usermod -aG docker $USER
newgrp docker
```

**"Port already in use"**
```bash
# Check running containers
docker ps

# Use different port
./docker/scripts/run.sh cuda --port 8899

# Stop conflicting services
docker stop $(docker ps -q)
```

### Container Debugging
```bash
# Enter running container
docker exec -it gpu101-cuda-dev bash

# Check container logs
docker logs gpu101-cuda-dev

# Inspect container configuration
docker inspect gpu101-cuda-dev

# Monitor resource usage
docker stats gpu101-cuda-dev
```

## 📚 Learning Workflows

### Beginner Workflow
1. **Setup**: `./docker/scripts/build.sh cuda`
2. **Start**: `./docker/scripts/run.sh cuda`
3. **Test**: `/workspace/test-gpu.sh`
4. **Learn**: `cd modules/module1 && cat README.md`
5. **Practice**: `cd examples && make && ./01_vector_addition_cuda`

### Development Workflow  
1. **Background**: `./docker/scripts/run.sh cuda --detach`
2. **Code**: Edit files in your host IDE
3. **Test**: Compile and run in container
4. **Debug**: Use integrated debugging tools
5. **Profile**: Analyze performance with profilers

### Multi-Platform Testing
```bash
# Test CUDA implementation
./docker/scripts/run.sh cuda
make module1 && cd modules/module1/examples && make test

# Test HIP implementation  
./docker/scripts/run.sh rocm
cd modules/module1/examples && make vector_add_hip && ./vector_add_hip
```

## 🤝 Contributing

When contributing Docker improvements:

1. **Test both platforms**: Verify changes work on CUDA and ROCm
2. **Document changes**: Update this README
3. **Version control**: Tag container versions appropriately
4. **Security review**: Ensure no vulnerabilities introduced

## 📖 Additional Resources

- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [ROCm Container Guide](https://rocmdocs.amd.com/en/latest/deploy/docker.html)
- [CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda)
- [ROCm Docker Hub](https://hub.docker.com/r/rocm/dev-ubuntu-22.04)

---

**Happy containerized GPU programming!** 🚀🐳