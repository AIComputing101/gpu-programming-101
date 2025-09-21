# GPU Programming 101 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.9.1-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![ROCm](https://img.shields.io/badge/ROCm-7.0-red?logo=amd)](https://rocmdocs.amd.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![Examples](https://img.shields.io/badge/Examples-71-green)](modules/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/features/actions)

**A comprehensive, hands-on educational project for mastering GPU programming with CUDA and HIP**

*From beginner fundamentals to production-ready optimization techniques*

## 📑 Table of Contents

- [📋 Project Overview](#-project-overview)
- [🏗️ GPU Programming Architecture](#️-gpu-programming-architecture)
- [✨ Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
- [🎯 Learning Path](#-learning-path)
- [📚 Modules](#-modules)
- [🛠️ Prerequisites](#️-prerequisites)
- [🐳 Docker Development](#-docker-development)
- [🔧 Build System](#-build-system)
- [📊 Performance Expectations](#-performance-expectations)
- [🐛 Troubleshooting](#-troubleshooting)
- [📖 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 📋 Project Overview

**GPU Programming 101** is a complete educational resource for learning modern GPU programming. This project provides:

- **9 comprehensive modules** covering beginner to expert topics
- **71 working code examples** in both CUDA and HIP
- **Cross-platform support** for NVIDIA and AMD GPUs  
- **Production-ready development environment** with Docker
- **Professional tooling** including profilers, debuggers, and CI/CD

Perfect for students, researchers, and developers looking to master GPU computing.

## 🏗️ GPU Programming Architecture

Understanding how GPU programming works from high-level code to hardware execution is crucial for effective GPU development. This section provides a comprehensive overview of the CUDA and HIP ROCm software-hardware stack.

### Architecture Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                APPLICATION LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  High-Level Code (C++/CUDA/HIP)                                                   │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐    │
│  │   CUDA C++ Code     │    │    HIP C++ Code     │    │   OpenCL/SYCL       │    │
│  │   (.cu files)       │    │   (.hip files)      │    │   (Cross-platform)   │    │
│  │                     │    │                     │    │                     │    │
│  │ __global__ kernels  │    │ __global__ kernels  │    │ kernel functions    │    │
│  │ cudaMalloc()        │    │ hipMalloc()         │    │ clCreateBuffer()    │    │
│  │ cudaMemcpy()        │    │ hipMemcpy()         │    │ clEnqueueNDRange()  │    │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              COMPILATION LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Compiler Frontend                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐    │
│  │      NVCC           │    │      HIP Clang      │    │    LLVM/Clang       │    │
│  │  (NVIDIA Compiler)  │    │   (AMD Compiler)    │    │   (Open Standard)   │    │
│  │                     │    │                     │    │                     │    │
│  │ • Parse CUDA syntax │    │ • Parse HIP syntax  │    │ • Parse OpenCL/SYCL │    │
│  │ • Host/Device split │    │ • Host/Device split │    │ • Generate SPIR-V   │    │
│  │ • Generate PTX      │    │ • Generate GCN ASM  │    │ • Target backends   │    │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           INTERMEDIATE REPRESENTATION                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐    │
│  │        PTX          │    │      GCN ASM        │    │      SPIR-V         │    │
│  │ (Parallel Thread    │    │  (Graphics Core     │    │  (Standard Portable │    │
│  │  Execution)         │    │   Next Assembly)    │    │   IR - Vulkan)      │    │
│  │                     │    │                     │    │                     │    │
│  │ • Virtual ISA       │    │ • AMD GPU ISA       │    │ • Cross-platform    │    │
│  │ • Device agnostic   │    │ • RDNA/CDNA arch    │    │ • Vendor neutral    │    │
│  │ • JIT compilation   │    │ • Direct execution  │    │ • Multiple targets  │    │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               DRIVER LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐    │
│  │    CUDA Driver      │    │     ROCm Driver     │    │   OpenCL Driver     │    │
│  │                     │    │                     │    │                     │    │
│  │ • PTX → SASS JIT    │    │ • GCN → Machine     │    │ • SPIR-V → Native   │    │
│  │ • Memory management │    │ • Memory management │    │ • Memory management │    │
│  │ • Kernel launch     │    │ • Kernel launch     │    │ • Kernel launch     │    │
│  │ • Context mgmt      │    │ • Context mgmt      │    │ • Context mgmt      │    │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              HARDWARE LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐                               │
│  │    NVIDIA GPU       │    │      AMD GPU        │                               │
│  │                     │    │                     │                               │
│  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    ┌─────────────────────┐    │
│  │ │   SM (Cores)    │ │    │ │   CU (Cores)    │ │    │   Intel Xe Cores    │    │
│  │ │ ┌─────────────┐ │ │    │ │ ┌─────────────┐ │ │    │ ┌─────────────────┐ │    │
│  │ │ │FP32 | INT32 │ │ │    │ │ │FP32 | INT32 │ │ │    │ │  Vector Engines │ │    │
│  │ │ │FP64 | BF16  │ │ │    │ │ │FP64 | BF16  │ │ │    │ │  Matrix Engines │ │    │
│  │ │ │Tensor Cores │ │ │    │ │ │Matrix Cores │ │ │    │ │  Ray Tracing    │ │    │
│  │ │ └─────────────┘ │ │    │ │ └─────────────┘ │ │    │ └─────────────────┘ │    │
│  │ └─────────────────┘ │    │ └─────────────────┘ │    └─────────────────────┘    │
│  │                     │    │                     │                               │
│  │ Memory Hierarchy:   │    │ Memory Hierarchy:   │    Memory Hierarchy:          │
│  │ • L1 Cache (KB)     │    │ • L1 Cache (KB)     │    • L1 Cache                 │
│  │ • L2 Cache (MB)     │    │ • L2 Cache (MB)     │    • L2 Cache                 │
│  │ • Global Mem (GB)   │    │ • Global Mem (GB)   │    • Global Memory            │
│  │ • Shared Memory     │    │ • LDS (Local Data   │    • Shared Local Memory      │
│  │ • Constant Memory   │    │   Store)            │    • Constant Memory          │
│  │ • Texture Memory    │    │ • Constant Memory   │                               │
│  └─────────────────────┘    └─────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Compilation Pipeline Deep Dive

#### 1. **Source Code → Frontend Parsing**
- **CUDA**: NVCC separates host (CPU) and device (GPU) code, parses CUDA extensions
- **HIP**: Clang-based compiler with HIP runtime API that maps to either CUDA or ROCm
- **OpenCL/SYCL**: LLVM-based compilation with cross-platform intermediate representation

#### 2. **Frontend → Intermediate Representation**
```
High-Level Code                    Intermediate Form
─────────────────                 ───────────────────
__global__ void kernel()    →     PTX (NVIDIA)
{                                 GCN Assembly (AMD)  
    int id = threadIdx.x;         SPIR-V (OpenCL/Vulkan)
    output[id] = input[id] * 2;   LLVM IR (SYCL)
}
```

#### 3. **Runtime Compilation & Optimization**
- **NVIDIA**: PTX → SASS (GPU-specific machine code) via JIT compilation
- **AMD**: GCN Assembly → GPU microcode via ROCm runtime
- **Optimizations**: Register allocation, memory coalescing, instruction scheduling

#### 4. **Hardware Execution Model**

| Abstraction Level | NVIDIA Term | AMD Term | Description |
|------------------|-------------|----------|-------------|
| **Thread** | Thread | Work-item | Single execution unit |
| **Thread Group** | Warp (32 threads) | Wavefront (64 threads) | SIMD execution group |
| **Thread Block** | Block | Work-group | Shared memory + synchronization |
| **Grid** | Grid | NDRange | Collection of all thread blocks |

#### 5. **Memory Architecture Mapping**

```
Programming Model              Hardware Implementation
─────────────────              ─────────────────────────
Global Memory        →         GPU DRAM (HBM/GDDR)
Shared Memory        →         On-chip SRAM (48-164KB per SM/CU)
Local Memory         →         GPU DRAM (spilled registers)
Constant Memory      →         Cached read-only GPU DRAM
Texture Memory       →         Cached GPU DRAM with interpolation
Registers            →         On-chip register file (32K-64K per SM/CU)
```

### Performance Implications

Understanding this architecture helps optimize GPU code:

1. **Memory Coalescing**: Access patterns that align with hardware memory buses
2. **Occupancy**: Balancing registers, shared memory, and thread blocks per SM/CU
3. **Divergence**: Minimizing different execution paths within warps/wavefronts
4. **Latency Hiding**: Using enough threads to hide memory access latency
5. **Memory Hierarchy**: Optimal use of each memory type based on access patterns

This architectural knowledge is essential for writing efficient GPU code and is covered progressively throughout our modules.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Complete Curriculum** | 9 progressive modules from basics to advanced topics |
| 💻 **Cross-Platform** | Full CUDA and HIP support for NVIDIA and AMD GPUs |
| 🐳 **Docker Ready** | Complete containerized development environment with CUDA 12.9.1 & ROCm 7.0 |
| 🔧 **Production Quality** | Professional build systems, auto-detection, testing, and profiling |
| 📊 **Performance Focus** | Optimization techniques and benchmarking throughout |
| 🌐 **Community Driven** | Open source with comprehensive contribution guidelines |
| 🧪 **Advanced Libraries** | Support for Thrust, MIOpen, and production ML frameworks |

## 🚀 Quick Start

### Option 1: Docker (Recommended)
Get started immediately without installing CUDA/ROCm on your host system:

```bash
# Clone the repository
git clone https://github.com/AIComputing101/gpu-programming-101.git
cd gpu-programming-101

# Auto-detect your GPU and start development environment
./docker/scripts/run.sh --auto

# Inside container: verify GPU access and start learning
/workspace/test-gpu.sh
cd modules/module1 && make && ./build/01_vector_addition_cuda
```

### Option 2: Native Installation
For direct system installation:

```bash
# Prerequisites: CUDA 12.0+ or ROCm 7.0+, GCC 9+, Make

# Clone and build
git clone https://github.com/AIComputing101/gpu-programming-101.git
cd gpu-programming-101

# Verify your setup
make check-system

# Build and run first example
make module1
cd modules/module1/examples
./01_vector_addition_cuda
```

## 🎯 Learning Path

Choose your track based on your experience level:

- **👶 Beginner Track** (Modules 1-3) - GPU fundamentals, memory management, first kernels
- **🔥 Intermediate Track** (Modules 4-5) - Advanced programming, performance optimization  
- **🚀 Advanced Track** (Modules 6-9) - Parallel algorithms, domain applications, production deployment

*Each track builds on the previous one, so start with the appropriate level for your background.*

## 📚 Modules

Our comprehensive curriculum progresses from fundamental concepts to production-ready optimization techniques:

| Module | Level | Duration | Focus Area | Key Topics | Examples |
|--------|-------|----------|------------|------------|----------|
| [**Module 1**](modules/module1/) | 👶 Beginner | 4-6h | **GPU Fundamentals** | Architecture, Memory, First Kernels | 13 |
| [**Module 2**](modules/module2/) | 👶→🔥 | 6-8h | **Memory Optimization** | Coalescing, Shared Memory, Texture | 10 |
| [**Module 3**](modules/module3/) | 🔥 Intermediate | 6-8h | **Execution Models** | Warps, Occupancy, Synchronization | 12 |
| [**Module 4**](modules/module4/) | 🔥→🚀 | 8-10h | **Advanced Programming** | Streams, Multi-GPU, Unified Memory | 9 |
| [**Module 5**](modules/module5/) | 🚀 Advanced | 6-8h | **Performance Engineering** | Profiling, Bottleneck Analysis | 5 |
| [**Module 6**](modules/module6/) | 🚀 Advanced | 8-10h | **Parallel Algorithms** | Reduction, Scan, Convolution | 10 |
| [**Module 7**](modules/module7/) | 🚀 Expert | 8-10h | **Algorithmic Patterns** | Sorting, Graph Algorithms | 4 |
| [**Module 8**](modules/module8/) | 🚀 Expert | 10-12h | **Domain Applications** | ML, Scientific Computing | 4 |
| [**Module 9**](modules/module9/) | 🚀 Expert | 6-8h | **Production Deployment** | Libraries, Integration, Scaling | 4 |

**📈 Progressive Learning Path: 71 Examples • 50+ Hours • Beginner to Expert**

### Learning Progression

```
Module 1: Hello GPU World          Module 6: Parallel Algorithms
    ↓                                 ↓
Module 2: Memory Mastery          Module 7: Advanced Patterns  
    ↓                                 ↓
Module 3: Execution Deep Dive     Module 8: Real Applications
    ↓                                 ↓
Module 4: Advanced Features       Module 9: Production Ready
    ↓                             
Module 5: Performance Tuning     
```

**[📚 View All Modules →](modules/)**

## 🛠️ Prerequisites

### Hardware Requirements

#### NVIDIA GPU Systems
- **Minimum GPU**: GTX 1060 6GB, GTX 1650, RTX 2060 or better
- **Recommended GPU**: RTX 3070/4070 (12GB+), RTX 3080/4080 (16GB+) 
- **Professional/Advanced**: RTX 4090 (24GB), RTX A6000 (48GB), Tesla/Quadro series
- **Architecture Support**: Maxwell, Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper
- **Compute Capability**: 5.0+ (Maxwell architecture or newer)

#### AMD GPU Systems  
- **Minimum GPU**: RX 580 8GB, RX 6600, RX 7600 or better
- **Recommended GPU**: RX 6700 XT/7700 XT (12GB+), RX 6800 XT/7800 XT (16GB+)
- **Professional/Advanced**: RX 7900 XTX (24GB), Radeon PRO W7800 (48GB), Instinct MI series
- **Architecture Support**: RDNA2, RDNA3, RDNA4, GCN 5.0+, CDNA series
- **ROCm Compatibility**: Officially supported AMD GPUs only

#### System Memory & CPU
- **Minimum RAM**: 16GB system RAM 
- **Recommended RAM**: 32GB+ for advanced modules and multi-GPU setups
- **Professional Setup**: 64GB+ for large-scale scientific computing
- **CPU Requirements**: 
  - **Intel**: Haswell (2013) or newer for PCIe atomics support
  - **AMD**: Zen 1 (2017) or newer for PCIe atomics support
- **Storage**: 20GB+ free space for Docker containers and examples

### Software Requirements

#### Operating System Support
- **Linux** (Recommended): Ubuntu 22.04/24.04 LTS, RHEL 8/9, SLES 15 SP5
- **Windows**: Windows 10/11 with WSL2 recommended for optimal compatibility
- **macOS**: macOS 12+ (Metal Performance Shaders for basic GPU compute)

#### GPU Computing Platforms
- **CUDA Toolkit**: 12.0+ (Docker uses CUDA 12.9.1)
  - **Driver Requirements**: 
    - Linux: 550.54.14+ for CUDA 12.4+
    - Windows: 551.61+ for CUDA 12.4+
- **ROCm Platform**: 7.0+ (Docker uses ROCm 7.0)
  - **Driver Requirements**: Latest AMDGPU-PRO or open-source AMDGPU drivers
  - **Kernel Support**: Linux kernel 5.4+ recommended

#### Development Environment
- **Compilers**:
  - **GCC**: 9.0+ (GCC 11+ recommended for C++17 features)
  - **Clang**: 10.0+ (Clang 14+ recommended)
  - **MSVC**: 2019+ (2022 17.10+ for CUDA 12.4+ support)
- **Build Tools**: Make 4.0+, CMake 3.18+ (optional)
- **Docker**: 20.10+ with GPU runtime support (nvidia-container-toolkit or ROCm containers)

#### Additional Tools (Included in Docker)
- **Profiling**: Nsight Compute, Nsight Systems (NVIDIA), rocprof (AMD)
- **Debugging**: cuda-gdb, rocgdb, compute-sanitizer
- **Libraries**: cuBLAS, cuFFT, rocBLAS, rocFFT (for advanced modules)
- **ML Libraries**: Thrust (NVIDIA), MIOpen (AMD) for deep learning applications
- **System Management**: NVML (NVIDIA), ROCm SMI (AMD) for hardware monitoring

### Performance Expectations by Hardware Tier

| Hardware Tier | Example GPUs | VRAM | Expected Performance | Suitable Modules |
|---------------|--------------|------|---------------------|------------------|
| **Entry Level** | GTX 1060 6GB, RX 580 8GB | 6-8GB | 10-50x CPU speedup | Modules 1-3 |
| **Mid-Range** | RTX 3060 Ti, RX 6700 XT | 12GB | 50-200x CPU speedup | Modules 1-6 |
| **High-End** | RTX 4070 Ti, RX 7800 XT | 16GB | 100-500x CPU speedup | All modules |
| **Professional** | RTX 4090, RX 7900 XTX | 24GB | 200-1000x+ CPU speedup | All modules + research |

### Programming Knowledge
- **C/C++**: Intermediate level (pointers, memory management, basic templates)
- **Parallel Programming**: Basic understanding of threads and synchronization helpful
- **Command Line**: Comfortable with terminal/shell operations
- **Mathematics**: Linear algebra and calculus basics beneficial for advanced modules
- **Version Control**: Basic Git knowledge for contributing

### Network Requirements (Docker Setup)
- **Internet Connection**: Required for initial Docker image downloads (~8GB total)
- **Bandwidth**: 50+ Mbps recommended for efficient container downloads
- **Storage**: Additional 20GB for Docker images and build cache

## 🐳 Docker Development

Experience the full development environment with zero setup:

```bash
# Build development containers
./docker/scripts/build.sh --all

# Start interactive development
./docker/scripts/run.sh cuda    # For NVIDIA GPUs
./docker/scripts/run.sh rocm    # For AMD GPUs
./docker/scripts/run.sh --auto  # Auto-detect GPU type
```

**Docker Benefits:**
- 🎯 Zero host configuration required
- 🔧 Complete development environment (compilers, debuggers, profilers)
- 🌐 Cross-platform testing (test your code on both CUDA and HIP)
- 📦 Isolated and reproducible builds
- 🧹 Easy cleanup when done

**Container Specifications:**
- **CUDA**: NVIDIA CUDA 12.9.1 on Ubuntu 22.04
- **ROCm**: AMD ROCm 7.0 on Ubuntu 24.04 
- **Libraries**: Production-ready toolchains with debugging support

**[📖 Complete Docker Guide →](docker/README.md)**

## 🔧 Build System

Our advanced build system features automatic GPU vendor detection and optimized configurations:

### Project-Wide Commands
```bash
make all           # Build all modules with auto-detection
make test          # Run comprehensive tests  
make clean         # Clean all artifacts
make check-system  # Verify GPU setup and dependencies
make status        # Show module completion status
```

### Module-Specific Commands
```bash
cd modules/module1/examples
make               # Build all examples with vendor auto-detection
make test          # Run module tests
make profile       # Performance profiling
make debug         # Debug builds with extra checks
```

### Advanced Build Features
- **Automatic GPU Detection**: Detects NVIDIA/AMD hardware and builds accordingly
- **Production Optimization**: `-O3`, fast math, architecture-specific optimizations
- **Debug Support**: Full debugging symbols and validation checks
- **Library Management**: Automatic detection of optional dependencies (NVML, MIOpen)
- **Cross-Platform**: Single Makefile supports both CUDA and HIP builds

##  Performance Expectations

| Module Level | Typical GPU Speedup | Memory Efficiency | Code Quality |
|--------------|-------------------|------------------|--------------|
| **Beginner** | 10-100x | 60-80% | Educational |
| **Intermediate** | 50-500x | 80-95% | Optimized |
| **Advanced** | 100-1000x | 85-95% | Production |
| **Expert** | 500-5000x | 95%+ | Library-Quality |

## 🐛 Troubleshooting

### Common Issues & Solutions

**GPU Not Detected**
```bash
# NVIDIA
nvidia-smi  # Should show your GPU
export PATH=/usr/local/cuda/bin:$PATH

# AMD  
rocm-smi   # Should show your GPU
export HIP_PLATFORM=amd
```

**Compilation Errors**
```bash
# Check CUDA installation
nvcc --version
make check-cuda

# Check HIP installation  
hipcc --version
make check-hip
```

**Docker Issues**
```bash
# Test Docker GPU access
./docker/scripts/test.sh

# Rebuild containers
./docker/scripts/build.sh --clean --all
```

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **README.md** | Main project documentation and getting started guide |
| [**CONTRIBUTING.md**](CONTRIBUTING.md) | How to contribute to the project |
| [**Docker Guide**](docker/README.md) | Complete Docker setup and usage |
| [**Module READMEs**](modules/) | Individual module documentation |

## 🤝 Contributing

We welcome contributions from the community! This project thrives on:

- 📝 **New Examples**: Implementing additional GPU algorithms
- 🐛 **Bug Fixes**: Improving existing code and documentation  
- 📚 **Documentation**: Enhancing explanations and tutorials
- 🔧 **Optimizations**: Performance improvements and best practices
- 🌐 **Platform Support**: Cross-platform compatibility improvements

**[📖 Contributing Guidelines →](CONTRIBUTING.md)** • **[🐛 Report Issues →](https://github.com/AIComputing101/gpu-programming-101/issues)** • **[💡 Request Features →](https://github.com/AIComputing101/gpu-programming-101/issues/new?template=feature_request.md)**

## 🏆 Community & Support

- 🌟 **Star this project** if you find it helpful!
- 🐛 **Report bugs** using our [issue templates](https://github.com/AIComputing101/gpu-programming-101/issues/new/choose)
- 💬 **Join discussions** in [GitHub Discussions](https://github.com/AIComputing101/gpu-programming-101/discussions)
- 📧 **Get help** from the community and maintainers

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this project in your research, education, or publications, please cite it as:

### BibTeX
```bibtex
@misc{gpu-programming-101,
  title={GPU Programming 101: A Comprehensive Educational Project for CUDA and HIP},
  author={{Stephen Shao}},
  year={2025},
  howpublished={\url{https://github.com/AIComputing101/gpu-programming-101}},
  note={A complete GPU programming educational resource with 70+ production-ready examples covering fundamentals through advanced optimization techniques for NVIDIA CUDA and AMD HIP platforms}
}
```

### IEEE Format
Stephen Shao, "GPU Programming 101: A Comprehensive Educational Project for CUDA and HIP," GitHub, 2025. [Online]. Available: https://github.com/AIComputing101/gpu-programming-101

## 🙏 Acknowledgments

- 🎯 **NVIDIA** and **AMD** for excellent GPU computing ecosystems
- 📚 **GPU computing community** for sharing knowledge and best practices  
- 👥 **Contributors** who make this project better every day

---

**Ready to unlock the power of GPU computing?**

**[🚀 Get Started Now](#-quick-start)** • **[📚 View Modules](modules/)** • **[🐳 Try Docker](docker/README.md)**

---

**⭐ Star this project • 🍴 Fork and contribute • 📢 Share with others**

*Built with ❤️ for the AI Computing 101*
