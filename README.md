# GPU Programming 101 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.9.1-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![ROCm](https://img.shields.io/badge/ROCm-6.4.3-red?logo=amd)](https://rocmdocs.amd.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![Examples](https://img.shields.io/badge/Examples-70%2B-green)](modules/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/features/actions)

**A comprehensive, hands-on educational project for mastering GPU programming with CUDA and HIP**

*From beginner fundamentals to production-ready optimization techniques*

**Quick Navigation:** [🚀 Quick Start](#-quick-start) • [📚 Modules](#-modules) • [🐳 Docker Setup](#-docker-development) • [🤝 Contributing](CONTRIBUTING.md)

---

## 📋 Project Overview

**GPU Programming 101** is a complete educational resource for learning modern GPU programming. This project provides:

- **9 comprehensive modules** covering beginner to expert topics
- **70+ working code examples** in both CUDA and HIP
- **Cross-platform support** for NVIDIA and AMD GPUs  
- **Production-ready development environment** with Docker
- **Professional tooling** including profilers, debuggers, and CI/CD

Perfect for students, researchers, and developers looking to master GPU computing.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Complete Curriculum** | 9 progressive modules from basics to advanced topics |
| 💻 **Cross-Platform** | Full CUDA and HIP support for NVIDIA and AMD GPUs |
| 🐳 **Docker Ready** | Complete containerized development environment |
| 🔧 **Production Quality** | Professional build systems, testing, and profiling |
| 📊 **Performance Focus** | Optimization techniques and benchmarking throughout |
| 🌐 **Community Driven** | Open source with comprehensive contribution guidelines |

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
cd modules/module1 && make && ./01_vector_addition_cuda
```

### Option 2: Native Installation
For direct system installation:

```bash
# Prerequisites: CUDA 11.0+ or ROCm 5.0+, GCC 7+, Make

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

### 🎯 What You'll Learn

**👶 Beginner Track** - Start here if you're new to GPU programming
- GPU architecture fundamentals
- Writing your first CUDA/HIP kernels  
- Memory management between CPU and GPU
- Basic parallel algorithms
- Debugging and profiling basics

**🔥 Intermediate Track** - For developers with some parallel programming experience
- Advanced memory optimization techniques
- Multi-dimensional data processing
- GPU architecture deep dive
- Performance engineering
- Multi-GPU programming

**🚀 Advanced Track** - For experts seeking production-level skills
- Fundamental parallel algorithms (reduction, scan, convolution)
- Advanced algorithmic patterns (sorting, sparse matrices)
- Domain-specific applications (ML, scientific computing)
- Production deployment and optimization
- Next-generation GPU architectures

## 📚 Modules

| Module | Level | Duration | Topics | Examples |
|--------|-------|----------|--------|----------|
| [**Module 1**](modules/module1/) | Beginner | 4-6h | GPU Fundamentals, CUDA/HIP Basics | 13 |
| [**Module 2**](modules/module2/) | Beginner-Intermediate | 6-8h | Multi-Dimensional Data Processing | 10 |
| [**Module 3**](modules/module3/) | Intermediate | 6-8h | GPU Architecture & Execution Models | 12 |
| [**Module 4**](modules/module4/) | Intermediate-Advanced | 8-10h | Advanced GPU Programming | 9 |
| [**Module 5**](modules/module5/) | Advanced | 6-8h | Performance Engineering | 5 |
| [**Module 6**](modules/module6/) | Intermediate-Advanced | 8-10h | Fundamental Parallel Algorithms | 10 |
| [**Module 7**](modules/module7/) | Advanced | 8-10h | Advanced Algorithmic Patterns | 4 |
| [**Module 8**](modules/module8/) | Advanced | 10-12h | Domain-Specific Applications | 4 |
| [**Module 9**](modules/module9/) | Expert | 6-8h | Production GPU Programming | 4 |

**📈 Progressive Learning Path: 70+ Examples • 50+ Hours • Beginner to Expert**

**[� View Learning Modules →](modules/)**

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
- **Linux** (Recommended): Ubuntu 22.04 LTS, RHEL 8/9, SLES 15 SP5
- **Windows**: Windows 10/11 with WSL2 recommended for optimal compatibility
- **macOS**: macOS 12+ (Metal Performance Shaders for basic GPU compute)

#### GPU Computing Platforms
- **CUDA Toolkit**: 12.0+ (Docker uses CUDA 12.9.1)
  - **Driver Requirements**: 
    - Linux: 550.54.14+ for CUDA 12.4+
    - Windows: 551.61+ for CUDA 12.4+
- **ROCm Platform**: 6.0+ (Docker uses ROCm 6.4.3)
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

**[📖 Complete Docker Guide →](docker/README.md)**

## 🔧 Build System

### Project-Wide Commands
```bash
make all           # Build all modules
make test          # Run comprehensive tests  
make clean         # Clean all artifacts
make check-system  # Verify GPU setup
make status        # Show module completion status
```

### Module-Specific Commands
```bash
cd modules/module1/examples
make               # Build all examples in module
make test          # Run module tests
make profile       # Performance profiling
make debug         # Debug builds with extra checks
```

## 🚦 Getting Started Guide

### 1. Choose Your Path
- **🐳 Docker**: No setup required, works everywhere → [Docker Guide](docker/README.md)  
- **💻 Native**: Direct system installation → [Installation Guide](#option-2-native-installation)

### 2. Start Learning
```bash
# Begin with Module 1
cd modules/module1
cat README.md        # Read learning objectives  
cd examples && make  # Build examples
./01_vector_addition_cuda  # Run your first GPU program!
```

### 3. Progress Through Modules
- Each module builds on previous concepts
- Complete examples and exercises in order
- Use profiling tools to understand performance
- Experiment with different optimizations

### 4. Advanced Topics
- Modules 6-9 cover production-level techniques
- Focus on algorithms and applications relevant to your domain
- Contribute back with improvements and new examples

## 📊 Performance Expectations

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

**[� Need Help? Check Common Issues →](README.md#-troubleshooting)**

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

**TL;DR**: ✅ Commercial use ✅ Modification ✅ Distribution ✅ Private use

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
- 🏫 **Educational institutions** advancing parallel computing education
- 👥 **Contributors** who make this project better every day

---

**Ready to unlock the power of GPU computing?**

**[🚀 Get Started Now](#-quick-start)** • **[📚 View Modules](modules/)** • **[🐳 Try Docker](docker/README.md)**

---

**⭐ Star this project • 🍴 Fork and contribute • 📢 Share with others**

*Built with ❤️ for the AI Computing 101*