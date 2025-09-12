---
name: Bug report
about: Create a report to help us improve the course
title: '[BUG] '
labels: bug
assignees: ''

---

## 🐛 Bug Description
A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## ✅ Expected Behavior
A clear and concise description of what you expected to happen.

## ❌ Actual Behavior
A clear and concise description of what actually happened.

## 🖼️ Screenshots
If applicable, add screenshots to help explain your problem.

## 💻 Environment Information
**Operating System:**
- [ ] Ubuntu 20.04/22.04
- [ ] Windows 10/11
- [ ] macOS
- [ ] Other: ___________

**GPU Information:**
- **GPU Model**: [e.g. RTX 4090, RX 7900 XTX]
- **Driver Version**: [e.g. NVIDIA 535.98, AMD 23.4.2]
- **CUDA Version**: [e.g. 12.2] (if applicable)
- **ROCm Version**: [e.g. 5.6.0] (if applicable)

**Development Environment:**
- [ ] Native installation
- [ ] Docker container
- **Container Image**: [e.g. gpu-programming-101:cuda]

**Module Information:**
- **Module**: [e.g. Module 1, Module 4]
- **Example**: [e.g. 01_vector_addition_cuda.cu]

## 🔧 Additional Context
Add any other context about the problem here.

## 📋 Error Output
```bash
# Paste complete error output here
```

## 🔍 System Information
If possible, include output from:
```bash
# For NVIDIA systems
nvidia-smi
nvcc --version

# For AMD systems  
rocm-smi
hipcc --version

# Build system
make --version
gcc --version
```

## ✅ Checklist
- [ ] I have searched existing issues for this problem
- [ ] I have tried the troubleshooting steps in the module documentation
- [ ] I have included all required environment information
- [ ] I have provided a clear description of the issue