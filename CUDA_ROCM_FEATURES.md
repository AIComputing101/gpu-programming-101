# CUDA and ROCm Feature Guide (Living Document)

Last updated: 2025-09-22

This guide summarizes current, officially documented features of NVIDIA CUDA and AMD ROCm that we leverage across this project. It is designed to be easy to maintain as new versions ship. Where possible, we link to authoritative sources instead of restating volatile details.

Tip: Prefer the linked release notes and programming guides for exact, version-specific behavior. Update checklist is at the end of this document.

---

## Current Versions at a Glance

- CUDA: 13.0 Update 1 (13.0.U1)
  - Source of truth: NVIDIA CUDA Toolkit Release Notes
  - Driver requirement overview: CUDA Compatibility Guide for Drivers
- ROCm: 7.0.1
  - Source of truth: ROCm Release History and ROCm docs index

Reference links are provided at the bottom for maintenance.

---

## CUDA 13.x overview

Highlights pulled from NVIDIA’s official docs (see links):

- General platform
  - CUDA 13.x is ABI-stable within the major series; requires r580+ driver on Linux.
  - Increased MPS server client limits on Ampere and newer architectures (subject to architectural limits).
- Compiler and runtime
  - NVCC/NVRTC updates; PTX ISA updates (see PTX 9.0 notes in release docs).
  - Programmatic Dependent Launch (PDL) support in select library kernels on sm_90+.
- Developer tools
  - Nsight Systems and Nsight Compute continue as the primary profilers.
  - Compute Sanitizer updates; Visual Profiler and nvprof are removed in 13.0.
- Deprecations and removals
  - Dropped offline compilation/library support for pre-Turing architectures (Maxwell, Pascal, Volta) in CUDA 13.0. Continue to use 12.x to target these.
  - Windows Toolkit no longer bundles a display driver (install separately).
  - Removed multi-device cooperative group launch APIs; several legacy headers removed.

Architectures and typical use cases (non-exhaustive):

- Blackwell/Blackwell Ultra (SM110+): next‑gen AI/HPC; FP4/FP8 workflows via libraries.
- Hopper (H100/H200, SM90): transformer engine, thread block clusters, DPX; AI training/HPC.
- Ada (RTX 40): workstation/development; AV1 encode; content creation/AI dev.
- Ampere (A100/RTX 30): MIG, 3rd‑gen tensor cores; research/mixed workloads.

Core libraries snapshot (examples; see library release notes for specifics):

- cuBLAS/cuBLASLt: autotuning options; improvements on newer architectures; mixed precision and block‑scaled formats.
- cuFFT: new error codes; performance changes; dropped pre‑Turing support.
- cuSPARSE: generic API enhancements; 64‑bit indices in SpGEMM; various bug fixes.
- Math/NPP/nvJPEG: targeted perf/accuracy improvements and API cleanups.

Authoritative references:

- CUDA Toolkit Release Notes (13.0 U1)
- CUDA Compatibility Guide for Drivers
- Nsight Systems Release Notes; Nsight Compute Release Notes
- CUDA C++ Programming Guide changelog

---

## ROCm 7.0.x overview

Highlights from AMD’s official docs (see links):

- ROCm 7.0.1 is the latest as of 2025‑09‑17; consult the release history for point updates.
- HIP as the primary programming model, with CUDA‑like APIs and HIP‑Clang toolchain.
- Windows support targets HIP SDK for development; full ROCm stack targets Linux.
- ROCm Libraries monorepo: multiple core math and support libraries are consolidated in the ROCm Libraries monorepo for unified CI/build. Projects included (as of rocm‑7.0.1): composablekernel, hipblas, hipblas-common, hipblaslt, hipcub, hipfft, hiprand, hipsolver, hipsparse, hipsparselt, miopen, rocblas, rocfft, rocprim, rocrand, rocsolver, rocsparse, rocthrust. Shared components: rocroller, tensile, mxdatagenerator. Most of these are marked “Completed” in the monorepo migration status and the monorepo is the source of truth; see its README for current status.
- Tooling and system components: ROCr runtime, ROCm SMI, rocprof/rocprofiler, rocgdb/rocm‑debug‑agent.

Nomenclature: project names in the monorepo are standardized to match released package names (for example, hipblas/hipfft/rocsparse instead of mixed casing).

Architectures (illustrative, not exhaustive):

- CDNA3 (MI300 family): AI training and HPC; unified memory on APUs (MI300A), large HBM configs (MI300X).
- RDNA3 (Radeon 7000 series): workstation/gaming; AV1 encode/decode; hardware ray tracing.

Common libraries (see ROCm Libraries reference and monorepo):

- BLAS/solver/sparse: rocBLAS / hipBLAS, hipBLASLt, rocSOLVER / hipSOLVER, rocSPARSE / hipSPARSE, hipSPARSElt.
- FFT/random/core: rocFFT / hipFFT, rocRAND / hipRAND, rocPRIM / hipCUB, rocThrust.
- Kernel building blocks: composablekernel; shared dependencies like Tensile and rocRoller (used by rocBLAS/hipBLASLt).
- ML/DL: MIOpen; framework integrations via the ROCm for AI guide.

Authoritative references:

- ROCm Docs index (What is ROCm?, install, reference)
- ROCm Release History (7.0.1, 7.0.0, …)
- ROCm libraries reference; tools/compilers/runtimes reference
- ROCm Libraries monorepo (status, structure, releases): https://github.com/ROCm/rocm-libraries

---

## Cross‑platform mapping (CUDA ⇄ HIP)

Quick mapping for common concepts. Always check specific APIs for support and behavior differences.

- Kernel launch
  - CUDA: <<<grid, block, shared, stream>>>; HIP: hipLaunchKernelGGL
- Memory management
  - CUDA: cudaMalloc/cudaMemcpy/etc.; HIP: hipMalloc/hipMemcpy/etc.
- Streams and events
  - CUDA: cudaStream_t/cudaEvent_t; HIP: hipStream_t/hipEvent_t
- Graphs
  - CUDA: cudaGraph_t and Graph Exec; HIP: hipGraph_t and equivalents; feature coverage evolves, verify against ROCm docs.
- Cooperative groups
  - CUDA: cooperative_groups; HIP: HIP cooperative groups header; multi‑device variants differ (and some CUDA multi‑device APIs removed in 13.0).
- Libraries
  - cuBLAS ↔ hipBLAS/rocBLAS; cuFFT ↔ hipFFT/rocFFT; cuSPARSE ↔ hipSPARSE/rocSPARSE; Thrust/CUB ↔ rocThrust/hipCUB/rocPRIM.

Porting aids:

- hipify (perl/python) for source translation; hip‑clang for compilation.

---

## Compatibility and supported platforms

- CUDA drivers and OS
  - See the CUDA Compatibility Guide for minimum driver versions by toolkit series (e.g., 13.x requires r580+ on Linux). Windows driver no longer bundled starting with 13.0.
- CUDA architectures
  - 13.0 drops offline compilation/library support for Maxwell/Pascal/Volta; continue to use 12.x for those targets.
- ROCm OS/GPU support
  - See ROCm install guides and GPU/accelerator support references for Linux and Windows HIP SDK system requirements.

---

## Educational integration (this repository)

This course demonstrates both CUDA and HIP across modules. Key tool updates to note:

- Profiling and analysis
  - NVIDIA: Nsight Systems, Nsight Compute, CUPTI changes in 13.x, Compute Sanitizer
  - AMD: rocprof/rocprofiler, ROCm SMI
- Memory and graphs
  - CUDA: CUDA Graphs; memory pools and VMM; asynchronous copy
  - ROCm: HIP graph APIs (coverage evolves); ROCr runtime memory features

Example module alignment (indicative; see each module’s README for details):

- Module 1: Runtime APIs, device queries, build/tooling
- Module 2: Memory management (device, pinned, unified/coherent where available)
- Module 3: Synchronization and cooperation (warp/wavefront‑level, cooperative groups)
- Module 4: Streams, events, graphs, and multi‑GPU basics
- Module 5: Profiling and debugging (Nsight Tools, Compute Sanitizer, rocprof, rocm‑smi)
- Module 6+: Libraries (BLAS/FFT/SPARSE) and domain examples (AI/HPC)

### New features by module (CUDA 13.x and ROCm 7.0.x)

| Module | CUDA (what you’ll learn) | ROCm/HIP (what you’ll learn) |
|---|---|---|
| Module 1: Getting Started | Toolchain (nvcc), project layout, kernel launch basics (grid/block/thread indexing), device vs host code, cudaMalloc/cudaMemcpy, device query and error handling | Toolchain (hipcc/hip-clang), hipLaunchKernelGGL, hipMalloc/hipMemcpy, hipGetDeviceProperties, mapping CUDA concepts to HIP |
| Module 2: Memory & Data Movement | Global/shared/constant/texture memory usage, coalesced access, pinned memory, unified memory and prefetch, async copies and measuring bandwidth | HIP memory APIs and ROCr memory model, pinned host buffers, unified/coherent memory notes, async transfers, using rocm-smi/rocprof to observe bandwidth |
| Module 3: Parallel Patterns & Sync | Reductions, scans, sorting; warp-level primitives; cooperative groups; shared memory tiling; atomics and barriers; occupancy considerations | rocPRIM/hipCUB/rocThrust equivalents; wavefront-level ops; HIP cooperative groups; LDS usage; atomics and synchronization semantics |
| Module 4: Concurrency, Streams & Multi‑GPU | Streams/events, priorities, CUDA Graphs (capture/instantiate/launch), peer-to-peer (UVA/P2P), basic multi‑GPU patterns | hipStream/hipEvent, HIP Graph API coverage and usage, peer access where supported, multi‑GPU fundamentals with ROCm tools |
| Module 5: Profiling, Debugging & Sanitizers | Nsight Systems (timeline/tracing), Nsight Compute (kernel analysis), Compute Sanitizer (racecheck/memcheck), intro to CUPTI-based profiling | rocprof/rocprofiler for traces and metrics, rocm-smi telemetry, rocgdb/ROCm Debug Agent basics, best practices for profiling |
| Module 6: Math & Core Libraries | cuBLAS/cuBLASLt (GEMM, batched ops, mixed precision), cuFFT, cuSPARSE, Thrust/CUB algorithms, choosing/tuning library routines | rocBLAS/hipBLAS, rocFFT/hipFFT, rocSPARSE/hipSPARSE, rocThrust/hipCUB/rocPRIM; Tensile-backed tuning in rocBLAS; API parity tips |
| Module 7: Advanced Algorithms & Optimization | Tiling and cache use, shared memory bank conflicts, cooperative groups for complex patterns, intro to memory pools/VMM, kernel fusion patterns | Wavefront-aware tuning, LDS patterns, rocPRIM building blocks, HIP-specific perf tips, memory behavior across devices |
| Module 8: AI/ML Workflows | cuDNN basics, TensorRT concepts (dynamic shapes/precision), mixed precision (FP16/BF16/FP8 via libs), graphs for inference pipelines | MIOpen basics, framework setup on ROCm (PyTorch/TF where supported), MIGraphX or framework runtimes, mixed precision support |
| Module 9: Packaging, Deployment & Containers | CUDA containers (base/runtime-devel), driver/runtime compatibility, minimal deployment artifacts, reproducible builds | ROCm container bases (rocm/dev), runtime setup (kernel modules, groups/permissions), compatibility guidance and reproducibility |

---

## Maintenance: how to update this document

When CUDA or ROCm releases a new version, follow this checklist:

1) Update versions at the top
   - CUDA: consult CUDA Toolkit Release Notes page; record the latest major.minor (e.g., 13.0 Update 1) and driver requirements.
   - ROCm: consult ROCm Release History; record latest (e.g., 7.0.1).
2) Scan notable changes
   - CUDA: skim “New Features”, “Deprecated or Dropped Features”, and library sections (cuBLAS/cuFFT/…); note any course‑impacting changes.
   - ROCm: skim “What is ROCm?”, “ROCm libraries”, and “Tools/Compilers/Runtimes” sections for new features or renamed packages.
3) Verify cross‑platform notes
   - Confirm HIP Graph API coverage and any caveats; update mapping if needed.
4) Update references
   - Keep the link reference list (below) current; avoid copying long tables—link out to authoritative docs.
5) Record the date in “Last updated”.

Tip: Avoid claiming specific percentage speedups unless you include a citation. Prefer phrasing like “performance improvements in X; see release notes.”

---

## Reference links (authoritative sources)

- NVIDIA
  - CUDA Toolkit Release Notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
  - CUDA Compatibility Guide (drivers): https://docs.nvidia.com/deploy/cuda-compatibility/index.html
  - CUDA C++ Programming Guide (changelog): https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#changelog
  - Nsight Systems Release Notes: https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html
  - Nsight Compute Release Notes: https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html
- AMD
  - ROCm docs index: https://rocm.docs.amd.com/en/latest/index.html
  - ROCm release history: https://rocm.docs.amd.com/en/latest/release/versions.html
  - ROCm libraries reference: https://rocm.docs.amd.com/en/latest/reference/api-libraries.html
  - ROCm tools/compilers/runtimes: https://rocm.docs.amd.com/en/latest/reference/rocm-tools.html
  - HIP documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/index.html