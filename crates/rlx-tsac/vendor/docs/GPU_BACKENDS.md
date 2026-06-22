# tsac-ng GPU Backend Guide

## Architecture Comparison

| Aspect | CUDA | HIP | Vulkan |
|--------|------|-----|--------|
| API level | Runtime (`cuda_runtime.h`) | Runtime (`hip_runtime.h`) | Loader (`dlopen` + `vulkan.h`) |
| Kernel language | CUDA C++ | HIP C++ | GLSL/SPIR-V |
| Grid config | 1D/2D blocks | Same as CUDA | Workgroups |
| Device pointers | `float *d_dev` | Same | `VkBuffer` |
| Async | Streams | Streams | Command buffers |
| Compile | nvcc | clang (HIP) | glslangValidator |
| Driver | NVIDIA | AMD ROCm | Vulkan loader (cross-platform) |

## CUDA Backend (`src/cuda/`)

### Files
| File | Lines | Content |
|------|-------|---------|
| `cuda_kernels.cu` | 284 | conv1d, convt1d, snake, add_bias, tanh_clip, rvq_lookup, rvq_quantize, rvq_subtract |
| `cuda_backend.cu` | 798 | CudaBackend struct, weight upload, encode, decode, shutdown |

### Key Design
- Runtime API (`cuda_runtime.h`), NOT Driver API
- Weights uploaded lazily on first decode/encode call
- 8 reusable GPU buffers managed by `cuda_backend_get_buf()`
- SM 8.0+ (RTX 4060 = CC 8.9, RTX 4090 = CC 8.9)

### Weight Upload
```c
// Called once on first decode. Uploads 18 decoder + 18 encoder weights + bias + codebooks
GpuWeight gw;  // name, Ci, K, Co, is_convt, d_data (GPU ptr)
// Snapshots:
// w0[0..3] = -0.487303 -0.448057 -0.379377 -0.748942  ✅
```

### Encode/Decode Flow
```
tsac_cuda_encode:
  PCM → model.6→Snake→Block4→Block3→Block2→Block1→model.0→RVQ→indices
  
tsac_cuda_decode:
  indices→RVQ→model.0→Block1→Block2→Block3→Block4→Snake→model.6→PCM
```

## HIP Backend (`hip/`)

### Files
| File | Lines | Content |
|------|-------|---------|
| `hip_arch.hip.cpp` | 136 | HipBackend struct, init, encode, decode, shutdown |
| `hip_kernels.hip.cpp` | 374 | HIP kernels (neg, exp, relu, gelu, add, mul, sub, layernorm, softmax) |
| `dac_decoder.hip.cpp` | 1005 | DAC kernels + encoder/decoder graph |

### Key Differences from CUDA
- Uses `upload_f32()` for CPU-side dequant (same concept as CUDA's weight upload)
- Buffer management: single large allocation (~9MB) split into b0/b1/b2
- GPU target: gfx1036 (RX 610M) — Radeon 6000 series
- ROCm 7.2 with Clang 18.0.0

### HIP-Specific Issues
| Issue | Fix | Status |
|-------|-----|:------:|
| `hipLaunchKernelGGL` deprecated in ROCm 7.2 | Replace with `<<<>>>` syntax | ✅ |
| `nodiscard` warnings treated as errors | Add `(void)` cast | ✅ |
| Device variables not declared | `set_source_files_properties(LANGUAGE HIP)` | ✅ |

## Vulkan Backend (`src/vulkan/`)

### Files
| File | Lines | Content |
|------|-------|---------|
| `vulkan_arch.c` | 216 | Vulkan instance, device, pipeline creation + decode |
| `vulkan_shaders.h` | ~1500 | Embedded SPIR-V byte arrays (4 shaders) |
| `shaders/*.comp` | — | GLSL shader sources |

### Architecture
- dlopen-based loader: no compile-time dependency on libvulkan
- Function pointers loaded via `dlsym()` at runtime
- 4 SPIR-V shaders: conv1d, snake, group_norm, add
- Designed for ARM64 Mali (G925) cross-compilation

### On-Device Testing (TODO)
1. Push ARM64 binary + SPIR-V shaders to device
2. Run: `./tsac-ng --vulkan d input.txc output.wav`
3. Expected: `[vk] GPU: Mali-G925` + `4/4 pipelines ready`

## LLVM JIT Backend (`src/llvm/`)

### Status: Experimental — Init hangs on LLVM 22

The LLVM MCJIT backend generates LLVM IR at runtime for conv1d, convt, snake,
and add kernels. The IR uses nested loops with bounds checking.

**Problem**: LLVM 22 deprecated the old pass manager API.
`LLVMPassManagerRef` is removed. Need to use new pass builder API.

**Fix** (when needed):
```c
// Old API (LLVM 16-):
LLVMPassManagerRef pm = LLVMCreatePassManager();
LLVMAddPromoteMemoryToRegisterPass(pm);
LLVMRunPassManager(pm, module);

// New API (LLVM 17+):
// Use LLVM's new PassBuilder via C API or skip passes entirely
// (The JIT'd code works without optimization — just slower)
```

## Cross-Compilation for GPU Backends

### ARM64 CUDA → Not possible (NVIDIA doesn't support ARM64 CUDA)

### ARM64 HIP → Not applicable (no AMD GPUs in ARM64)

### ARM64 Vulkan ✅ (Primary target)
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-arm64.cmake \
      -DUSE_VULKAN=ON
make -j$(nproc)
```
Then deploy to Termux with SPIR-V shaders.

### RISC-V → CPU only (no GPU backends available)
