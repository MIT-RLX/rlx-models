# tsac-ng Project Handbook v0.1.0

## Project Identity

```
tsac-ng v0.1.0 — Copyright (c) 2026 Hope2333（幽零小喵）
Clean-room implementation of a neural audio codec,
independently developed from first principles.
Like Linux to Unix — same ecosystem compatibility,
built from scratch with zero shared code.
```

## Architecture Overview

### System Stack
```
┌─────────────────────────────────────────────────────┐
│                     CLI (main.c)                     │
│   tsac [options] c|d|t infile outfile                │
├─────────────────────────────────────────────────────┤
│               Codec API (tsac_codec.c)               │
│   tsac_init → tsac_compress → tsac_decompress        │
├────────────────┬────────────────┬───────────────────┤
│  CPU Decoder   │  CUDA Backend  │   HIP Backend     │
│  (cpu_decoder) │  (cuda_kernels │  (dac_decoder.hip │
│   + arch/arm   │   + backend)   │   + hip_arch)     │
│   + arch/riscv)│                │                    │
├────────────────┴────────────────┴───────────────────┤
│            Model I/O Layer                           │
│  txc_format.c  ←→  .txc container                    │
│  model_loader.c → .bin model (BF8/float32)           │
└─────────────────────────────────────────────────────┘
```

### Weight Format Detection
```
Input: weight_v[d0, d1, d2] + bias[Co] + weight_g[1,1,N]

Given:
  Co = bias->dims[0]
  K  = d1

If (d0 == Co):  → format: [Co, K, Ci]
  Ci = d2
  conv_transpose = true
Else:            → format: [Ci, K, Co]
  Ci = d0
  conv_transpose = false

Output target: [Co, Ci, K] for kernel access w[oc*Ci*K + ic*K + j]
```

### BF8 Dequantization
```c
// Stored: uint8 v ∈ [0,255] = clamp(round(w/scale), -127, 127) + 128
// Dequant: w_f32 = g * ((int8_t)v - 128.0f) / 127.0f
int8_t v_val = (int8_t)v_data[idx];
float g = g_scales[per_input ? ci : co];
w_f32[dst_idx] = g * ((float)v_val - 128.0f) / 127.0f;
```

### Encoder/Decoder Graph
```
Encoder (PCM → codebook indices):        Decoder (indices → PCM):
PCM[ch, T]                               indices[n_frames × n_cb]
  │                                          │
  ▼ Conv1d(ch→96, K=7)                      ▼ RVQ Lookup → [1024, T]
  ▼ Snake(96)                               
  ▼ Block4: 96→192 (stride=1)               ▼ Conv1d(1024→1536)
  ▼ Block3: 192→384 (stride=1)              ▼ Block1: 1536→768 (×2↑)
  ▼ Block2: 384→768 (stride=1)              ▼ Block2: 768→384 (×2↑)
  ▼ Block1: 768→1536 (stride=1)             ▼ Block3: 384→192 (×2↑)
  ▼ Conv1d(1536→1024)                       ▼ Block4: 192→96 (×2↑)
  ▼ RVQ Quantize (L2 argmin)                ▼ Snake(96)
    12 codebooks × 8 entries                 ▼ Conv1d(96→ch)
    1024-dim vectors                         
  indices → .txc                            PCM[ch, T×16]
```

## Debugging Patterns

### When NaN appears in output

1. **Check model.0 output first**
   - If model.0 output is valid → NaN is in blocks 1-4
   - If model.0 output is NaN → RVQ lookup or input issue

2. **Check conv_transpose output after each block**
   - Overflow (>1e10) → weight format mismatch ([Co,K,Ci] vs [Co,Ci,K])
   - Values ~1e5 but correct patterns → stride bug (should be stride=1)
   - All zeros → missing bias or weight tensor

3. **Check kernel launch configurations**
   - conv1d needs 2D grid: `dim3(Co, (T+BLK-1)/BLK)`
   - convt needs 2D grid: `dim3(Ci, (Ti+BLK-1)/BLK)`
   - 1D grid = only first row processed

### When GPU page fault occurs
1. Check that GPU device pointers ≠ CPU pointers
2. Check that kernel doesn't read past buffer end
3. Check that weight_g broadcast doesn't exceed its allocation
4. Check `data_size` is in ELEMENTS not BYTES

## Build Matrix

```bash
# x86-64 CPU (default)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# x86-64 CUDA
cmake .. -DUSE_CUDA=ON -DCUDAToolkit_ROOT=/opt/cuda

# x86-64 HIP/ROCm
cmake .. -DUSE_HIP=ON -DHIP_PATH=/opt/rocm

# ARM64 cross-compile
cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-arm64.cmake

# RISC-V cross-compile
cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-riscv64.cmake
```

## Model Requirements

- `dac_stereo_q8.bin`: 82MB, 322 tensors (full decode capable)
  - decoder.model.0-6 (119 tensors) — decoder graph
  - quantizer.quantizers.0-11 (84 tensors) — RVQ codebooks
  - encoder.block.0-6 (119 tensors) — encoder graph
- GitHub Release `v0.1.0-models`: 85MB on server (same file)
- **WARNING**: 26MB variant with only 206 tensors exists (missing decoder blocks 1-4),
  will produce all-zeros on decode

## Key Files

| File | Lines | Purpose | Read Order |
|------|-------|---------|:----------:|
| `src/cpu_decoder.c` | 946 | CPU decoder + SIMD dispatch | 1st |
| `src/cuda/cuda_backend.cu` | 798 | CUDA encode+decode (reference) | 2nd |
| `src/tsac_codec.c` | 464 | Core API + WAV I/O | 3rd |
| `src/txc_format.c` | 179 | .txc container | 4th |
| `src/model_loader.c` | 113 | .bin model loader | 5th |
| `hip/dac_decoder.hip.cpp` | 1005 | HIP encode+decode port | 6th |
| `src/vulkan/vulkan_arch.c` | 216 | Vulkan compute | 7th |
| `src/llvm/llvm_backend.c` | ~460 | LLVM JIT experimental | 8th |

## Common Commands

```bash
# Encode
./tsac-ng --cuda -v c input.wav output.txc

# Decode
./tsac-ng --cuda -v d input.txc output.wav

# Round-trip test
./tsac-ng -v t input.wav roundtrip.wav

# CUDA build + quick test
cd build-cuda-dbg && make -j$(nproc)
./tsac-ng --cuda c /tmp/test_stereo.wav /tmp/test.txc
./tsac-ng --cuda d /tmp/test.txc /tmp/test_rt.wav
python3 -c "import struct; print(sum(1 for v in struct.unpack(f'<{os.path.getsize(chr(47)+chr(116)+chr(109)+chr(112)+chr(47)+chr(116)+chr(101)+chr(115)+chr(116)+chr(95)+chr(114)+chr(116)+chr(46)+chr(119)+chr(97)+chr(118))//4}f', open(chr(47)+chr(116)+chr(109)+chr(112)+chr(47)+chr(116)+chr(101)+chr(115)+chr(116)+chr(95)+chr(114)+chr(116)+chr(46)+chr(119)+chr(97)+chr(118),'rb').read()[44:]) if abs(v)>1e-10)) if v==v else 0)"
```
