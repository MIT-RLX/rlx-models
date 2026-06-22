# tsac-ng Architecture

## Decoder Graph

```
Input: codebook_indices[n_frames × n_codebooks]  (uint8, 0..255)

Step 1 — RVQ Lookup:
  For each codebook 0..n_cb-1:
    Look up codebook[codebook_idx][1024] from quantizer.quantizers.X.codebook.weight
    Add to feature vector [1024 × n_frames]

Step 2 — Decoder (7-layer DAC):
  model.0: Conv1d(1024→1536, K=7)                    → [1536, n_frames]

  Block 1 (1536→768, K=16, stride=2):
    Snake(1536) → ConvTranspose1d → [768, n_frames×2]
    ×3 inner: Snake(768)→Conv1d(768→768,K=7)→Snake(768)→Conv1d(768→768,K=1)

  Block 2 (768→384, K=16, stride=2):
    Snake(768) → ConvTranspose1d → [384, n_frames×4]
    ×3 inner: same pattern with 384 channels

  Block 3 (384→192, K=8, stride=2):
    Snake(384) → ConvTranspose1d → [192, n_frames×8]
    ×3 inner: same pattern with 192 channels

  Block 4 (192→96, K=4, stride=2):
    Snake(192) → ConvTranspose1d → [96, n_frames×16]
    ×3 inner: same pattern with 96 channels

  model.5: Snake(96)                                   → [96, n_frames×16]

  model.6: Conv1d(96→2, K=7)                          → [2, n_frames×16]

Output: PCM float [2, n_frames×16]  (stereo, interleaved or planar)
```

## Weight Storage

Model weights are stored in `.bin` container (magic 0x23f4aefb) with per-tensor
headers (magic 0x23f4aefa). 322 tensors total.

### BF8 Quantization

Most convolution weights are BF8 quantized:
- `weight_v`: uint8 values [0,255], stored in [Co,K,Ci] or [Ci,K,Co] layout
- `weight_g`: float32 per-channel scales

Dequantization: `w = g[channel] * ((int8_t)v - 128.0f) / 127.0f`

### Storage Layout Detection

Convolution weight storage order varies:
- Conv1d layers: `[Ci, K, Co]` — bias dimension matches dim[2]
- ConvTranspose1d layers: `[Co, K, Ci]` — bias dimension matches dim[0]

Detection: compare bias->dims[0] against weight_v->dims[0] and dims[2].

### Float32 Weights

model.4.block.1 and model.6 use float32 (not BF8):
- Detected by: `data_size == dims_product × 4`
- No dequantization needed — direct copy

## SIMD Architecture

```
get_ops()
├── x86-64: CPUID → AVX-512F > AVX2 > AVX+FMA > scalar
├── ARM64:   getauxval(HWCAP) → SVE > NEON > scalar
├── RISC-V:  /proc/cpuinfo → RVV > scalar
└── fallback: scalar C loops
```

Each SIMD level has 4 kernel variants (conv1d, conv_transpose1d, snake, add)
compiled with `__attribute__((target(...)))` for multi-versioning.

## GPU Backend Design

### CUDA/HIP Pattern
1. **Init**: Create device context, allocate scratch buffers
2. **Upload** (lazy, on first decode): Dequantize all weights on CPU, copy to GPU
3. **Decode**:
   a. Upload codebook indices → GPU
   b. RVQ lookup kernel
   c. Decoder graph kernels (same as CPU but GPU-parallel)
   d. Download PCM ← GPU
4. **Shutdown**: Free all GPU allocations

### Weight Format on GPU
All weights stored as `[Co, Ci, K]` float32 — dequantized and transposed on CPU
before upload. This matches the kernel access pattern `w[oc*Ci*K + ic*K + j]`.

## .txc Container Format

```
Offset  Size  Description
0       4     Magic "FBAZ" (ASCII)
4       2     Version (BE u16)
6       1     Flags (bit0 = stereo)
7       1     n_codebooks (1-12)
8       4     Parameter (BE u32)
12      4     Parameter (BE u32)
16+     var   Optional extended header
?       var   uint8 codebook_indices[n_frames × n_codebooks]
```

Header end auto-detected by `find_header_end()`:
first offset where `(file_size - offset) % n_codebooks == 0`.
