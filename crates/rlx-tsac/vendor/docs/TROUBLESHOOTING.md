# tsac-ng Troubleshooting Guide

## Build Failures

### "fatal error: llvm-c/Transforms/Scalar.h: No such file or directory"
**Cause**: LLVM 22 removed the old pass manager headers.
**Fix**: Remove `#include <llvm-c/Transforms/Scalar.h>` and `LLVMPassManagerRef` code.
Use the new pass builder API or skip optimization passes entirely.

### hipLaunchKernelGGL was not declared
**Cause**: ROCm 7.2 deprecated `hipLaunchKernelGGL` macro.
**Fix**: Replace with triple-chevron syntax: `kernel<<<grid, block, shmem, stream>>>(args...)`.

### blockIdx/blockDim/threadIdx not declared
**Cause**: `.hip.cpp` files not being compiled as HIP language.
**Fix** (CMakeLists.txt):
```cmake
set_source_files_properties(${HIP_SOURCE} PROPERTIES LANGUAGE HIP)
```

## Runtime Errors

### "Error: operation failed (code -3)" — TSAC_ERR_FORMAT

| Scenario | Cause | Fix |
|----------|-------|-----|
| Decoding .txc file | Version byte order mismatch | Ensure file written with BE version (our txc_write does this now) |
| Decoding .txc file | n_codebooks reads as 0 | Re-serialize with manual byte layout (not struct memcpy) |
| Encoding WAV file | WAV format tag not 1 or 3 | Use PCM int16 (format=1) or IEEE float (format=3) |
| Loading empty file | Not an actual .txc file | Check file existence and size |

### "Error: operation failed (code -7)" — TSAC_ERR_BACKEND

| Scenario | Cause | Fix |
|----------|-------|-----|
| `--cuda` on CPU-only build | CUDA stub returns error | Use CPU backend, or build with -DUSE_CUDA=ON |
| `--hip` on non-ROCm system | HIP stub returns error | Use CPU backend, or install ROCm 7.x |
| CPU encode | Not implemented | Use `--cuda` or `--hip` for encode |
| Vulkan without libvulkan | dlopen fails | Install vulkan-loader or use CPU backend |

### NaN in Output

| Pattern | Cause | Fix |
|---------|-------|-----|
| Channels 42-95 all NaN | Float32 weight_v misread as uint8 | Update model_loader elem_size detection |
| First 32 samples/ch NaN | conv_transpose edge effect | Ignore (border artifact) |
| All output NaN | Model missing decoder blocks | Use full 82MB model (322 tensors) |
| Non-deterministic NaN | CUDA in-place race condition | Use temp buffer for inner conv |
| Progressive overflow → NaN | Weight format mismatch (GPU) | Transpose to [Co,Ci,K] before upload |

### GPU Memory Access Fault

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Page not present" | GPU kernel reads past buffer | Check data_size vs kernel element count |
| "Memory access fault" | CPU pointer passed to GPU kernel | Use cudaMalloc/hipMalloc, not malloc |
| "Unspecified launch failure" | Grid too large or kernel timeout | Check grid dimensions and block size |

## Model Issues

### Decoder output is all zeros
1. Check model file size: should be 82MB (dac_stereo_q8.bin) or 26MB (incomplete)
2. Check tensor count: `322` = full, `206` = missing decoder blocks 1-4
3. If 26MB/206 tensors: re-download full model from GitHub Release

### Model not found
```
Error: failed to initialize codec
```
1. Check model search paths (hardcoded in main.c:173):
   - `models/tsac` (relative to CWD)
   - `/usr/share/tsac`
   - Termux: use `-m /data/data/com.termux/files/home/...`
2. Use `-m /path/to/models` to specify explicit model directory

## CLI Issues

### Options after command are ignored
```bash
tsac d -v input.txc output.wav   # -v is IGNORED!
```
**Cause**: `optstring "+T:q:..."` enables POSIX mode — all options must come BEFORE the command character.
**Workaround**: `tsac -v d input.txc output.wav` (options before command)

### WAV format not recognized
```bash
Error: operation failed (code -3)
```
**Check**: WAV must be PCM int16 (format=1) or IEEE float (format=3).
Compressed WAV (format≠1/≠3) is rejected.
