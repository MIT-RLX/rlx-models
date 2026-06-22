# LLVM ORC JIT Accelerator (Experimental)

Not merged into main codebase. This directory is an independent experiment.

## Goal

Replace hand-written SIMD intrinsics (AVX/AVX2/AVX-512/NEON) with LLVM IR
generation + ORC JIT compilation at runtime. Generate ONE kernel IR, let
LLVM optimize it for the current CPU's capabilities.

## Status

Placeholder — requires `llvm-dev` (apt) or `llvm-libs` (pacman) installed.

## Build

```bash
cd experimental/llvm-jit
clang -O3 -o llvm_jit_test llvm_jit_test.c $(llvm-config --cflags --ldflags --libs core orcjit native)
```

## Architecture

```
cpu_decoder_run()
  → conv1d_llvm(T, K, Ci, Co)          // one-time: generate + JIT
    → LLVMOrcCreateLLJIT()
    → conv1d_ir_builder()               // build LLVM IR
    → LLVMOrcAddLLJIT()                 // JIT compile
    → return function pointer
  → conv1d_jit(., ., ., ., T, K, Ci, Co) // use JIT'd code
```

Key benefit: compile ONE IR → optimal machine code for ANY CPU.
Covers x86 AVX/AVX2/AVX-512 and ARM64 NEON/SVE from the same IR.
