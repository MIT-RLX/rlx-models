/*
 * nc_kernels.h — CUDA kernel function declarations and launch configuration
 * for all ~200 kernels extracted from libnc_cuda.so.
 *
 * Each kernel has a known name (matching the original fatbin), dtype suffix,
 * expected block configuration, and shared memory requirement.
 *
 * This header is used by:
 *   - Template engine for generating CUDA C kernel sources
 *   - Kernel dispatch table in nc_cuda_device.c
 *   - Architecture port: HIP translator, Metal translator, etc.
 */

#ifndef NC_KERNELS_H
#define NC_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel launch configuration descriptor */
typedef struct {
    const char *name;           /* CUDA kernel function name */
    const char *op;             /* operation name (e.g., "add", "matmul_nn") */
    const char *dtype;          /* "f32", "f16", "bf16", "i32", "i8" */
    int block_x;               /* default block X dimension */
    int block_y;
    int block_z;
    int shared_mem;            /* shared memory per block (0 = none) */
    int grid_factor;           /* grid = (n_elements + block_x * grid_factor - 1) / (block_x * grid_factor) */
    const char *description;   /* human-readable */
} NCKernelDesc;

/* Element-wise kernels (block_x = 256, grid = ceil_div(n, 256)) */
#define ELEM_KERNEL(name, dtype_) \
    { "cu_" name "_" dtype_, name, dtype_, 256, 1, 1, 0, 1, name " (" dtype_ ")" }

/* Reduction kernels (use warp-level primitives) */
#define REDUCE_KERNEL(name, dtype_) \
    { "cu_" name "_" dtype_, name, dtype_, 256, 1, 1, 0, 1, name " (" dtype_ ")" }

/* MatMul kernels (specialized tiled configurations) */
#define MATMUL_KERNEL(name, dtype_, bx_, by_, smem_) \
    { "cu_" name "_" dtype_, name, dtype_, bx_, by_, 1, smem_, 1, name " (" dtype_ ")" }

/* All known kernels from libnc_cuda.so binary analysis */
static const NCKernelDesc nc_kernel_table[] = {
    /* Element-wise unary */
    ELEM_KERNEL("neg",       "f32"),
    ELEM_KERNEL("neg",       "f16"),
    ELEM_KERNEL("neg",       "bf16"),
    ELEM_KERNEL("exp",       "f32"),
    ELEM_KERNEL("exp",       "f16"),
    ELEM_KERNEL("exp",       "bf16"),
    ELEM_KERNEL("log",       "f32"),
    ELEM_KERNEL("log",       "f16"),
    ELEM_KERNEL("log",       "bf16"),
    ELEM_KERNEL("recip",     "f32"),
    ELEM_KERNEL("recip",     "f16"),
    ELEM_KERNEL("recip",     "bf16"),
    ELEM_KERNEL("relu",      "f32"),
    ELEM_KERNEL("relu",      "f16"),
    ELEM_KERNEL("relu",      "bf16"),
    ELEM_KERNEL("gelu",      "f32"),
    ELEM_KERNEL("gelu",      "f16"),
    ELEM_KERNEL("gelu",      "bf16"),
    ELEM_KERNEL("sigmoid",   "f32"),
    ELEM_KERNEL("sigmoid",   "f16"),
    ELEM_KERNEL("sigmoid",   "bf16"),
    ELEM_KERNEL("tanh",      "f32"),
    ELEM_KERNEL("tanh",      "f16"),
    ELEM_KERNEL("tanh",      "bf16"),
    ELEM_KERNEL("swish",     "f32"),
    ELEM_KERNEL("swish",     "f16"),
    ELEM_KERNEL("swish",     "bf16"),
    ELEM_KERNEL("snake",     "f32"),
    ELEM_KERNEL("snake",     "f16"),
    ELEM_KERNEL("snake",     "bf16"),
    ELEM_KERNEL("sqr_relu",  "f32"),
    ELEM_KERNEL("sqr_relu",  "f16"),
    ELEM_KERNEL("sqr_relu",  "bf16"),

    /* Element-wise binary */
    ELEM_KERNEL("add",       "f32"),
    ELEM_KERNEL("add",       "f16"),
    ELEM_KERNEL("add",       "bf16"),
    ELEM_KERNEL("mul",       "f32"),
    ELEM_KERNEL("mul",       "f16"),
    ELEM_KERNEL("mul",       "bf16"),
    ELEM_KERNEL("mul_dup0",  "f32"),
    ELEM_KERNEL("mul_dup0",  "f16"),
    ELEM_KERNEL("mul_dup0",  "bf16"),
    ELEM_KERNEL("add_col",   "f32"),
    ELEM_KERNEL("add_col",   "f16"),
    ELEM_KERNEL("add_col",   "bf16"),
    ELEM_KERNEL("cmul",      "f32"),
    ELEM_KERNEL("cmul",      "f16"),
    ELEM_KERNEL("cmul",      "bf16"),
    ELEM_KERNEL("cmul_4d",   "f32"),
    ELEM_KERNEL("cmul_4d",   "f16"),
    ELEM_KERNEL("cmul_4d",   "bf16"),
    ELEM_KERNEL("cmul_planar", "f32"),
    ELEM_KERNEL("cmul_planar", "f16"),
    ELEM_KERNEL("cmul_planar", "bf16"),
    ELEM_KERNEL("lerp",      "f32"),
    ELEM_KERNEL("lerp",      "f16"),
    ELEM_KERNEL("lerp",      "bf16"),

    /* Activation backward */
    ELEM_KERNEL("relu_bw",    "f32"),
    ELEM_KERNEL("relu_bw",    "f16"),
    ELEM_KERNEL("relu_bw",    "bf16"),
    ELEM_KERNEL("gelu_bw",    "f32"),
    ELEM_KERNEL("gelu_bw",    "f16"),
    ELEM_KERNEL("gelu_bw",    "bf16"),
    ELEM_KERNEL("sigmoid_bw", "f32"),
    ELEM_KERNEL("sigmoid_bw", "f16"),
    ELEM_KERNEL("sigmoid_bw", "bf16"),
    ELEM_KERNEL("tanh_bw",    "f32"),
    ELEM_KERNEL("tanh_bw",    "f16"),
    ELEM_KERNEL("tanh_bw",    "bf16"),
    ELEM_KERNEL("swish_bw",   "f32"),
    ELEM_KERNEL("swish_bw",   "f16"),
    ELEM_KERNEL("swish_bw",   "bf16"),
    ELEM_KERNEL("lerp_bw",    "f32"),
    ELEM_KERNEL("lerp_bw",    "f16"),
    ELEM_KERNEL("lerp_bw",    "bf16"),

    /* Softmax / log softmax */
    ELEM_KERNEL("soft_max2",       "f32"),
    ELEM_KERNEL("soft_max2",       "f16"),
    ELEM_KERNEL("soft_max2",       "bf16"),
    ELEM_KERNEL("soft_max_bw",     "f32"),
    ELEM_KERNEL("soft_max_bw",     "f16"),
    ELEM_KERNEL("soft_max_bw",     "bf16"),
    ELEM_KERNEL("soft_max2_int",   "f32"),
    ELEM_KERNEL("soft_max2_int",   "f16"),
    ELEM_KERNEL("soft_max2_int",   "bf16"),

    /* Reductions */
    REDUCE_KERNEL("reduce_sum",     "f32"),
    REDUCE_KERNEL("reduce_sum",     "f16"),
    REDUCE_KERNEL("reduce_sum",     "bf16"),
    REDUCE_KERNEL("reduce_max",     "f32"),
    REDUCE_KERNEL("reduce_max",     "f16"),
    REDUCE_KERNEL("reduce_max",     "bf16"),
    REDUCE_KERNEL("reduce_sumexp",  "f32"),
    REDUCE_KERNEL("reduce_sumexp",  "f16"),
    REDUCE_KERNEL("reduce_sumexp",  "bf16"),
    REDUCE_KERNEL("reduce_sum_sqr", "f32"),
    REDUCE_KERNEL("reduce_sum_sqr", "f16"),

    /* Normalization */
    { "cu_layer_norm1",           "layer_norm1",  "f32", 256, 1, 1, 0, 1, "LayerNorm fwd (f32)" },
    { "cu_layer_norm2_f32",       "layer_norm2",  "f32", 128, 2, 1, 0, 1, "LayerNorm 2D (f32)" },
    { "cu_layer_norm2_f16",       "layer_norm2",  "f16", 128, 2, 1, 0, 1, "LayerNorm 2D (f16)" },
    { "cu_layer_norm2_bf16",      "layer_norm2",  "bf16",128, 2, 1, 0, 1, "LayerNorm 2D (bf16)" },
    { "cu_layer_norm_bw_f32",     "layer_norm_bw", "f32",128, 2, 1, 0, 1, "LayerNorm bw (f32)" },
    { "cu_layer_norm_bw_f16",     "layer_norm_bw", "f16",128, 2, 1, 0, 1, "LayerNorm bw (f16)" },
    { "cu_layer_norm_bw_bf16",    "layer_norm_bw", "bf16",128, 2, 1, 0, 1, "LayerNorm bw (bf16)" },
    { "cu_group_norm1",           "group_norm1",   "f32", 256, 1, 1, 0, 1, "GroupNorm base" },
    { "cu_group_norm1_f16",       "group_norm1",   "f16", 256, 1, 1, 0, 1, "GroupNorm (f16)" },
    { "cu_group_norm1_bf16",      "group_norm1",   "bf16",256, 1, 1, 0, 1, "GroupNorm (bf16)" },
    { "cu_group_norm2",           "group_norm2",   "f32", 128, 2, 1, 0, 1, "GroupNorm 2nd pass" },

    /* Fused attention (forward) */
    { "cu_fused_att_64x16x8_f16",    "fused_att",   "f16", 64, 16, 1, 0, 1, "FusedAtt 64x16x8 (f16)" },
    { "cu_fused_att_64x16x8_bf16",   "fused_att",   "bf16",64, 16, 1, 0, 1, "FusedAtt 64x16x8 (bf16)" },
    { "cu_fused_att_64x16x128_f16",  "fused_att",   "f16", 64, 16, 1, 0, 1, "FusedAtt 64x16x128 (f16)" },
    { "cu_fused_att_64x16x128_bf16", "fused_att",   "bf16",64, 16, 1, 0, 1, "FusedAtt 64x16x128 (bf16)" },
    { "cu_fused_att_128x16x8_f16",   "fused_att",   "f16", 128,16, 1, 0, 1, "FusedAtt 128x16x8 (f16)" },
    { "cu_fused_att_128x16x8_bf16",  "fused_att",   "bf16",128,16, 1, 0, 1, "FusedAtt 128x16x8 (bf16)" },
    { "cu_fused_att_128x16x64_f16",  "fused_att",   "f16", 128,16, 1, 0, 1, "FusedAtt 128x16x64 (f16)" },
    { "cu_fused_att_128x16x64_bf16", "fused_att",   "bf16",128,16, 1, 0, 1, "FusedAtt 128x16x64 (bf16)" },
    { "cu_fused_att_256x16x16_f16",  "fused_att",   "f16", 256,16, 1, 0, 1, "FusedAtt 256x16x16 (f16)" },
    { "cu_fused_att_256x16x16_bf16", "fused_att",   "bf16",256,16, 1, 0, 1, "FusedAtt 256x16x16 (bf16)" },

    /* Fused attention (backward) */
    { "cu_fused_att_bw_64x128x32_f16",  "fused_att_bw", "f16", 64, 128, 1, 0, 1, "FusedAtt bw 64x128x32 (f16)" },
    { "cu_fused_att_bw_64x128x32_bf16", "fused_att_bw", "bf16",64, 128, 1, 0, 1, "FusedAtt bw 64x128x32 (bf16)" },
    { "cu_fused_att_bw_128x128x16_f16", "fused_att_bw", "f16", 128,128, 1, 0, 1, "FusedAtt bw 128x128x16 (f16)" },
    { "cu_fused_att_bw_128x128x16_bf16","fused_att_bw", "bf16",128,128, 1, 0, 1, "FusedAtt bw 128x128x16 (bf16)" },

    /* MatMul (non-transposed, NN) */
    MATMUL_KERNEL("matmul_nn_128x128x8_stage3_f32_f32",  "f32", 128,128, 8192),
    MATMUL_KERNEL("matmul_nn_128x256x32_stage3_f16_bf4", "f16", 128,256, 16384),
    MATMUL_KERNEL("matmul_nn_128x256x32_stage3_f16_bf8", "f16", 128,256, 16384),
    MATMUL_KERNEL("matmul_nn_128x256x32_stage3_bf16_bf4","bf16",128,256, 16384),
    MATMUL_KERNEL("matmul_nn_128x256x32_stage3_bf16_bf8","bf16",128,256, 16384),
    MATMUL_KERNEL("matmul_nn_64x64x32_stage3_f16_bf4",   "f16", 64, 64,  8192),
    MATMUL_KERNEL("matmul_nn_64x64x32_stage3_f16_bf8",   "f16", 64, 64,  8192),
    MATMUL_KERNEL("matmul_nn_64x64x32_stage3_bf16_bf4",  "bf16",64, 64,  8192),
    MATMUL_KERNEL("matmul_nn_64x64x32_stage3_bf16_bf8",  "bf16",64, 64,  8192),
    MATMUL_KERNEL("matmul_nn_64x128x64_stage3_f16_bf3",  "f16", 64, 128, 12288),
    MATMUL_KERNEL("matmul_nn_16x64x32_stage4_f16_bf4",   "f16", 16, 64,  4096),
    MATMUL_KERNEL("matmul_nn_16x64x32_stage4_f16_bf8",   "f16", 16, 64,  4096),
    MATMUL_KERNEL("matmul_nn_16x64x32_stage4_bf16_bf4",  "bf16",16, 64,  4096),
    MATMUL_KERNEL("matmul_nn_16x64x32_stage4_bf16_bf8",  "bf16",16, 64,  4096),
    MATMUL_KERNEL("matmul_nn_16x128x32_stage4_f16_bf4",  "f16", 16, 128, 4096),
    MATMUL_KERNEL("matmul_nn_16x128x32_stage4_bf16_bf4", "bf16",16, 128, 4096),
    MATMUL_KERNEL("matmul_nn_16x128x64_stage4_f16_bf3",  "f16", 16, 128, 6144),

    /* MatMul (NT and TN variants) */
    MATMUL_KERNEL("matmul_nt_128x64x64_stage3_bf8l_bf8l","bf16",128, 64, 16384),
    MATMUL_KERNEL("matmul_nt_128x128x8_stage3_f32_f32",  "f32", 128,128, 8192),
    MATMUL_KERNEL("matmul_tn_128x128x8_stage3_f32_f32",  "f32", 128,128, 8192),

    /* Convolution */
    { "cu_conv_64x64x32_stage3_f16",  "conv",  "f16", 64, 64, 1, 8192, 1, "Conv 64x64x32 (f16)" },
    { "cu_conv_128x128x32_stage3_f16", "conv",  "f16", 128,128, 1, 16384, 1, "Conv 128x128x32 (f16)" },

    /* Conversions / Quantization */
    ELEM_KERNEL("convert_f32_to_f16",  "f32"),
    ELEM_KERNEL("convert_f32_to_bf16", "f32"),
    ELEM_KERNEL("convert_f16_to_f32",  "f16"),
    ELEM_KERNEL("convert_bf16_to_f32", "bf16"),
    ELEM_KERNEL("convert_f32_to_e4m3", "f32"),
    ELEM_KERNEL("convert_f32_to_e5m2", "f32"),
    ELEM_KERNEL("convert_bf16_to_e4m3","bf16"),
    ELEM_KERNEL("convert_bf16_to_e5m2","bf16"),

    /* LSTM / RWKV */
    ELEM_KERNEL("lstm_clamped",    "f32"),
    ELEM_KERNEL("lstm_clamped",    "f16"),
    ELEM_KERNEL("lstm_clamped",    "bf16"),
    ELEM_KERNEL("rwkv_att",       "f32"),
    ELEM_KERNEL("rwkv_att",       "f16"),
    ELEM_KERNEL("rwkv_att",       "bf16"),

    /* Optimizers */
    ELEM_KERNEL("rmsprop",      "f32"),
    ELEM_KERNEL("rmsprop",      "f16"),
    ELEM_KERNEL("rmsprop",      "bf16"),
    { "cu_sparse_rmsprop_f32",  "sparse_rmsprop", "f32", 256, 1, 1, 0, 1, "Sparse RMSProp (f32)" },

    /* Random */
    ELEM_KERNEL("rnd_unif",     "f32"),
    ELEM_KERNEL("rnd_unif",     "f16"),
    ELEM_KERNEL("rnd_unif",     "bf16"),
    ELEM_KERNEL("rnd_gaussian", "f32"),
    ELEM_KERNEL("rnd_gaussian", "f16"),
    ELEM_KERNEL("rnd_gaussian", "bf16"),
    ELEM_KERNEL("rnd_dropout",  "f32"),
    ELEM_KERNEL("rnd_dropout",  "f16"),
    ELEM_KERNEL("rnd_dropout",  "bf16"),

    /* Memory operations */
    { "cu_memcpy2d_u8",  "memcpy2d", "u8",  16, 16, 1, 0, 1, "Memcpy 2D (u8)" },
    { "cu_memcpy3d_u8",  "memcpy3d", "u8",  8,  8,  8, 0, 1, "Memcpy 3D (u8)" },
    { "cu_memset2d_u8",  "memset2d", "u8",  16, 16, 1, 0, 1, "Memset 2D (u8)" },

    /* Sentinel */
    { NULL, NULL, NULL, 0, 0, 0, 0, 0, NULL }
};

/* Find kernel descriptor by name */
static inline const NCKernelDesc *nc_find_kernel(const char *name)
{
    for (int i = 0; nc_kernel_table[i].name != NULL; i++) {
        if (strcmp(nc_kernel_table[i].name, name) == 0)
            return &nc_kernel_table[i];
    }
    return NULL;
}

/* Find kernel descriptor by operation and dtype */
static inline const NCKernelDesc *nc_find_kernel_op(const char *op,
                                                      const char *dtype)
{
    for (int i = 0; nc_kernel_table[i].name != NULL; i++) {
        if (strcmp(nc_kernel_table[i].op, op) == 0 &&
            strcmp(nc_kernel_table[i].dtype, dtype) == 0)
            return &nc_kernel_table[i];
    }
    return NULL;
}

#ifdef __cplusplus
}
#endif

#endif /* NC_KERNELS_H */
