/*
 * llvm_stubs.c — LLVM JIT backend stub (experimental).
 * Falls back to CPU when USE_LLVM is not enabled.
 */

#include "tsac_codec.h"
#include <stdio.h>

int tsac_llvm_init(void **priv)
{
    (void)priv;
    fprintf(stderr, "[llvm] LLVM JIT backend not compiled in (stub)\n");
    return TSAC_ERR_BACKEND;
}

int tsac_llvm_encode(void *priv, void *model,
                      const float *pcm, int n_samples, int channels,
                      int n_codebooks, int block_len,
                      int **codebook_indices, int *n_frames)
{
    (void)priv; (void)model; (void)pcm; (void)n_samples; (void)channels;
    (void)n_codebooks; (void)block_len; (void)codebook_indices; (void)n_frames;
    return TSAC_ERR_BACKEND;
}

int tsac_llvm_decode(void *priv, void *model,
                      const int *codebook_indices, int n_frames,
                      int n_codebooks, int block_len, int channels,
                      float *pcm, int n_samples)
{
    (void)priv; (void)model; (void)codebook_indices; (void)n_frames;
    (void)n_codebooks; (void)block_len; (void)channels; (void)pcm; (void)n_samples;
    return TSAC_ERR_BACKEND;
}

void tsac_llvm_shutdown(void *priv)
{
    (void)priv;
}
