/* tsac_codec.h — Public API for TSAC neural audio codec. */
#ifndef TSAC_CODEC_H
#define TSAC_CODEC_H

#include "tsac.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Backend-specific init/shutdown */
int tsac_cuda_init(void **priv);
int tsac_cuda_encode(void *priv, void *model,
                     const float *pcm, int n_samples, int channels,
                     int n_codebooks, int block_len,
                     int **codebook_indices, int *n_frames);
int tsac_cuda_decode(void *priv, void *model,
                     const int *codebook_indices, int n_frames,
                     int n_codebooks, int block_len, int channels,
                     float *pcm, int n_samples);
void tsac_cuda_shutdown(void *priv);

int tsac_hip_init(void **priv);
int tsac_hip_encode(void *priv, void *model,
                    const float *pcm, int n_samples, int channels,
                    int n_codebooks, int block_len,
                    int **codebook_indices, int *n_frames);
int tsac_hip_decode(void *priv, void *model,
                    const int *codebook_indices, int n_frames,
                    int n_codebooks, int block_len, int channels,
                    float *pcm, int n_samples);
void tsac_hip_shutdown(void *priv);

/* Vulkan backend */
int  tsac_vk_init(void **priv);
int  tsac_vk_encode(void *priv, void *model,
                     const float *pcm, int n_samples, int channels,
                     int n_codebooks, int block_len,
                     int **codebook_indices, int *n_frames);
int  tsac_vk_decode(void *priv, void *model,
                     const int *codebook_indices, int n_frames,
                     int n_codebooks, int block_len, int channels,
                     float *pcm, int n_samples);
void tsac_vk_shutdown(void *priv);

/* LLVM JIT backend (experimental) */
int  tsac_llvm_init(void **priv);
int  tsac_llvm_encode(void *priv, void *model,
                       const float *pcm, int n_samples, int channels,
                       int n_codebooks, int block_len,
                       int **codebook_indices, int *n_frames);
int  tsac_llvm_decode(void *priv, void *model,
                       const int *codebook_indices, int n_frames,
                       int n_codebooks, int block_len, int channels,
                       float *pcm, int n_samples);
void tsac_llvm_shutdown(void *priv);

#ifdef __cplusplus
}
#endif

#endif /* TSAC_CODEC_H */
