/* Stub implementations for Vulkan backend when USE_VULKAN is OFF */
#include "tsac_codec.h"
int  tsac_vk_init(void **priv) { (void)priv; return TSAC_ERR_BACKEND; }
int  tsac_vk_encode(void *priv, void *model, const float *pcm, int n_samples, int channels, int n_codebooks, int block_len, int **codebook_indices, int *n_frames)
{ (void)priv;(void)model;(void)pcm;(void)n_samples;(void)channels;(void)n_codebooks;(void)block_len;(void)codebook_indices;(void)n_frames;return TSAC_ERR_BACKEND; }
int  tsac_vk_decode(void *priv, void *model, const int *codebook_indices, int n_frames, int n_codebooks, int block_len, int channels, float *pcm, int n_samples)
{ (void)priv;(void)model;(void)codebook_indices;(void)n_frames;(void)n_codebooks;(void)block_len;(void)channels;(void)pcm;(void)n_samples;return TSAC_ERR_BACKEND; }
void tsac_vk_shutdown(void *priv) { (void)priv; }
