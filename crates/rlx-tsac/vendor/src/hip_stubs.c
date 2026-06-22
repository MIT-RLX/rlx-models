#include "tsac_codec.h"
/* Stub implementation — real code is in the backend-specific modules. */

int tsac_hip_init(void **priv)
{
    (void)priv;
    return TSAC_ERR_BACKEND;
}

void tsac_hip_shutdown(void *priv)
{
    (void)priv;
}

int tsac_hip_encode(void *priv, void *model,
                    const float *pcm, int n_samples, int channels,
                    int n_codebooks, int block_len,
                    int **codebook_indices, int *n_frames)
{
    (void)priv; (void)model; (void)pcm;
    (void)n_samples; (void)channels;
    (void)n_codebooks; (void)block_len;
    (void)codebook_indices; (void)n_frames;
    return TSAC_ERR_BACKEND;
}

int tsac_hip_decode(void *priv, void *model,
                    const int *codebook_indices, int n_frames,
                    int n_codebooks, int block_len, int channels,
                    float *pcm, int n_samples)
{
    (void)priv; (void)model;
    (void)codebook_indices; (void)n_frames;
    (void)n_codebooks; (void)block_len; (void)channels;
    (void)pcm; (void)n_samples;
    return TSAC_ERR_BACKEND;
}
