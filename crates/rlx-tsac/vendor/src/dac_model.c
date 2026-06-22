/* dac_model.c — dac model for tsac-ng. */
#include "dac_model.h"
/* Wrapper around cpu_decoder.c for decoder dispatch. */
#include "model_loader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* CPU decoder for dac_model_decode (fallback) */
int cpu_decoder_run(DACTensor *tensors, int n_tensors,
                     const int *codes, int n_frames, int n_codebooks,
                     float *pcm, int n_samples, int channels,
                     int n_threads);

/* HIP decoder — defined in dac_decoder.hip.cpp */
int dac_decoder_run(DACTensor *tensors, int n_tensors,
                     const int *codes, int n_frames, int n_codebooks,
                     float *pcm, int n_samples, int channels);

DACModel *dac_model_create(void)
{
    DACModel *m = (DACModel *)calloc(1, sizeof(DACModel));
    if (!m) return NULL;
    return m;
}

void dac_model_destroy(DACModel *model)
{
    if (!model) return;
    model_loader_free(model);
    free(model);
}

DACTensor *dac_model_find(DACModel *model, const char *name)
{
    if (!model || !name) return NULL;
    for (int i = 0; i < model->n_tensors; i++) {
        if (strcmp(model->tensors[i].name, name) == 0)
            return &model->tensors[i];
    }
    return NULL;
}

int dac_model_decode(DACModel *model,
                     const int *codebook_indices, int n_frames,
                     int n_codebooks, int block_len, int channels,
                     float *pcm, int n_samples,
                     int n_threads)
{
    if (!model || !model->tensors || !codebook_indices || !pcm)
        return TSAC_ERR_PARAM;

    /* Try HIP decoder first, fall back to CPU */
    int ret = cpu_decoder_run(model->tensors, model->n_tensors,
                                codebook_indices, n_frames, n_codebooks,
                                pcm, n_samples, channels,
                                n_threads);
    return (ret == 0) ? TSAC_OK : TSAC_ERR_BACKEND;
}
