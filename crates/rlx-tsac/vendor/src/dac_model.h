#ifndef DAC_MODEL_H
#define DAC_MODEL_H

#include "tsac.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* DAC model — stores loaded params and context */
#define DAC_MAX_PARAMS 512

typedef struct {
    char     name[128];
    int      ndims;
    int      dims[8];
    uint8_t *data;
    int      data_size;
    int      elem_size;
    void    *dev;        /* HIP/CUDA device pointer */
    float   *dev_f32;    /* dequantized fp32 device ptr */
} DACTensor;

typedef struct DACModel {
    DACTensor *tensors;
    int        n_tensors;
} DACModel;

/* Create an empty DAC model */
DACModel *dac_model_create(void);

/* Destroy the DAC model */
void dac_model_destroy(DACModel *model);

/* Find a tensor by name */
DACTensor *dac_model_find(DACModel *model, const char *name);

/*
 * Decode RVQ codebook indices back to PCM audio.
 */
int dac_model_decode(DACModel *model,
                     const int *codebook_indices, int n_frames,
                     int n_codebooks, int block_len, int channels,
                     float *pcm, int n_samples,
                     int n_threads);

#ifdef __cplusplus
}
#endif

#endif /* DAC_MODEL_H */
