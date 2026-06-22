#include "dac_model.h"
#include "tsac_transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/usr/share/tsac/tsac_stereo_q8.bin";
    
    /* Load model */
    DACModel *model = dac_model_create();
    if (!model) { fprintf(stderr, "Failed to create model\n"); return 1; }
    
    int ret = dac_model_load(model, model_path);
    if (ret != 0) { fprintf(stderr, "Failed to load model: %d\n", ret); return 1; }
    
    /* Load transformer */
    TSACTransformer tf;
    ret = tsac_transformer_load(&tf, model->tensors, model->n_tensors);
    if (ret != 0) { fprintf(stderr, "Failed to load transformer\n"); return 1; }
    
    printf("Transformer loaded: %d layers, %d heads, d_model=%d\n",
           TF_N_LAYERS, TF_N_HEAD, TF_D_MODEL);
    
    /* Test forward pass with zero input */
    int pos_ids[1] = {0};
    float logits[TF_MAX_SEQ * TF_D_MODEL];
    ret = tsac_transformer_forward(&tf, NULL, pos_ids, 1, logits);
    
    if (ret == 0) {
        printf("Forward pass OK. Logits[0:4] = %.4f, %.4f, %.4f, %.4f\n",
               logits[0], logits[1], logits[2], logits[3]);
        /* Check for NaN */
        int nan_count = 0;
        for (int i = 0; i < TF_D_MODEL; i++)
            if (isnan(logits[i])) nan_count++;
        printf("NaN count: %d/%d\n", nan_count, TF_D_MODEL);
    } else {
        printf("Forward pass failed: %d\n", ret);
    }
    
    tsac_transformer_free(&tf);
    dac_model_destroy(model);
    return ret;
}
