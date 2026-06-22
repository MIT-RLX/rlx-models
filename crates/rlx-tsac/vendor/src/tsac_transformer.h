#include "dac_model.h"
#ifndef TSAC_TRANSFORMER_H
#define TSAC_TRANSFORMER_H

#include "dac_model.h"

/* GPT-2 style transformer for Normal TXC decode */
/* Architecture: 12 layers, d_model=512, n_head=4, d_head=128 */

#define TF_N_LAYERS 12
#define TF_D_MODEL 512
#define TF_N_HEAD 4
#define TF_D_HEAD 128  /* 512 / 4 */
#define TF_FFN_DIM 2048  /* 512 * 4 */
#define TF_MAX_SEQ 512

/* Layer normalization */
typedef struct {
    float *g, *b;  /* [d_model] */
} LayerNorm;

/* Attention with fused QKV projection */
typedef struct {
    float *c_attn_w, *c_attn_b;  /* [d_model, 3*d_model], [3*d_model] */
    float *c_proj_w, *c_proj_b;  /* [d_model, d_model], [d_model] */
} TfAttention;

/* MLP (FFN) */
typedef struct {
    float *c_fc_w, *c_fc_b;   /* [d_model, FFN_dim], [FFN_dim] */
    float *c_proj_w, *c_proj_b; /* [FFN_dim, d_model], [d_model] */
} TfMLP;

/* Transformer layer */
typedef struct {
    LayerNorm ln1, ln2;
    TfAttention attn;
    TfMLP mlp;
} TfLayer;

/* Full transformer model */
typedef struct {
    float *wpe;       /* position embeddings [max_seq, 12] */
    float *wte;       /* token embeddings (if used) */
    TfLayer layers[TF_N_LAYERS];
    float *ln_f_g, *ln_f_b;  /* final layer norm */
    float *g;         /* output projection */
} TSACTransformer;

/* Load transformer weights from tsac model */
int tsac_transformer_load(TSACTransformer *tf, DACTensor *tensors, int n_tensors);

/* Forward pass: compute logits from input_ids + position_ids */
/* input_ids: [seq_len], position_ids: [seq_len] */
/* logits: [seq_len, vocab_size] (or custom output) */
int tsac_transformer_forward(TSACTransformer *tf,
                              const int *input_ids, const int *position_ids,
                              int seq_len, float *logits);

/* Free transformer resources */
void tsac_transformer_free(TSACTransformer *tf);

#endif
