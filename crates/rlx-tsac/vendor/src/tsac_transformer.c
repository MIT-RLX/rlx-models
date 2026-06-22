#include "tsac_transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================================================================ */
/*  Tensor lookup helper                                            */
/* ================================================================ */
static DACTensor *tf_find(DACTensor *ts, int nt, const char *name) {
    for (int i = 0; i < nt; i++)
        if (strcmp(ts[i].name, name) == 0) return &ts[i];
    return NULL;
}

static float *tf_data(DACTensor *t) {
    return t ? (float *)t->data : NULL;
}

/* ================================================================ */
/*  Weight loading                                                  */
/* ================================================================ */
int tsac_transformer_load(TSACTransformer *tf, DACTensor *ts, int nt) {
    if (!tf || !ts) return TSAC_ERR_PARAM;
    memset(tf, 0, sizeof(*tf));

    /* Position embeddings */
    DACTensor *wpe = tf_find(ts, nt, "wpe");
    if (wpe) tf->wpe = tf_data(wpe);

    /* Load each layer */
    for (int h = 0; h < TF_N_LAYERS && h < 12; h++) {
        char name[128];
        TfLayer *L = &tf->layers[h];

        /* Layer norms */
        snprintf(name, sizeof(name), "h%d/ln_1/g", h);
        L->ln1.g = tf_data(tf_find(ts, nt, name));
        snprintf(name, sizeof(name), "h%d/ln_1/b", h);
        L->ln1.b = tf_data(tf_find(ts, nt, name));
        snprintf(name, sizeof(name), "h%d/ln_2/g", h);
        L->ln2.g = tf_data(tf_find(ts, nt, name));
        snprintf(name, sizeof(name), "h%d/ln_2/b", h);
        L->ln2.b = tf_data(tf_find(ts, nt, name));

        /* Attention */
        snprintf(name, sizeof(name), "h%d/attn/c_attn/w", h);
        L->attn.c_attn_w = tf_data(tf_find(ts, nt, name));
        snprintf(name, sizeof(name), "h%d/attn/c_attn/b", h);
        L->attn.c_attn_b = tf_data(tf_find(ts, nt, name));
        snprintf(name, sizeof(name), "h%d/attn/c_proj/w", h);
        L->attn.c_proj_w = tf_data(tf_find(ts, nt, name));
        snprintf(name, sizeof(name), "h%d/attn/c_proj/b", h);
        L->attn.c_proj_b = tf_data(tf_find(ts, nt, name));

        /* MLP */
        snprintf(name, sizeof(name), "h%d/mlp/c_fc/w", h);
        L->mlp.c_fc_w = tf_data(tf_find(ts, nt, name));
        snprintf(name, sizeof(name), "h%d/mlp/c_fc/b", h);
        L->mlp.c_fc_b = tf_data(tf_find(ts, nt, name));
        snprintf(name, sizeof(name), "h%d/mlp/c_proj/w", h);
        L->mlp.c_proj_w = tf_data(tf_find(ts, nt, name));
        snprintf(name, sizeof(name), "h%d/mlp/c_proj/b", h);
        L->mlp.c_proj_b = tf_data(tf_find(ts, nt, name));
    }

    /* Final layer */
    DACTensor *ln_f_g = tf_find(ts, nt, "ln_f/g");
    DACTensor *ln_f_b = tf_find(ts, nt, "ln_f/b");
    tf->ln_f_g = tf_data(ln_f_g);
    tf->ln_f_b = tf_data(ln_f_b);
    tf->g = tf_data(tf_find(ts, nt, "g"));

    return TSAC_OK;
}

/* ================================================================ */
/*  Core operations                                                 */
/* ================================================================ */

/* Layer normalization: y = g * (x - mean) / sqrt(var + eps) + b */
static void layer_norm(float *y, const float *x, const float *g, const float *b,
                       int n, float eps) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++)
        y[i] = g[i] * (x[i] - mean) * inv_std + b[i];
}

/* GELU activation */
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

/* Matrix multiply: C[M,N] = A[M,K] * B[K,N] */
static void matmul(float *C, const float *A, const float *B,
                   int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

/* Add bias: y[i] = x[i] + b[i] */
static void add_bias(float *y, const float *x, const float *b, int n) {
    for (int i = 0; i < n; i++) y[i] = x[i] + b[i];
}

/* ================================================================ */
/*  Attention (single sequence)                                     */
/* ================================================================ */
static void attention_forward(float *out, const float *x,
                               const TfAttention *attn, int seq_len) {
    int D = TF_D_MODEL, H = TF_N_HEAD, DH = TF_D_HEAD;
    /* x: [seq, D] */

    /* Fused QKV projection: qkv = x @ W + b  [seq, 3*D] */
    float *qkv = (float *)malloc(seq_len * 3 * D * sizeof(float));
    matmul(qkv, x, attn->c_attn_w, seq_len, D, 3 * D);
    add_bias(qkv, qkv, attn->c_attn_b, seq_len * 3 * D);

    /* Split Q, K, V: each [seq, D] */
    float *Q = qkv, *K = qkv + seq_len * D, *V = qkv + 2 * seq_len * D;

    /* Scaled dot-product attention per head */
    float *attn_out = (float *)calloc(seq_len * D, sizeof(float));
    float *scores = (float *)malloc(seq_len * seq_len * sizeof(float));

    for (int h = 0; h < H; h++) {
        int off = h * DH;
        float scale = 1.0f / sqrtf((float)DH);

        /* scores = Q_h @ K_h^T * scale */
        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < seq_len; j++) {
                float s = 0;
                for (int k = 0; k < DH; k++)
                    s += Q[i * D + off + k] * K[j * D + off + k];
                scores[i * seq_len + j] = s * scale;
                /* Causal mask */
                if (j > i) scores[i * seq_len + j] = -1e9f;
            }

        /* Softmax per row */
        for (int i = 0; i < seq_len; i++) {
            float maxv = scores[i * seq_len];
            for (int j = 1; j < seq_len; j++)
                if (scores[i * seq_len + j] > maxv) maxv = scores[i * seq_len + j];
            float sum = 0;
            for (int j = 0; j < seq_len; j++) {
                scores[i * seq_len + j] = expf(scores[i * seq_len + j] - maxv);
                sum += scores[i * seq_len + j];
            }
            for (int j = 0; j < seq_len; j++)
                scores[i * seq_len + j] /= sum;
        }

        /* weighted sum of V */
        for (int i = 0; i < seq_len; i++)
            for (int k = 0; k < DH; k++) {
                float s = 0;
                for (int j = 0; j < seq_len; j++)
                    s += scores[i * seq_len + j] * V[j * D + off + k];
                attn_out[i * D + off + k] = s;
            }
    }

    /* Output projection */
    matmul(out, attn_out, attn->c_proj_w, seq_len, D, D);
    add_bias(out, out, attn->c_proj_b, seq_len * D);

    free(qkv);
    free(scores);
    free(attn_out);
}

/* ================================================================ */
/*  MLP (FFN) forward                                               */
/* ================================================================ */
static void mlp_forward(float *out, const float *x, const TfMLP *mlp, int seq_len) {
    int D = TF_D_MODEL, F = TF_FFN_DIM;
    float *hidden = (float *)malloc(seq_len * F * sizeof(float));

    /* FC up: [seq, D] @ [D, F] → [seq, F] */
    matmul(hidden, x, mlp->c_fc_w, seq_len, D, F);
    add_bias(hidden, hidden, mlp->c_fc_b, seq_len * F);

    /* GELU activation */
    for (int i = 0; i < seq_len * F; i++) hidden[i] = gelu(hidden[i]);

    /* FC down: [seq, F] @ [F, D] → [seq, D] */
    matmul(out, hidden, mlp->c_proj_w, seq_len, F, D);
    add_bias(out, out, mlp->c_proj_b, seq_len * D);

    free(hidden);
}

/* ================================================================ */
/*  Single layer forward                                            */
/* ================================================================ */
static void layer_forward(float *out, const float *x, const TfLayer *L,
                          int seq_len) {
    int D = TF_D_MODEL;
    float *buf = (float *)malloc(seq_len * D * sizeof(float));

    /* Sub-layer 1: Self-attention with pre-norm */
    layer_norm(buf, x, L->ln1.g, L->ln1.b, D, 1e-5f);
    /* Apply per-position (same norm for all positions) */
    for (int s = 1; s < seq_len; s++)
        layer_norm(buf + s * D, x + s * D, L->ln1.g, L->ln1.b, D, 1e-5f);

    float *attn_out = (float *)malloc(seq_len * D * sizeof(float));
    attention_forward(attn_out, buf, &L->attn, seq_len);

    /* Residual */
    for (int i = 0; i < seq_len * D; i++) buf[i] = x[i] + attn_out[i];

    /* Sub-layer 2: MLP with pre-norm */
    for (int s = 0; s < seq_len; s++)
        layer_norm(attn_out + s * D, buf + s * D, L->ln2.g, L->ln2.b, D, 1e-5f);

    mlp_forward(out, attn_out, &L->mlp, seq_len);

    /* Residual */
    for (int i = 0; i < seq_len * D; i++) out[i] += buf[i];

    free(buf);
    free(attn_out);
}

/* ================================================================ */
/*  Full transformer forward                                        */
/* ================================================================ */
int tsac_transformer_forward(TSACTransformer *tf,
                              const int *input_ids, const int *position_ids,
                              int seq_len, float *logits) {
    if (!tf || !logits || seq_len <= 0 || seq_len > TF_MAX_SEQ)
        return TSAC_ERR_PARAM;

    int D = TF_D_MODEL;
    float *hidden = (float *)calloc(seq_len * D, sizeof(float));
    float *next_hidden = (float *)malloc(seq_len * D * sizeof(float));

    /* Add position embeddings */
    if (tf->wpe) {
        for (int s = 0; s < seq_len; s++) {
            int pos = position_ids ? position_ids[s] : s;
            if (pos < TF_MAX_SEQ) {
                /* wpe[pos][d] for d=0..11, replicate to D? */
                /* wpe is [max_seq, 12], d_model=512, so tile 12→512? */
                for (int d = 0; d < 12 && d < D; d++)
                    hidden[s * D + d] += tf->wpe[pos * 12 + d];
            }
        }
    }

    /* Process through layers */
    for (int h = 0; h < TF_N_LAYERS; h++) {
        layer_forward(next_hidden, hidden, &tf->layers[h], seq_len);
        float *tmp = hidden; hidden = next_hidden; next_hidden = tmp;
    }

    /* Final layer norm */
    for (int s = 0; s < seq_len; s++)
        layer_norm(logits + s * D, hidden + s * D,
                   tf->ln_f_g, tf->ln_f_b, D, 1e-5f);

    /* Output projection (if g exists) */
    if (tf->g) {
        for (int s = 0; s < seq_len; s++) {
            float dot = 0;
            for (int d = 0; d < D; d++)
                dot += logits[s * D + d] * tf->g[d];
            logits[s * D] = dot;  /* scalar output (next token logit) */
        }
    }

    free(hidden);
    free(next_hidden);
    return TSAC_OK;
}

void tsac_transformer_free(TSACTransformer *tf) {
    /* Weights are owned by the tensor array, not the transformer */
    memset(tf, 0, sizeof(*tf));
}
