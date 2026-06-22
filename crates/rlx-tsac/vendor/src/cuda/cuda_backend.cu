/*
 * cuda_backend.cu — CUDA backend for TSAC-ng codec.
 * Exposes tsac_cuda_init / tsac_cuda_decode / tsac_cuda_encode / tsac_cuda_shutdown.
 * Weights are uploaded lazily on first decode call and cached on GPU.
 */

#include "../tsac_codec.h"
#include "../dac_model.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[cuda] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return TSAC_ERR_BACKEND; \
    } \
} while(0)

#define CUDA_CHK_V(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) \
        fprintf(stderr, "[cuda] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
} while(0)

/* Forward declarations for kernel launchers (defined in cuda_kernels.cu) */
extern "C" {
cudaError_t launch_conv1d(float *d_o, const float *d_x, const float *d_w, const float *d_b,
                           int T, int K, int Ci, int Co, cudaStream_t stream);
cudaError_t launch_conv1d_strided(float *d_o, const float *d_x, const float *d_w, const float *d_b,
                                   int T_out, int K, int Ci, int Co, int T_in, int stride, cudaStream_t stream);
cudaError_t launch_conv_transpose1d(float *d_o, const float *d_x, const float *d_w,
                                     int Ti, int To, int K, int Ci, int Co, cudaStream_t stream);
cudaError_t launch_snake(float *d_x, const float *d_alpha, int n, int C, cudaStream_t stream);
cudaError_t launch_add(float *d_o, const float *d_a, const float *d_b, int n, cudaStream_t stream);
cudaError_t launch_add_bias(float *d_x, const float *d_bias, int T, int C, cudaStream_t stream);
cudaError_t launch_rvq_lookup(float *d_features, const int *d_codes,
                               const float *d_cb_data, const int *d_cb_offsets,
                               int n_frames, int n_cb, int rvq_dim, cudaStream_t stream);
cudaError_t launch_rvq_quantize(int *d_indices, const float *d_features,
                                 const float *d_codebook, int n_frames, int rvq_dim, int entries, cudaStream_t stream);
cudaError_t launch_rvq_subtract(float *d_features, const float *d_codebook,
                                 const int *d_indices, int n_frames, int rvq_dim, cudaStream_t stream);
cudaError_t launch_tanh_clip(float *d_x, int n, cudaStream_t stream);
}

/* CPU-side dequantization helper */
extern "C" float *dequant_weights(const DACTensor *weight_v, const DACTensor *weight_g,
                                   const DACTensor *bias,
                                   int *out_Ci, int *out_K, int *out_Co, int *is_conv_transpose);

/* CPU-side tensor finder */
static DACTensor *tf(DACTensor *ts, int nt, const char *name) {
    for (int i = 0; i < nt; i++)
        if (!strcmp(ts[i].name, name)) return &ts[i];
    return NULL;
}

/* ================================================================ */
/*  GPU weight cache                                                  */
/* ================================================================ */

#define MAX_GPU_WEIGHTS 128

typedef struct {
    char     name[128];
    float   *d_data;    /* GPU pointer to dequantized float32 weights */
    int      Ci, K, Co;
    int      is_convt;  /* 1 = conv_transpose layer */
} GpuWeight;

typedef struct {
    int          initialized;
    int          weights_uploaded;

    /* GPU weights */
    GpuWeight    gpu_weights[MAX_GPU_WEIGHTS];
    int          n_gpu_weights;

    /* Codebook data (uploaded once) */
    float       *d_cb_data;      /* all codebooks concatenated [total_cb_entries * 1024] */
    int         *d_cb_offsets;   /* per-codebook base offset in d_cb_data */
    int          cb_offsets[13]; /* CPU copy for upload */
    int          n_cb;
    int          cb_dim;
    int          cb_entries;

    /* GPU buffers for intermediate activations (reused across calls) */
    float       *d_buf[8];
    int          buf_sizes[8];
    int          n_bufs;

    /* Scratch buffers */
    int         *d_codes;        /* codebook indices */
    float       *d_features;     /* RVQ output */

    cudaStream_t stream;
} CudaBackend;

static void cuda_backend_free_bufs(CudaBackend *b) {
    for (int i = 0; i < b->n_bufs; i++) {
        if (b->d_buf[i]) cudaFree(b->d_buf[i]);
        b->d_buf[i] = NULL;
        b->buf_sizes[i] = 0;
    }
    b->n_bufs = 0;
}

static float *cuda_backend_get_buf(CudaBackend *b, int idx, size_t needed) {
    if (idx >= 8) { fprintf(stderr, "[cuda] buf index %d out of range\n", idx); return NULL; }
    while (b->n_bufs <= idx) {
        b->d_buf[b->n_bufs] = NULL;
        b->buf_sizes[b->n_bufs] = 0;
        b->n_bufs++;
    }
    if (b->buf_sizes[idx] < (int)needed) {
        if (b->d_buf[idx]) cudaFree(b->d_buf[idx]);
        cudaError_t e = cudaMalloc(&b->d_buf[idx], needed);
        if (e != cudaSuccess) return NULL;
        b->buf_sizes[idx] = (int)needed;
    }
    return b->d_buf[idx];
}

/* ================================================================ */
/*  Model weight upload (called once per model)                      */
/* ================================================================ */

static int cuda_upload_weights(CudaBackend *b, DACModel *model) {
    if (!model || !model->tensors) return TSAC_ERR_PARAM;
    DACTensor *ts = model->tensors;
    int nt = model->n_tensors;

    /* Upload dequantized conv weights */
    const char *weight_names[] = {
        "decoder.model.0",
        "decoder.model.1.block.1",
        "decoder.model.2.block.1",
        "decoder.model.3.block.1",
        "decoder.model.4.block.1",
        "decoder.model.6",
        /* Inner blocks (model.1-4, inner 2-4) */
        "decoder.model.1.block.2.block.1",
        "decoder.model.1.block.3.block.1",
        "decoder.model.1.block.4.block.1",
        "decoder.model.2.block.2.block.1",
        "decoder.model.2.block.3.block.1",
        "decoder.model.2.block.4.block.1",
        "decoder.model.3.block.2.block.1",
        "decoder.model.3.block.3.block.1",
        "decoder.model.3.block.4.block.1",
        "decoder.model.4.block.2.block.1",
        "decoder.model.4.block.3.block.1",
        "decoder.model.4.block.4.block.1",
    };
    int n_wl = sizeof(weight_names) / sizeof(weight_names[0]);

    b->n_gpu_weights = 0;
    for (int i = 0; i < n_wl && b->n_gpu_weights < MAX_GPU_WEIGHTS; i++) {
        char wv[160], wg[160], bi[160];
        snprintf(wv, sizeof(wv), "%s.weight_v", weight_names[i]);
        snprintf(wg, sizeof(wg), "%s.weight_g", weight_names[i]);
        snprintf(bi, sizeof(bi), "%s.bias",      weight_names[i]);

        DACTensor *twv = tf(ts, nt, wv);
        DACTensor *twg = tf(ts, nt, wg);
        DACTensor *tbi = tf(ts, nt, bi);

        if (!twv) continue;

        int Ci, K, Co, is_convt;
        float *w_f32 = dequant_weights(twv, twg, tbi, &Ci, &K, &Co, &is_convt);
        if (!w_f32) continue;

        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
        snprintf(gw->name, sizeof(gw->name), "%s", weight_names[i]);
        gw->Ci = Ci;
        gw->K  = K;
        gw->Co = Co;
        gw->is_convt = is_convt;

        size_t w_bytes = (size_t)Ci * K * Co * sizeof(float);
        CUDA_CHK(cudaMalloc(&gw->d_data, w_bytes));
        CUDA_CHK(cudaMemcpy(gw->d_data, w_f32, w_bytes, cudaMemcpyHostToDevice));
        free(w_f32);

        /* BUG FIX 2: Upload bias tensor separately (dequant_weights doesn't return bias) */
        DACTensor *bias_t = tf(ts, nt, bi);
        if (bias_t && b->n_gpu_weights < MAX_GPU_WEIGHTS) {
            GpuWeight *gbias = &b->gpu_weights[b->n_gpu_weights++];
            snprintf(gbias->name, sizeof(gbias->name), "%s.bias", weight_names[i]);
            gbias->Ci = bias_t->dims[0];
            gbias->K = 1;
            gbias->Co = 1;
            gbias->is_convt = 0;
            size_t bias_bytes = (size_t)bias_t->dims[0] * sizeof(float);
            CUDA_CHK(cudaMalloc(&gbias->d_data, bias_bytes));
            CUDA_CHK(cudaMemcpy(gbias->d_data, bias_t->data, bias_bytes, cudaMemcpyHostToDevice));
        }
    }

    /* Upload snake alpha tensors (float32, no dequant) */
    const char *snake_names[] = {
        "decoder.model.1.block.0.alpha",
        "decoder.model.2.block.0.alpha",
        "decoder.model.3.block.0.alpha",
        "decoder.model.4.block.0.alpha",
        "decoder.model.5.alpha",
        "decoder.model.1.block.2.block.0.alpha",
        "decoder.model.1.block.3.block.0.alpha",
        "decoder.model.1.block.4.block.0.alpha",
        "decoder.model.2.block.2.block.0.alpha",
        "decoder.model.2.block.3.block.0.alpha",
        "decoder.model.2.block.4.block.0.alpha",
        "decoder.model.3.block.2.block.0.alpha",
        "decoder.model.3.block.3.block.0.alpha",
        "decoder.model.3.block.4.block.0.alpha",
        "decoder.model.4.block.2.block.0.alpha",
        "decoder.model.4.block.3.block.0.alpha",
        "decoder.model.4.block.4.block.0.alpha",
    };
    for (int i = 0; i < (int)(sizeof(snake_names)/sizeof(snake_names[0])) && b->n_gpu_weights < MAX_GPU_WEIGHTS; i++) {
        DACTensor *ta = tf(ts, nt, snake_names[i]);
        if (!ta) continue;
        size_t n_bytes = (size_t)ta->dims[0] * sizeof(float);
        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
        snprintf(gw->name, sizeof(gw->name), "%s", snake_names[i]);
        gw->Ci = ta->dims[0]; gw->K = 1; gw->Co = 1; gw->is_convt = 0;
        CUDA_CHK(cudaMalloc(&gw->d_data, n_bytes));
        CUDA_CHK(cudaMemcpy(gw->d_data, ta->data, n_bytes, cudaMemcpyHostToDevice));
    }

    /* Upload encoder weights with "enc:" prefix.
     * NOTE: model uses "encoder.block.X" naming (confirmed via strings). */
    const char *enc_weight_names[] = {
        "encoder.block.0",
        "encoder.block.6",
        /* Strided convs (block.4 = strided, K=4/8/16/16) */
        "encoder.block.1.block.4",
        "encoder.block.2.block.4",
        "encoder.block.3.block.4",
        "encoder.block.4.block.4",
        /* Inner blocks */
        "encoder.block.1.block.0.block.1",
        "encoder.block.1.block.1.block.1",
        "encoder.block.1.block.2.block.1",
        "encoder.block.2.block.0.block.1",
        "encoder.block.2.block.1.block.1",
        "encoder.block.2.block.2.block.1",
        "encoder.block.3.block.0.block.1",
        "encoder.block.3.block.1.block.1",
        "encoder.block.3.block.2.block.1",
        "encoder.block.4.block.0.block.1",
        "encoder.block.4.block.1.block.1",
        "encoder.block.4.block.2.block.1",
    };
    int n_enc_wl = sizeof(enc_weight_names) / sizeof(enc_weight_names[0]);
    for (int i = 0; i < n_enc_wl && b->n_gpu_weights < MAX_GPU_WEIGHTS; i++) {
        char wv[160], wg[160], bi[160];
        snprintf(wv, sizeof(wv), "%s.weight_v", enc_weight_names[i]);
        snprintf(wg, sizeof(wg), "%s.weight_g", enc_weight_names[i]);
        snprintf(bi, sizeof(bi), "%s.bias",      enc_weight_names[i]);

        DACTensor *twv = tf(ts, nt, wv);
        DACTensor *twg = tf(ts, nt, wg);
        DACTensor *tbi = tf(ts, nt, bi);

        if (!twv) continue;

        int Ci, K, Co, is_convt;
        float *w_f32 = dequant_weights(twv, twg, tbi, &Ci, &K, &Co, &is_convt);
        if (!w_f32) continue;

        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
        snprintf(gw->name, sizeof(gw->name), "enc:%s", enc_weight_names[i]);
        gw->Ci = Ci;
        gw->K  = K;
        gw->Co = Co;
        gw->is_convt = is_convt;

        size_t w_bytes = (size_t)Ci * K * Co * sizeof(float);
        CUDA_CHK(cudaMalloc(&gw->d_data, w_bytes));
        CUDA_CHK(cudaMemcpy(gw->d_data, w_f32, w_bytes, cudaMemcpyHostToDevice));
        free(w_f32);

        /* Upload bias separately */
        if (tbi && b->n_gpu_weights < MAX_GPU_WEIGHTS) {
            GpuWeight *gbias = &b->gpu_weights[b->n_gpu_weights++];
            snprintf(gbias->name, sizeof(gbias->name), "enc:%s.bias", enc_weight_names[i]);
            gbias->Ci = tbi->dims[0];
            gbias->K = 1;
            gbias->Co = 1;
            gbias->is_convt = 0;
            size_t bias_bytes = (size_t)tbi->dims[0] * sizeof(float);
            CUDA_CHK(cudaMalloc(&gbias->d_data, bias_bytes));
            CUDA_CHK(cudaMemcpy(gbias->d_data, tbi->data, bias_bytes, cudaMemcpyHostToDevice));
        }
    }

    /* Upload encoder snake alphas */
    const char *enc_snake_names[] = {
        "encoder.block.5.alpha",
        "encoder.block.1.block.0.alpha",
        "encoder.block.1.block.2.block.0.alpha",
        "encoder.block.1.block.3.block.0.alpha",
        "encoder.block.1.block.4.block.0.alpha",
        "encoder.block.2.block.0.alpha",
        "encoder.block.2.block.2.block.0.alpha",
        "encoder.block.2.block.3.block.0.alpha",
        "encoder.block.2.block.4.block.0.alpha",
        "encoder.block.3.block.0.alpha",
        "encoder.block.3.block.2.block.0.alpha",
        "encoder.block.3.block.3.block.0.alpha",
        "encoder.block.3.block.4.block.0.alpha",
        "encoder.block.4.block.0.alpha",
        "encoder.block.4.block.2.block.0.alpha",
        "encoder.block.4.block.3.block.0.alpha",
        "encoder.block.4.block.4.block.0.alpha",
    };
    for (int i = 0; i < (int)(sizeof(enc_snake_names)/sizeof(enc_snake_names[0])) && b->n_gpu_weights < MAX_GPU_WEIGHTS; i++) {
        DACTensor *ta = tf(ts, nt, enc_snake_names[i]);
        if (!ta) continue;
        size_t n_bytes = (size_t)ta->dims[0] * sizeof(float);
        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
        snprintf(gw->name, sizeof(gw->name), "enc:%s", enc_snake_names[i]);
        gw->Ci = ta->dims[0]; gw->K = 1; gw->Co = 1; gw->is_convt = 0;
        CUDA_CHK(cudaMalloc(&gw->d_data, n_bytes));
        CUDA_CHK(cudaMemcpy(gw->d_data, ta->data, n_bytes, cudaMemcpyHostToDevice));
    }

    /* Upload all codebooks (quantizer.quantizers.0-11.codebook.weight) */
    {
        int total_entries = 0;
        for (int cb = 0; cb < 12; cb++) {
            char cb_name[160];
            snprintf(cb_name, sizeof(cb_name), "quantizer.quantizers.%d.codebook.weight", cb);
            DACTensor *cb_t = tf(ts, nt, cb_name);
            if (!cb_t) break;
            b->cb_offsets[cb] = total_entries;
            total_entries += cb_t->dims[0] * cb_t->dims[1];
        }
        b->n_cb = 12;
        b->cb_dim = 1024;
        b->cb_entries = 8;

        size_t cb_total = (size_t)total_entries * sizeof(float);
        CUDA_CHK(cudaMalloc(&b->d_cb_data, cb_total));
        CUDA_CHK(cudaMalloc(&b->d_cb_offsets, 13 * sizeof(int)));

        size_t copied = 0;
        for (int cb = 0; cb < 12; cb++) {
            char cb_name[160];
            snprintf(cb_name, sizeof(cb_name), "quantizer.quantizers.%d.codebook.weight", cb);
            DACTensor *cb_t = tf(ts, nt, cb_name);
            if (!cb_t) { b->cb_offsets[cb] = (int)copied / (int)sizeof(float); continue; }
            size_t sz = (size_t)cb_t->dims[0] * cb_t->dims[1] * sizeof(float);
            CUDA_CHK(cudaMemcpy((char *)b->d_cb_data + copied, cb_t->data, sz, cudaMemcpyHostToDevice));
            b->cb_offsets[cb] = (int)(copied / sizeof(float));
            copied += sz;
        }
        b->cb_offsets[12] = (int)(copied / sizeof(float));
        CUDA_CHK(cudaMemcpy(b->d_cb_offsets, b->cb_offsets, 13 * sizeof(int), cudaMemcpyHostToDevice));
    }

    b->weights_uploaded = 1;
    fprintf(stderr, "[cuda] uploaded %d decoder weights + %d enc weights + codebooks\n",
            n_wl, n_enc_wl);

    return TSAC_OK;
}

/* ================================================================ */
/*  Weight lookup helper                                              */
/* ================================================================ */

static GpuWeight *cuda_find_weight(CudaBackend *b, const char *name) {
    for (int i = 0; i < b->n_gpu_weights; i++)
        if (!strcmp(b->gpu_weights[i].name, name))
            return &b->gpu_weights[i];
    return NULL;
}

static GpuWeight *cuda_find_enc_weight(CudaBackend *b, const char *name) {
    char fullname[256];
    snprintf(fullname, sizeof(fullname), "enc:%s", name);
    return cuda_find_weight(b, fullname);
}

/* ================================================================ */
/*  Public API                                                        */
/* ================================================================ */

extern "C" int tsac_cuda_init(void **priv) {
    if (!priv) return TSAC_ERR_PARAM;

    CudaBackend *b = (CudaBackend *)calloc(1, sizeof(CudaBackend));
    if (!b) return TSAC_ERR_MEMORY;

    int count;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count < 1) { free(b); return TSAC_ERR_BACKEND; }

    CUDA_CHK(cudaSetDevice(0));
    CUDA_CHK(cudaStreamCreate(&b->stream));

    b->initialized = 1;
    b->weights_uploaded = 0;
    *priv = b;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "[cuda] init OK: %s (%d SMs, CC %d.%d)\n",
            prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    return TSAC_OK;
}

extern "C" int tsac_cuda_decode(void *priv, void *model_ptr,
                                 const int *codebook_indices, int n_frames,
                                 int n_codebooks, int block_len, int channels,
                                 float *pcm, int n_samples) {
    (void)block_len;
    CudaBackend *b = (CudaBackend *)priv;
    DACModel *model = (DACModel *)model_ptr;

    if (!b || !b->initialized || !model) return TSAC_ERR_PARAM;

    cudaStream_t s = b->stream;

    /* Lazy weight upload on first call */
    if (!b->weights_uploaded) {
        int ret = cuda_upload_weights(b, model);
        if (ret != TSAC_OK) return ret;
        /* Weight sanity check: verify decoder.model.0 weights are valid */
        GpuWeight *w0 = cuda_find_weight(b, "decoder.model.0");
        if (w0) {
            float test[4];
            cudaMemcpy(test, w0->d_data, 4*sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[cuda] w0[0..3]=%f %f %f %f\n", test[0], test[1], test[2], test[3]);
        } else {
            fprintf(stderr, "[cuda] WARNING: decoder.model.0 weight not found!\n");
        }
    }

    if (n_frames < 1) return TSAC_OK;

    int rvq_dim = 1024;

    /* Upload codebook indices */
    size_t codes_bytes = (size_t)n_frames * n_codebooks * sizeof(int);
    CUDA_CHK(cudaMalloc(&b->d_codes, codes_bytes));
    CUDA_CHK(cudaMemcpy(b->d_codes, codebook_indices, codes_bytes, cudaMemcpyHostToDevice));

    /* RVQ lookup → features [1024, n_frames] */
    size_t feat_bytes = (size_t)rvq_dim * n_frames * sizeof(float);
    float *d_feat = cuda_backend_get_buf(b, 0, feat_bytes);
    if (!d_feat) return TSAC_ERR_MEMORY;
    CUDA_CHK(cudaMemsetAsync(d_feat, 0, feat_bytes, s));
    CUDA_CHK(launch_rvq_lookup(d_feat, b->d_codes, b->d_cb_data, b->d_cb_offsets,
                                n_frames, n_codebooks, rvq_dim, s));
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaStreamSynchronize(s));

    /* Decoder graph — identical structure to CPU decoder */
    /* model.0: Conv1d(1024→1536, K=7) */
    GpuWeight *w0 = cuda_find_weight(b, "decoder.model.0");
    int T0 = n_frames;
    int C0 = w0 ? w0->Co : 1536;
    size_t b0_bytes = (size_t)C0 * T0 * sizeof(float);
    float *d_buf0 = cuda_backend_get_buf(b, 1, b0_bytes);
    CUDA_CHK(cudaMemsetAsync(d_buf0, 0, b0_bytes, s));
    if (w0) {
        CUDA_CHK(launch_conv1d(d_buf0, d_feat, w0->d_data, NULL, T0, w0->K, w0->Ci, w0->Co, s));
        CUDA_CHK(cudaGetLastError());
        /* BUG FIX 2: Add bias after model.0 conv1d */
        GpuWeight *b0 = cuda_find_weight(b, "decoder.model.0.bias");
        if (b0) {
            CUDA_CHK(launch_add_bias(d_buf0, b0->d_data, T0, C0, s));
            CUDA_CHK(cudaGetLastError());
        } else {
            fprintf(stderr, "[cuda] WARNING: decoder.model.0.bias not found!\n");
        }
    }
    CUDA_CHK(cudaStreamSynchronize(s));

    float *d_cur = d_buf0;
    int cur_C = C0;
    int cur_T = T0;

    /* Blocks 1-4 */
    int c_out[4] = {768, 384, 192, 96};
    for (int blk = 1; blk <= 4; blk++) {
        char wname[160], sname[160];
        snprintf(wname, sizeof(wname), "decoder.model.%d.block.1", blk);
        snprintf(sname, sizeof(sname), "decoder.model.%d.block.0.alpha", blk);

        /* Snake before conv */
        GpuWeight *gs = cuda_find_weight(b, sname);
        if (gs)
            CUDA_CHK(launch_snake(d_cur, gs->d_data, cur_C * cur_T, gs->Ci, s));

        /* ConvTranspose1d — 2x upsampling */
        GpuWeight *gw = cuda_find_weight(b, wname);
        int next_C = c_out[blk - 1];
        int next_T = cur_T * 2;

        size_t nb = (size_t)next_C * next_T * sizeof(float);
        float *d_next = cuda_backend_get_buf(b, blk + 1, nb);
        CUDA_CHK(cudaMemsetAsync(d_next, 0, nb, s));

        if (gw) {
            if (gw->is_convt)
                CUDA_CHK(launch_conv_transpose1d(d_next, d_cur, gw->d_data,
                                                  cur_T, next_T, gw->K, gw->Ci, gw->Co, s));
            else
                CUDA_CHK(launch_conv1d(d_next, d_cur, gw->d_data, NULL,
                                        cur_T, gw->K, gw->Ci, gw->Co, s));
            CUDA_CHK(cudaGetLastError());
            /* BUG FIX 2: Add bias after conv_transpose1d/conv1d */
            char bname[160];
            snprintf(bname, sizeof(bname), "decoder.model.%d.block.1.bias", blk);
            GpuWeight *gbias = cuda_find_weight(b, bname);
            if (gbias) {
                CUDA_CHK(launch_add_bias(d_next, gbias->d_data, next_T, next_C, s));
                CUDA_CHK(cudaGetLastError());
            }
        }

        /* Debug check: verify conv_transpose1d output is finite after first block */
        if (blk == 1) {
            CUDA_CHK(cudaStreamSynchronize(s));
            float *check = (float*)malloc(next_C * next_T * sizeof(float));
            if (check) {
                cudaMemcpy(check, d_next, next_C * next_T * sizeof(float), cudaMemcpyDeviceToHost);
                int nan_count = 0;
                for (int i = 0; i < next_C * next_T; i++) if (check[i] != check[i]) nan_count++;
                fprintf(stderr, "[cuda] block1 convt_out: %d/%d NaN\n", nan_count, next_C * next_T);
                free(check);
            }
        }

        d_cur = d_next;
        cur_C = next_C;
        cur_T = next_T;

        /* Inner blocks (3× per residual block) */
        for (int inner = 2; inner <= 4; inner++) {
            char iname[160], isname[160], bname[160];
            snprintf(iname, sizeof(iname), "decoder.model.%d.block.%d.block.1", blk, inner);
            snprintf(isname, sizeof(isname), "decoder.model.%d.block.%d.block.0.alpha", blk, inner);
            snprintf(bname, sizeof(bname), "decoder.model.%d.block.%d.block.1.bias", blk, inner);

            GpuWeight *gis = cuda_find_weight(b, isname);
            if (gis)
                CUDA_CHK(launch_snake(d_cur, gis->d_data, cur_C * cur_T, gis->Ci, s));

            /* BUG FIX 1: Use temp buffer for conv1d to avoid in-place race condition */
            GpuWeight *giw = cuda_find_weight(b, iname);
            GpuWeight *gbias = cuda_find_weight(b, bname);
            if (giw) {
                float *d_tmp = cuda_backend_get_buf(b, 6, (size_t)cur_C * cur_T * sizeof(float));
                CUDA_CHK(cudaMemsetAsync(d_tmp, 0, (size_t)cur_C * cur_T * sizeof(float), s));
                CUDA_CHK(launch_conv1d(d_tmp, d_cur, giw->d_data, NULL,
                                        cur_T, giw->K, giw->Ci, giw->Co, s));
                CUDA_CHK(cudaGetLastError());
                /* BUG FIX 2: Add bias after conv1d */
                if (gbias) {
                    CUDA_CHK(launch_add_bias(d_tmp, gbias->d_data, cur_T, giw->Co, s));
                    CUDA_CHK(cudaGetLastError());
                }
                d_cur = d_tmp;
            }
        }
        CUDA_CHK(cudaStreamSynchronize(s));
    }

    /* model.5: Snake(96) */
    GpuWeight *gm5 = cuda_find_weight(b, "decoder.model.5.alpha");
    if (gm5) {
        CUDA_CHK(launch_snake(d_cur, gm5->d_data, cur_C * cur_T, gm5->Ci, s));
        CUDA_CHK(cudaGetLastError());
    }
    CUDA_CHK(cudaStreamSynchronize(s));

    /* model.6: Conv1d(96→2, K=7) */
    GpuWeight *gm6 = cuda_find_weight(b, "decoder.model.6");
    if (gm6) {
        int out_C = gm6->Co;
        int out_T = cur_T;
        size_t ob = (size_t)out_C * out_T * sizeof(float);
        float *d_out = cuda_backend_get_buf(b, 7, ob);
        CUDA_CHK(cudaMemsetAsync(d_out, 0, ob, s));
        CUDA_CHK(launch_conv1d(d_out, d_cur, gm6->d_data, NULL,
                                out_T, gm6->K, gm6->Ci, gm6->Co, s));
        CUDA_CHK(cudaGetLastError());
        /* BUG FIX 2: Add bias after model.6 conv1d */
        GpuWeight *b6 = cuda_find_weight(b, "decoder.model.6.bias");
        if (b6) {
            CUDA_CHK(launch_add_bias(d_out, b6->d_data, out_T, out_C, s));
            CUDA_CHK(cudaGetLastError());
        }
        CUDA_CHK(launch_tanh_clip(d_out, out_C * out_T, s));
        CUDA_CHK(cudaGetLastError());

        /* Copy to host PCM */
        int to_copy = out_T;
        if (to_copy > (int)n_samples) to_copy = n_samples;
        if (to_copy > 0)
            CUDA_CHK(cudaMemcpyAsync(pcm, d_out, (size_t)to_copy * sizeof(float),
                                      cudaMemcpyDeviceToHost, s));
    }

    CUDA_CHK(cudaStreamSynchronize(s));

    return TSAC_OK;
}

extern "C" int tsac_cuda_encode(void *priv, void *model_ptr,
                                  const float *pcm, int n_samples, int channels,
                                  int n_codebooks, int block_len,
                                  int **codebook_indices, int *n_frames) {
    (void)block_len;
    CudaBackend *b = (CudaBackend *)priv;
    DACModel *model = (DACModel *)model_ptr;

    if (!b || !b->initialized || !model) return TSAC_ERR_PARAM;

    cudaStream_t s = b->stream;

    /* Lazy weight upload on first call */
    if (!b->weights_uploaded) {
        int ret = cuda_upload_weights(b, model);
        if (ret != TSAC_OK) return ret;
    }

    if (n_samples < 1) return TSAC_OK;

    /* Calculate frame count */
    int nf = (n_samples + block_len - 1) / block_len;
    if (nf < 1) nf = 1;
    *n_frames = nf;

    /* Upload PCM to GPU */
    size_t pcm_bytes = (size_t)n_samples * channels * sizeof(float);
    float *d_pcm;
    CUDA_CHK(cudaMalloc(&d_pcm, pcm_bytes));
    CUDA_CHK(cudaMemcpy(d_pcm, pcm, pcm_bytes, cudaMemcpyHostToDevice));

    /* encoder.block.6: Conv1d(channels→96, K=7) */
    GpuWeight *e6 = cuda_find_enc_weight(b, "enc:encoder.block.6");
    float *d_cur = cuda_backend_get_buf(b, 0, 96 * nf * sizeof(float));
    if (!d_cur) { cudaFree(d_pcm); return TSAC_ERR_MEMORY; }
    CUDA_CHK(cudaMemsetAsync(d_cur, 0, 96 * nf * sizeof(float), s));
    if (e6) {
        CUDA_CHK(launch_conv1d(d_cur, d_pcm, e6->d_data, NULL, nf, e6->K, e6->Ci, e6->Co, s));
        GpuWeight *b6 = cuda_find_enc_weight(b, "enc:encoder.block.6.bias");
        if (b6) CUDA_CHK(launch_add_bias(d_cur, b6->d_data, nf, 96, s));
    }
    cudaFree(d_pcm);

    /* encoder.block.5: Snake(96) */
    GpuWeight *e5 = cuda_find_enc_weight(b, "enc:encoder.block.5.alpha");
    if (e5) CUDA_CHK(launch_snake(d_cur, e5->d_data, 96 * nf, 96, s));

    /* Blocks 4→3→2→1 (reverse order: 96→192→384→768→1536) */
    int cur_C = 96;
    int cur_T = nf;

    for (int blk = 4; blk >= 1; blk--) {
        /* Snake before strided conv */
        char sname[128];
        snprintf(sname, sizeof(sname), "enc:encoder.block.%d.block.0.alpha", blk);
        GpuWeight *sa = cuda_find_enc_weight(b, sname);
        if (sa) CUDA_CHK(launch_snake(d_cur, sa->d_data, cur_C * cur_T, sa->Ci, s));

        /* Strided conv: block.4.weight_v with K=4/8/16/16, stride=K/2 */
        char wname[128];
        snprintf(wname, sizeof(wname), "enc:encoder.block.%d.block.4", blk);
        GpuWeight *cw = cuda_find_enc_weight(b, wname);
        if (!cw) continue;
        int next_C = cw->Co;
        int stride = cw->K / 2;
        int next_T = (cur_T + stride - 1) / stride;

        float *d_next = cuda_backend_get_buf(b, blk, next_C * next_T * sizeof(float));
        if (!d_next) return TSAC_ERR_MEMORY;
        CUDA_CHK(cudaMemsetAsync(d_next, 0, next_C * next_T * sizeof(float), s));
        CUDA_CHK(launch_conv1d_strided(d_next, d_cur, cw->d_data, NULL,
                                        next_T, cw->K, cw->Ci, cw->Co, cur_T, stride, s));
        char bname[128];
        snprintf(bname, sizeof(bname), "enc:encoder.block.%d.block.4.bias", blk);
        GpuWeight *cb = cuda_find_enc_weight(b, bname);
        if (cb) CUDA_CHK(launch_add_bias(d_next, cb->d_data, next_T, next_C, s));

        d_cur = d_next;
        cur_C = next_C;
        cur_T = next_T;

        /* Inner residual units (snake→K=7 conv → snake → K=1 conv) × 3 */
        for (int inner = 0; inner <= 2; inner++) {
            char isname[160];
            snprintf(isname, sizeof(isname), "enc:encoder.block.%d.block.%d.block.0.alpha", blk, inner);
            GpuWeight *gis = cuda_find_enc_weight(b, isname);
            if (gis) CUDA_CHK(launch_snake(d_cur, gis->d_data, cur_C * cur_T, gis->Ci, s));

            char iwname[160], ibname[160];
            snprintf(iwname, sizeof(iwname), "enc:encoder.block.%d.block.%d.block.1", blk, inner);
            snprintf(ibname, sizeof(ibname), "enc:encoder.block.%d.block.%d.block.1.bias", blk, inner);
            GpuWeight *giw = cuda_find_enc_weight(b, iwname);
            GpuWeight *gib = cuda_find_enc_weight(b, ibname);
            if (giw) {
                float *d_tmp = cuda_backend_get_buf(b, 6 + inner, cur_C * cur_T * sizeof(float));
                if (!d_tmp) return TSAC_ERR_MEMORY;
                CUDA_CHK(cudaMemsetAsync(d_tmp, 0, cur_C * cur_T * sizeof(float), s));
                CUDA_CHK(launch_conv1d(d_tmp, d_cur, giw->d_data, NULL,
                                        cur_T, giw->K, giw->Ci, giw->Co, s));
                if (gib) CUDA_CHK(launch_add_bias(d_tmp, gib->d_data, cur_T, cur_C, s));
                d_cur = d_tmp;
            }
        }
    }

    /* encoder.block.0: Conv1d(1536→1024, K=7) */
    GpuWeight *e0 = cuda_find_enc_weight(b, "enc:encoder.block.0");
    float *d_features = cuda_backend_get_buf(b, 5, 1024 * cur_T * sizeof(float));
    if (!d_features) return TSAC_ERR_MEMORY;
    CUDA_CHK(cudaMemsetAsync(d_features, 0, 1024 * cur_T * sizeof(float), s));
    if (e0) {
        CUDA_CHK(launch_conv1d(d_features, d_cur, e0->d_data, NULL,
                                cur_T, e0->K, e0->Ci, e0->Co, s));
        GpuWeight *b0 = cuda_find_enc_weight(b, "enc:encoder.block.0.bias");
        if (b0) CUDA_CHK(launch_add_bias(d_features, b0->d_data, cur_T, 1024, s));
    }

    CUDA_CHK(cudaStreamSynchronize(s));

    /* RVQ quantization */
    int *h_indices = (int*)malloc(cur_T * n_codebooks * sizeof(int));
    if (!h_indices) return TSAC_ERR_MEMORY;

    float *d_residual;
    CUDA_CHK(cudaMalloc(&d_residual, 1024 * cur_T * sizeof(float)));
    CUDA_CHK(cudaMemcpy(d_residual, d_features, 1024 * cur_T * sizeof(float), cudaMemcpyDeviceToDevice));

    int *d_indices;
    CUDA_CHK(cudaMalloc(&d_indices, cur_T * sizeof(int)));

    for (int cb = 0; cb < n_codebooks; cb++) {
        /* Quantize against codebook cb */
        CUDA_CHK(launch_rvq_quantize(d_indices, d_residual,
                                      b->d_cb_data + b->cb_offsets[cb],
                                      cur_T, 1024, b->cb_entries, s));
        CUDA_CHK(cudaGetLastError());

        /* Copy indices to host */
        CUDA_CHK(cudaMemcpy(h_indices + cb * cur_T, d_indices,
                            cur_T * sizeof(int), cudaMemcpyDeviceToHost));

        /* Subtract codebook entry */
        CUDA_CHK(launch_rvq_subtract(d_residual, b->d_cb_data + b->cb_offsets[cb],
                                      d_indices, cur_T, 1024, s));
        CUDA_CHK(cudaGetLastError());
    }

    cudaFree(d_residual);
    cudaFree(d_indices);

    *codebook_indices = h_indices;
    *n_frames = cur_T;

    CUDA_CHK(cudaStreamSynchronize(s));

    return TSAC_OK;
}

extern "C" void tsac_cuda_shutdown(void *priv) {
    CudaBackend *b = (CudaBackend *)priv;
    if (!b) return;

    for (int i = 0; i < b->n_gpu_weights; i++)
        if (b->gpu_weights[i].d_data) cudaFree(b->gpu_weights[i].d_data);

    if (b->d_cb_data) cudaFree(b->d_cb_data);
    if (b->d_cb_offsets) cudaFree(b->d_cb_offsets);
    if (b->d_codes) cudaFree(b->d_codes);

    cuda_backend_free_bufs(b);

    if (b->stream) cudaStreamDestroy(b->stream);

    free(b);
}
