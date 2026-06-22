        if (!vp) return NULL;
        memcpy(vp, "weight_g", 8);
        DACTensor *gt = F(ts, nt, gn);
        if (!gt || gt->elem_size != 4) return NULL;

        /* Weight dims: [Ci, K, Co] or [Co, K, Ci] - figure out which */
        int d0 = t->dims[0];
        int d1 = t->dims[1];
        int d2 = t->dims[2];

        /* Use bias size for Co */
        int Co = bias_t ? bias_t->dims[0] : d2;
        int K = d1;
        int Ci = (d0 == Co) ? d2 : d0;

        /* Determine if scales are per-input-channel or per-output-channel */
        int per_input = (Ci == (int)gt->dims[2]);

        int nel_elements = Ci * K * Co;

        float *cpu_buf = (float*)malloc(nel_elements * sizeof(float));
        const float *g_scales = (const float*)gt->data;
        const uint8_t *v_data = (const uint8_t*)t->data;

        /* Dequantize with transposition to [Co, Ci, K] format */
        for (int co = 0; co < Co; co++) {
            for (int ci = 0; ci < Ci; ci++) {
                float g_val = per_input ? g_scales[ci] : g_scales[co];
                for (int k = 0; k < K; k++) {
                    int src_idx;
                    if (per_input) {
                        /* [Co, K, Ci] layout: src[co][k][ci] */
                        src_idx = co * K * Ci + k * Ci + ci;
                    } else {
                        /* [Ci, K, Co] layout: src[ci][k][co] */
                        src_idx = ci * K * Co + k * Co + co;
                    }
                    int8_t v_val = (int8_t)v_data[src_idx];
                    int dst_idx = co * Ci * K + ci * K + k;  /* [Co, Ci, K] layout */
                    cpu_buf[dst_idx] = g_val * ((float)v_val - 128.0f) / 127.0f;
                }
            }
        }
        hipMalloc(&t->dev_f32, nel_elements * sizeof(float));
        hipMemcpyAsync(t->dev_f32, cpu_buf, nel_elements * sizeof(float), hipMemcpyHostToDevice, st);
        free(cpu_buf);
    }
    return t->dev_f32;
}

/* ================================================================ */
/*  GPU weight cache (ported from CUDA backend)                      */
/* ================================================================ */

#define MAX_GPU_WEIGHTS 128

typedef struct {
    char     name[128];
    float   *d_data;
    int      Ci, K, Co;
    int      is_convt;
} GpuWeight;

typedef struct {
    int          initialized;
    int          weights_uploaded;
    GpuWeight    gpu_weights[MAX_GPU_WEIGHTS];
    int          n_gpu_weights;
    float       *d_cb_data;
    int         *d_cb_offsets;
    int          cb_offsets[13];
    int          n_cb;
    int          cb_dim;
    int          cb_entries;
    float       *d_buf[8];
    int          buf_sizes[8];
    int          n_bufs;
    int         *d_codes;
    float       *d_features;
    hipStream_t  stream;
} HipBackend;

/* Release all GPU buffers held by the HIP backend. */
static void hip_backend_free_bufs(HipBackend *b) {
    for (int i = 0; i < b->n_bufs; i++) {
        if (b->d_buf[i]) hipFree(b->d_buf[i]);
        b->d_buf[i] = NULL;
        b->buf_sizes[i] = 0;
    }
    b->n_bufs = 0;
}

/* Get or allocate a GPU buffer of the requested size. */
static float *hip_backend_get_buf(HipBackend *b, int idx, size_t needed) {
    if (idx >= 8) { fprintf(stderr, "[hip] buf index %d out of range\n", idx); return NULL; }
    while (b->n_bufs <= idx) {
        b->d_buf[b->n_bufs] = NULL;
        b->buf_sizes[b->n_bufs] = 0;
        b->n_bufs++;
    }
    if (b->buf_sizes[idx] < (int)needed) {
        if (b->d_buf[idx]) hipFree(b->d_buf[idx]);
        hipError_t e = hipMalloc(&b->d_buf[idx], needed);
        if (e != hipSuccess) return NULL;
        b->buf_sizes[idx] = (int)needed;
    }
    return b->d_buf[idx];
}

/* Find a decoder weight tensor by name in the GPU weight cache. */
static GpuWeight *hip_find_weight(HipBackend *b, const char *name) {
    for (int i = 0; i < b->n_gpu_weights; i++)
        if (!strcmp(b->gpu_weights[i].name, name))
            return &b->gpu_weights[i];
    return NULL;
}

/* Find an encoder weight tensor by name in the GPU weight cache. */
#include "hip_dequant.inc"
            gbias->is_convt = 0;
            size_t bias_bytes = (size_t)tbi->dims[0] * sizeof(float);
            hipMalloc(&gbias->d_data, bias_bytes);
            hipMemcpy(gbias->d_data, tbi->data, bias_bytes, hipMemcpyHostToDevice);
        }
    }

    /* Upload snake alpha tensors */
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
        DACTensor *ta = F(ts, nt, snake_names[i]);
        if (!ta) continue;
        size_t n_bytes = (size_t)ta->dims[0] * sizeof(float);
        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
