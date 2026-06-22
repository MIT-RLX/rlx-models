/* dac_decoder.hip.cpp — HIP/ROCm GPU backend for tsac-ng. */
#include <hip/hip_runtime.h>
#include "../src/dac_model.h"
#include "../include/tsac.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BLK 256

__global__ void add_k(float *o, const float *a, const float *b, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}

__global__ void add_bias_k(float *x, const float *b, int T, int C) {
    int c = blockIdx.x, t = blockIdx.y * BLK + threadIdx.y;
    if (c >= C || t >= T) return;
    x[c * T + t] += b[c];
}

__global__ void mul_k(float *o, const float *a, const float *b, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < n) o[i] = a[i] * b[i];
}

__global__ void i8tof32_k(float *o, const int8_t *x, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < n) o[i] = (float)x[i] / 127.0f;
}

__global__ void snake_k(float *o, const float *x, const float *a,
                         int n, int C) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float al = a[i % C];
    o[i] = v + __sinf(al * v) * __sinf(al * v) / fmaxf(al, 1e-6f);
}

__global__ void conv1d_k(float *o, const float *x, const float *w,
    const float *b, int T, int K, int Ci, int Co, int S) {
    int oc = blockIdx.x, oi = blockIdx.y * BLK + threadIdx.y;
    if (oc >= Co || oi >= T) return;
    float s = b ? b[oc] : 0;
    int P = K/2;
    for (int ic = 0; ic < Ci; ic++)
        for (int j = 0; j < K; j++) {
            int ii = oi * S + j - P;
            if (ii >= 0 && ii < T)
                s += x[ic*T + ii] * w[oc*Ci*K + ic*K + j];
        }
    o[oc*T + oi] = s;
}

__global__ void convt_k(float *o, const float *x, const float *w,
    int T_in, int T_out, int K, int Ci, int Co, int S) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Co * T_out;
    if (tid >= total) return;

    int oi = tid % T_out;
    int oc = tid / T_out;
    int P = K / 2;

    float sum = 0.0f;
    for (int ic = 0; ic < Ci; ic++) {
        for (int j = 0; j < K; j++) {
            int tmp = oi + P - j;
            if (tmp >= 0 && tmp % S == 0) {
                int ii = tmp / S;
                if (ii < T_in)
                    sum += x[ic * T_in + ii] * w[oc * Ci * K + ic * K + j];
            }
        }
    }
    o[tid] = sum;
}

__global__ void gn_k(float *o, const float *x, const float *w,
    const float *b, int E, int G, float eps) {
    extern __shared__ float sh[];
    int t = threadIdx.x;
    float s = 0, sq = 0;
    for (int i = t; i < E; i += blockDim.x) {
        float v = x[blockIdx.x*E + i];
        s += v;
        sq += v*v;
    }
    sh[t] = s;
    sh[blockDim.x + t] = sq;
    __syncthreads();
    for (int r = blockDim.x/2; r > 0; r >>= 1) {
        if (t < r) { sh[t] += sh[t+r]; sh[blockDim.x+t] += sh[blockDim.x+t+r]; }
        __syncthreads();
    }
    float mn = sh[0]/E, vr = sh[blockDim.x]/E - mn*mn;
    float is = rsqrtf(fmaxf(vr + eps, 1e-10f));
    for (int i = t; i < E; i += blockDim.x) {
        int idx = blockIdx.x*E + i;
        o[idx] = (x[idx] - mn) * is * (w ? w[blockIdx.x] : 1) + (b ? b[blockIdx.x] : 0);
    }
}

__global__ void us2x_k(float *o, const float *x, int C, int T) {
    int c = blockIdx.x, t = threadIdx.x;
    if (c >= C || t >= T) return;
    o[c*T*2 + t*2] = o[c*T*2 + t*2+1] = x[c*T + t];
}

__global__ void rvq_lookup_k(float *features, const int *codes,
    const float **codebooks, int n_frames, int n_cb, int rvq_dim) {
    int f = blockIdx.x * BLK + threadIdx.x;
    if (f >= n_frames) return;
    
    for (int cb = 0; cb < n_cb; cb++) {
        int entry = codes[f * n_cb + cb];
        const float *cb_data = codebooks[cb];
        if (!cb_data) continue;
        
        for (int d = 0; d < rvq_dim; d++) {
            atomicAdd(&features[d * n_frames + f], cb_data[entry * rvq_dim + d]);
        }
    }
}

__global__ void rvq_lookup_simple_k(float *features, const int *codes,
    const float *cb_data, int cb_idx, int entries, int n_frames, int n_cb, int rvq_dim) {
    int f = blockIdx.x * BLK + threadIdx.x;
    if (f >= n_frames) return;
    
    int entry = codes[f * n_cb + cb_idx];
    if (entry < 0) entry = 0;
    if (entry >= entries) entry = entry % entries;
    if (entry < 0) entry = 0;
    
    for (int d = 0; d < rvq_dim; d++) {
        features[d * n_frames + f] += cb_data[entry * rvq_dim + d];
    }
}

__global__ void tanh_clip_k(float *x, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (v > 1.0f) v = tanhf(v);
    if (v < -1.0f) v = tanhf(v);
    x[i] = v;
}

/* RVQ quantization kernels for encoder */
__global__ void rvq_quantize_k(float *indices, const float *features,
    const float *codebook, int n_frames, int rvq_dim, int entries) {
    int f = blockIdx.x * BLK + threadIdx.x;
    if (f >= n_frames) return;
    float best = 1e30f; int best_e = 0;
    for (int e = 0; e < entries; e++) {
        float d = 0;
        for (int dim = 0; dim < rvq_dim; dim++) {
            float df = features[f * rvq_dim + dim] - codebook[e * rvq_dim + dim];
            d += df * df;
        }
        if (d < best) { best = d; best_e = e; }
    }
    indices[f] = (float)best_e;
}

__global__ void rvq_subtract_k(float *features, const float *codebook,
    const float *indices, int n_frames, int rvq_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_frames * rvq_dim) return;
    int f = tid / rvq_dim, d = tid % rvq_dim;
    features[tid] -= codebook[(int)indices[f] * rvq_dim + d];
}

/* ================================================================ */
/*  CPU-side tensor finder and weight upload                         */
/* ================================================================ */

/* Look up a tensor by name in the tensor array. */
static DACTensor *F(DACTensor *ts, int n, const char *s) {
    for (int i = 0; i < n; i++) if (!strcmp(ts[i].name, s)) return &ts[i];
    return NULL;
}

/* Upload a float32 tensor to GPU memory via HIP stream. */
static float *upload_f32(DACTensor *t, hipStream_t st, DACTensor *ts, int nt, DACTensor *bias_t) {
    if (!t || !t->data) return NULL;
    if (t->dev_f32) return t->dev_f32;

    if (t->elem_size == 4) {
        hipMalloc(&t->dev_f32, t->data_size);
        hipMemcpyAsync(t->dev_f32, t->data, t->data_size, hipMemcpyHostToDevice, st);
    } else if (t->elem_size == 1) {
        char gn[256];
        strncpy(gn, t->name, 255);
        gn[255] = 0;
        char *vp = strstr(gn, "weight_v");
#include "hip_mid.inc"
        snprintf(gw->name, sizeof(gw->name), "%s", snake_names[i]);
        gw->Ci = ta->dims[0]; gw->K = 1; gw->Co = 1; gw->is_convt = 0;
        hipMalloc(&gw->d_data, n_bytes);
        hipMemcpy(gw->d_data, ta->data, n_bytes, hipMemcpyHostToDevice);
    }

    /* Upload encoder snake alphas with "enc:" prefix */
    const char *enc_snake_names[] = {
        "encoder.model.1.block.0.alpha",
        "encoder.model.2.block.0.alpha",
        "encoder.model.3.block.0.alpha",
        "encoder.model.4.block.0.alpha",
        "encoder.model.5.alpha",
        "encoder.model.1.block.2.block.0.alpha",
        "encoder.model.1.block.3.block.0.alpha",
        "encoder.model.1.block.4.block.0.alpha",
        "encoder.model.2.block.2.block.0.alpha",
        "encoder.model.2.block.3.block.0.alpha",
        "encoder.model.2.block.4.block.0.alpha",
        "encoder.model.3.block.2.block.0.alpha",
        "encoder.model.3.block.3.block.0.alpha",
        "encoder.model.3.block.4.block.0.alpha",
        "encoder.model.4.block.2.block.0.alpha",
        "encoder.model.4.block.3.block.0.alpha",
        "encoder.model.4.block.4.block.0.alpha",
    };
    for (int i = 0; i < (int)(sizeof(enc_snake_names)/sizeof(enc_snake_names[0])) && b->n_gpu_weights < MAX_GPU_WEIGHTS; i++) {
        DACTensor *ta = F(ts, nt, enc_snake_names[i]);
        if (!ta) continue;
        size_t n_bytes = (size_t)ta->dims[0] * sizeof(float);
        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
        snprintf(gw->name, sizeof(gw->name), "enc:%s", enc_snake_names[i]);
        gw->Ci = ta->dims[0]; gw->K = 1; gw->Co = 1; gw->is_convt = 0;
        hipMalloc(&gw->d_data, n_bytes);
        hipMemcpy(gw->d_data, ta->data, n_bytes, hipMemcpyHostToDevice);
    }

    /* Upload all codebooks */
    {
        int total_entries = 0;
        for (int cb = 0; cb < 12; cb++) {
            char cb_name[160];
            snprintf(cb_name, sizeof(cb_name), "quantizer.quantizers.%d.codebook.weight", cb);
            DACTensor *cb_t = F(ts, nt, cb_name);
            if (!cb_t) break;
            b->cb_offsets[cb] = total_entries;
            total_entries += cb_t->dims[0] * cb_t->dims[1];
        }
        b->n_cb = 12;
#include "hip_upload.inc"
                d_cur = d_tmp;
            }
        }
    }

    /* encoder.model.0: Conv1d(1536->1024, K=7) */
    GpuWeight *e0 = hip_find_enc_weight(b, "encoder.model.0");
    float *d_features = hip_backend_get_buf(b, 5, 1024 * cur_T * sizeof(float));
    if (!d_features) return TSAC_ERR_MEMORY;
    hipMemsetAsync(d_features, 0, 1024 * cur_T * sizeof(float), s);
    if (e0) {
        conv1d_k<<<dim3(1024, (cur_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
            d_features, d_cur, e0->d_data, NULL,
            cur_T, e0->K, e0->Ci, e0->Co, 1);
        GpuWeight *b0 = hip_find_enc_weight(b, "encoder.model.0.bias");
        if (b0) {
            add_bias_k<<<dim3(cur_T, (1024+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                d_features, b0->d_data, cur_T, 1024);
        }
    }

    hipStreamSynchronize(s);

    /* RVQ quantization */
    float *d_residual;
    hipMalloc(&d_residual, 1024 * cur_T * sizeof(float));
    hipMemcpy(d_residual, d_features, 1024 * cur_T * sizeof(float), hipMemcpyDeviceToDevice);

    float *d_indices;
    hipMalloc(&d_indices, cur_T * sizeof(float));

    for (int cb = 0; cb < n_codebooks && cb < b->n_cb; cb++) {
        /* Quantize against codebook cb */
        rvq_quantize_k<<<(cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
            d_indices, d_residual,
            b->d_cb_data + b->cb_offsets[cb],
            cur_T, 1024, b->cb_entries);
        hipGetLastError();

        /* Copy indices to host */
        float *h_indices_f = (float *)malloc(cur_T * sizeof(float));
        hipMemcpy(h_indices_f, d_indices, cur_T * sizeof(float), hipMemcpyDeviceToHost);
        for (int i = 0; i < cur_T; i++) {
            codes[cb * cur_T + i] = (int)h_indices_f[i];
        }
        free(h_indices_f);

        /* Subtract codebook entry */
        rvq_subtract_k<<<(cur_T * 1024 + BLK - 1) / BLK, BLK, 0, s>>>(
            d_residual, b->d_cb_data + b->cb_offsets[cb],
            d_indices, cur_T, 1024);
        hipGetLastError();
    }

    hipFree(d_residual);
    hipFree(d_indices);

    hipStreamSynchronize(s);

    fprintf(stderr, "[hip] encoder completed: frames=%d\n", cur_T);
    return TSAC_OK;
}
