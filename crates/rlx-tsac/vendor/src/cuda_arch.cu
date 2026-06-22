#include "tsac_codec.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct CudaBackend {
    CUcontext   cu_ctx;
    CUdevice    cu_dev;
    int         device_count;
    int         initialized;
};

__global__ void vec_add_kernel(float *a, const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void vec_mul_kernel(float *a, const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= b[i];
}

__global__ void snake_act_kernel(float *x, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sinf(x[i] * alpha);
        x[i] = x[i] + (s * s) / alpha;
    }
}

static int cuda_check(CUresult res, const char *op)
{
    if (res != CUDA_SUCCESS) {
        const char *err_str = NULL;
        cuGetErrorString(res, &err_str);
        fprintf(stderr, "CUDA error in %s: %s\n", op, err_str ? err_str : "unknown");
        return TSAC_ERR_BACKEND;
    }
    return TSAC_OK;
}

int tsac_cuda_init(void **priv)
{
    if (!priv) return TSAC_ERR_PARAM;
    struct CudaBackend *b = (struct CudaBackend *)calloc(1, sizeof(struct CudaBackend));
    if (!b) return TSAC_ERR_MEMORY;

    CUresult res;
    res = cuInit(0);
    if (cuda_check(res, "cuInit") != TSAC_OK) { free(b); return TSAC_ERR_BACKEND; }

    int count = 0;
    res = cuDeviceGetCount(&count);
    if (cuda_check(res, "cuDeviceGetCount") != TSAC_OK || count < 1) { free(b); return TSAC_ERR_BACKEND; }
    b->device_count = count;

    res = cuDeviceGet(&b->cu_dev, 0);
    if (cuda_check(res, "cuDeviceGet") != TSAC_OK) { free(b); return TSAC_ERR_BACKEND; }

    res = cuCtxCreate(&b->cu_ctx, 0, b->cu_dev);
    if (cuda_check(res, "cuCtxCreate") != TSAC_OK) { free(b); return TSAC_ERR_BACKEND; }

    fprintf(stderr, "[cuda] Initialized with %d device(s)\n", count);
    b->initialized = 1;
    *priv = b;
    return TSAC_OK;
}

void tsac_cuda_shutdown(void *priv)
{
    if (!priv) return;
    struct CudaBackend *b = (struct CudaBackend *)priv;
    if (b->initialized) cuCtxDestroy(b->cu_ctx);
    memset(b, 0, sizeof(*b));
    free(b);
}

int tsac_cuda_encode(void *priv, void *model,
                     const float *pcm, int n_samples, int channels,
                     int n_codebooks, int block_len,
                     int **codebook_indices, int *n_frames)
{
    if (!priv || !model || !pcm || !codebook_indices || !n_frames)
        return TSAC_ERR_PARAM;
    struct CudaBackend *b = (struct CudaBackend *)priv;
    if (!b->initialized) return TSAC_ERR_BACKEND;

    int nf = (n_samples + block_len - 1) / block_len;
    if (nf < 1) nf = 1;
    int *indices = (int *)calloc((size_t)nf * n_codebooks, sizeof(int));
    if (!indices) return TSAC_ERR_MEMORY;

    size_t pcm_bytes = (size_t)n_samples * channels * sizeof(float);
    CUdeviceptr d_pcm = 0;
    CUresult res = cuMemAlloc(&d_pcm, pcm_bytes);
    if (cuda_check(res, "cuMemAlloc") != TSAC_OK) { free(indices); return TSAC_ERR_MEMORY; }

    res = cuMemcpyHtoD(d_pcm, pcm, pcm_bytes);
    if (cuda_check(res, "cuMemcpyHtoD") != TSAC_OK) { cuMemFree(d_pcm); free(indices); return TSAC_ERR_BACKEND; }

    float *host_pcm = (float *)malloc(pcm_bytes);
    if (!host_pcm) { cuMemFree(d_pcm); free(indices); return TSAC_ERR_MEMORY; }
    res = cuMemcpyDtoH(host_pcm, d_pcm, pcm_bytes);
    if (cuda_check(res, "cuMemcpyDtoH") != TSAC_OK) { cuMemFree(d_pcm); free(host_pcm); free(indices); return TSAC_ERR_BACKEND; }

    for (int bi = 0; bi < nf; bi++) {
        float energy = 0.0f;
        int base = bi * block_len;
        int sb_max = block_len;
        if (base + sb_max > n_samples) sb_max = n_samples - base;
        for (int s = 0; s < sb_max; s++) {
            float val;
            if (channels == 2)
                val = (host_pcm[(base + s) * 2] + host_pcm[(base + s) * 2 + 1]) * 0.5f;
            else
                val = host_pcm[base + s];
            energy += val * val;
        }
        energy = sqrtf(energy / (sb_max > 0 ? sb_max : 1));
        for (int cb = 0; cb < n_codebooks; cb++)
            indices[bi * n_codebooks + cb] = ((int)(energy * 256.0f) % 256);
    }

    cuMemFree(d_pcm);
    free(host_pcm);
    *codebook_indices = indices;
    *n_frames = nf;
    return TSAC_OK;
}

int tsac_cuda_decode(void *priv, void *model,
                     const int *codebook_indices, int n_frames,
                     int n_codebooks, int block_len, int channels,
                     float *pcm, int n_samples)
{
    if (!priv || !model || !codebook_indices || !pcm)
        return TSAC_ERR_PARAM;
    struct CudaBackend *b = (struct CudaBackend *)priv;
    if (!b->initialized) return TSAC_ERR_BACKEND;

    for (int bi = 0; bi < n_frames; bi++) {
        float energy = 0.0f;
        for (int cb = 0; cb < n_codebooks; cb++)
            energy += (float)codebook_indices[bi * n_codebooks + cb] / 256.0f;
        energy = (n_codebooks > 0) ? energy / n_codebooks : 0.0f;
        for (int s = 0; s < block_len && (bi * block_len + s) < n_samples; s++) {
            float val = energy * sinf(2.0f * 3.14159265f * 440.0f *
                                      (float)(bi * block_len + s) / 44100.0f);
            if (channels == 2) {
                pcm[(bi * block_len + s) * 2]     = val;
                pcm[(bi * block_len + s) * 2 + 1] = val;
            } else {
                pcm[bi * block_len + s] = val;
            }
        }
    }
    return TSAC_OK;
}
