/*
 * cuda_kernels.cu — CUDA kernels for TSAC neural audio codec.
 * Target: SM 8.9 (RTX 4060), Runtime API.
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define BLK 256

__global__ void conv1d_kernel(float *o, const float *x, const float *w, const float *b,
                               int T, int K, int Ci, int Co) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Co * T;
    if (tid >= total) return;

    int oi = tid % T;
    int oc = tid / T;
    int P = K / 2;

    float sum = b ? b[oc] : 0.0f;
    for (int ic = 0; ic < Ci; ic++) {
        for (int j = 0; j < K; j++) {
            int ii = oi + j - P;
            if (ii >= 0 && ii < T)
                sum += x[ic * T + ii] * w[oc * Ci * K + ic * K + j];
        }
    }
    o[tid] = sum;
}

__global__ void conv_transpose1d_kernel(float *o, const float *x, const float *w,
                                         int Ti, int To, int K, int Ci, int Co) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Co * To;
    if (tid >= total) return;

    int oi = tid % To;
    int oc = tid / To;
    int P = K / 2;

    float sum = 0.0f;
    for (int ic = 0; ic < Ci; ic++) {
        for (int j = 0; j < K; j++) {
            int tmp = oi + P - j;
            if (tmp >= 0 && tmp % 2 == 0) {
                int ii = tmp / 2;
                if (ii < Ti)
                    sum += x[ic * Ti + ii] * w[oc * Ci * K + ic * K + j];
            }
        }
    }
    o[tid] = sum;
}

__global__ void snake_kernel(float *x, const float *alpha, int n, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float a = alpha[i % C];
    if (a < 1e-6f) a = 1e-6f;
    float s = __sinf(a * v);
    x[i] = v + s * s / a;
}

__global__ void add_kernel(float *o, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}

__global__ void add_bias_kernel(float *x, const float *bias, int T, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C * T) return;
    int c = i / T;
    x[i] += bias[c];
}

__global__ void tanh_clip_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (v > 1.0f) v = tanhf(v);
    if (v < -1.0f) v = tanhf(v);
    x[i] = v;
}

__global__ void rvq_quantize_kernel(int *indices, const float *features,
    const float *codebook, int n_frames, int rvq_dim, int entries) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_frames) return;

    const float *feat = features + f * rvq_dim;
    float best_dist = 1e30f;
    int best_entry = 0;

    for (int e = 0; e < entries; e++) {
        const float *cb = codebook + e * rvq_dim;
        float dist = 0.0f;
        for (int d = 0; d < rvq_dim; d++) {
            float diff = feat[d] - cb[d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_entry = e;
        }
    }
    indices[f] = best_entry;
}

__global__ void rvq_subtract_kernel(float *features, const float *codebook,
    const int *indices, int n_frames, int rvq_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_frames * rvq_dim;
    if (tid >= total) return;
    int f = tid / rvq_dim;
    int d = tid % rvq_dim;
    features[tid] -= codebook[indices[f] * rvq_dim + d];
}

__global__ void conv1d_strided_kernel(float *o, const float *x, const float *w, const float *b,
    int T, int K, int Ci, int Co, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Co * T;
    if (tid >= total) return;

    int oi = tid % T;
    int oc = tid / T;
    int P = K / 2;

    float sum = b ? b[oc] : 0.0f;
    for (int ic = 0; ic < Ci; ic++) {
        for (int j = 0; j < K; j++) {
            int ii = oi * stride + j - P;
            if (ii >= 0 && ii < T * stride)
                sum += x[ic * (T * stride) + ii] * w[oc * Ci * K + ic * K + j];
        }
    }
    o[tid] = sum;
}

__global__ void rvq_lookup_kernel(float *features, const int *codes,
                                   const float *cb_data, const int *cb_offsets,
                                   int n_frames, int n_cb, int rvq_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rvq_dim * n_frames;
    if (tid >= total) return;

    int frame = tid / rvq_dim;
    int d = tid % rvq_dim;

    float sum = 0.0f;
    for (int cb = 0; cb < n_cb; cb++) {
        int entry = codes[frame * n_cb + cb];
        int cb_dim = 1024;
        int base = cb_offsets[cb];
        sum += cb_data[base + entry * cb_dim + d];
    }
    features[tid] = sum;
}

__global__ void group_norm_kernel(float *o, const float *x, const float *w, const float *b,
                                   int N, int G, float eps) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int gid = blockIdx.x;
    int E = N / G;
    int base = gid * E;

    float *s_mean = shared;
    float *s_var  = shared + blockDim.x;

    float loc_sum = 0.0f, loc_sq = 0.0f;
    for (int i = tid; i < E; i += blockDim.x) {
        float v = x[base + i];
        loc_sum += v;
        loc_sq  += v * v;
    }

    s_mean[tid] = loc_sum;
    s_var[tid]  = loc_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s_mean[tid] += s_mean[tid + s]; s_var[tid] += s_var[tid + s]; }
        __syncthreads();
    }

    float mn = s_mean[0] / (float)E;
    float vr = s_var[0] / (float)E - mn * mn;
    float is = rsqrtf(fmaxf(vr + eps, 1e-10f));

    for (int i = tid; i < E; i += blockDim.x) {
        int idx = base + i;
        o[idx] = (x[idx] - mn) * is * (w ? w[gid] : 1.0f) + (b ? b[gid] : 0.0f);
    }
}

extern "C" {

cudaError_t launch_conv1d(float *d_o, const float *d_x, const float *d_w, const float *d_b,
                           int T, int K, int Ci, int Co, cudaStream_t stream) {
    int total = Co * T;
    int grid = (total + BLK - 1) / BLK;
    conv1d_kernel<<<grid, BLK, 0, stream>>>(d_o, d_x, d_w, d_b, T, K, Ci, Co);
    return cudaGetLastError();
}

cudaError_t launch_conv_transpose1d(float *d_o, const float *d_x, const float *d_w,
                                     int Ti, int To, int K, int Ci, int Co, cudaStream_t stream) {
    int total = Co * To;
    int grid = (total + BLK - 1) / BLK;
    conv_transpose1d_kernel<<<grid, BLK, 0, stream>>>(d_o, d_x, d_w, Ti, To, K, Ci, Co);
    return cudaGetLastError();
}

cudaError_t launch_snake(float *d_x, const float *d_alpha, int n, int C, cudaStream_t stream) {
    int grid = (n + BLK - 1) / BLK;
    snake_kernel<<<grid, BLK, 0, stream>>>(d_x, d_alpha, n, C);
    return cudaGetLastError();
}

cudaError_t launch_add(float *d_o, const float *d_a, const float *d_b, int n, cudaStream_t stream) {
    int grid = (n + BLK - 1) / BLK;
    add_kernel<<<grid, BLK, 0, stream>>>(d_o, d_a, d_b, n);
    return cudaGetLastError();
}

cudaError_t launch_add_bias(float *d_x, const float *d_bias, int T, int C, cudaStream_t stream) {
    int n = C * T;
    int grid = (n + BLK - 1) / BLK;
    add_bias_kernel<<<grid, BLK, 0, stream>>>(d_x, d_bias, T, C);
    return cudaGetLastError();
}

cudaError_t launch_rvq_lookup(float *d_features, const int *d_codes,
                               const float *d_cb_data, const int *d_cb_offsets,
                               int n_frames, int n_cb, int rvq_dim, cudaStream_t stream) {
    int total = rvq_dim * n_frames;
    int grid = (total + BLK - 1) / BLK;
    rvq_lookup_kernel<<<grid, BLK, 0, stream>>>(d_features, d_codes, d_cb_data, d_cb_offsets,
                                                  n_frames, n_cb, rvq_dim);
    return cudaGetLastError();
}

cudaError_t launch_group_norm(float *d_o, const float *d_x, const float *d_w, const float *d_b,
                               int N, int G, float eps, int block_size, cudaStream_t stream) {
    int shmem = 2 * block_size * sizeof(float);
    group_norm_kernel<<<G, block_size, shmem, stream>>>(d_o, d_x, d_w, d_b, N, G, eps);
    return cudaGetLastError();
}

cudaError_t launch_tanh_clip(float *d_x, int n, cudaStream_t stream) {
    int grid = (n + BLK - 1) / BLK;
    tanh_clip_kernel<<<grid, BLK, 0, stream>>>(d_x, n);
    return cudaGetLastError();
}

cudaError_t launch_rvq_quantize(int *d_indices, const float *d_features,
    const float *d_codebook, int n_frames, int rvq_dim, int entries, cudaStream_t stream) {
    int grid = (n_frames + BLK - 1) / BLK;
    rvq_quantize_kernel<<<grid, BLK, 0, stream>>>(d_indices, d_features, d_codebook,
                                                    n_frames, rvq_dim, entries);
    return cudaGetLastError();
}

cudaError_t launch_rvq_subtract(float *d_features, const float *d_codebook,
    const int *d_indices, int n_frames, int rvq_dim, cudaStream_t stream) {
    int total = n_frames * rvq_dim;
    int grid = (total + BLK - 1) / BLK;
    rvq_subtract_kernel<<<grid, BLK, 0, stream>>>(d_features, d_codebook, d_indices,
                                                   n_frames, rvq_dim);
    return cudaGetLastError();
}

cudaError_t launch_conv1d_strided(float *d_o, const float *d_x, const float *d_w, const float *d_b,
    int T_out, int K, int Ci, int Co, int T_in, int stride, cudaStream_t stream) {
    int total = Co * T_out;
    int grid = (total + BLK - 1) / BLK;
    conv1d_strided_kernel<<<grid, BLK, 0, stream>>>(d_o, d_x, d_w, d_b, T_out, K, Ci, Co, stride);
    return cudaGetLastError();
}

} /* extern "C" */
