/* hip_kernels.hip.cpp — HIP/ROCm GPU backend for tsac-ng. */
/*
 * hip_kernels.hip.cpp — HIP kernels for TSAC neural network inference.
 *
 * Implements all operations needed by the DAC decoder + Transformer.
 * Replaces the custom NVIDIA cubins with native ROCm HIP kernels.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ================================================================ */
/*  Element-wise unary ops                                          */
/* ================================================================ */

template<typename T>
__global__ void neg_kernel(T *out, const T *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -in[i];
}

template<typename T>
__global__ void exp_kernel(T *out, const T *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf((float)in[i]);
}

template<typename T>
__global__ void log_kernel(T *out, const T *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf((float)in[i]);
}

template<typename T>
__global__ void relu_kernel(T *out, const T *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] > 0 ? in[i] : 0;
}

template<typename T>
__global__ void gelu_kernel(T *out, const T *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = (float)in[i];
        out[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
    }
}

template<typename T>
__global__ void sigmoid_kernel(T *out, const T *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-(float)in[i]));
}

template<typename T>
__global__ void tanh_kernel(T *out, const T *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf((float)in[i]);
}

template<typename T>
__global__ void swish_kernel(T *out, const T *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = (float)in[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

template<typename T>
__global__ void snake_kernel(T *out, const T *in, const T *alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = (float)in[i];
        float a = (float)alpha[0];
        out[i] = x + sinf(a * x) * sinf(a * x) / a;
    }
}

template<typename T>
__global__ void recip_kernel(T *out, const T *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (float)in[i];
}

template<typename T>
__global__ void sqrt_kernel(T *out, const T *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sqrtf((float)in[i]);
}

template<typename T>
__global__ void sqr_relu_kernel(T *out, const T *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = (float)in[i]; out[i] = x > 0 ? x * x : 0; }
}
#include "hip_kern.inc"
                                   float eps, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    /* Mean */
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        sum += in[row * cols + i];
    shared[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float mean = shared[0] / cols;
    
    /* Variance */
    float var_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float d = in[row * cols + i] - mean;
        var_sum += d * d;
    }
    shared[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float variance = shared[0] / cols;
    float inv_std = rsqrtf(variance + eps);
    
    /* Normalize */
    for (int i = tid; i < cols; i += blockDim.x) {
        out[row * cols + i] = (in[row * cols + i] - mean) * inv_std
                              * (weight ? weight[i] : 1.0f)
                              + (bias ? bias[i] : 0.0f);
    }
}

__global__ void rms_norm_kernel(float *out, const float *in,
                                 const float *weight, float eps, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float shared[];
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        sum_sq += in[row * cols + i] * in[row * cols + i];
    shared[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(shared[0] / cols + eps);
    
    for (int i = tid; i < cols; i += blockDim.x)
        out[row * cols + i] = in[row * cols + i] * rms * (weight ? weight[i] : 1.0f);
}

/* ================================================================ */
/*  Softmax                                                         */
/* ================================================================ */

__global__ void softmax_kernel(float *out, const float *in, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    
    float maxv = -1e10f;
    for (int i = tid; i < cols; i += blockDim.x)
        maxv = fmaxf(maxv, in[row * cols + i]);
    shared[tid] = maxv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    float row_max = shared[0];
    
    float sum_exp = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        sum_exp += expf(in[row * cols + i] - row_max);
    shared[tid] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float row_sum = shared[0];
    
    for (int i = tid; i < cols; i += blockDim.x)
        out[row * cols + i] = expf(in[row * cols + i] - row_max) / row_sum;
}

/* ================================================================ */
/*  Convolution (1D)                                                */
/* ================================================================ */

__global__ void conv1d_kernel(float *out, const float *in,
                               const float *weight, const float *bias,
                               int n_in, int n_out, int ksize,
                               int in_ch, int out_ch, int stride) {
    int oc = blockIdx.x;
    int oi = blockIdx.y * blockDim.y + threadIdx.y;
    if (oc >= out_ch || oi >= n_out) return;
    
    float sum = bias ? bias[oc] : 0.0f;
    for (int ic = 0; ic < in_ch; ic++) {
        for (int k = 0; k < ksize; k++) {
            int ii = oi * stride + k - ksize / 2;
            if (ii >= 0 && ii < n_in) {
                sum += in[ic * n_in + ii] * weight[oc * in_ch * ksize + ic * ksize + k];
            }
        }
    }
    out[oc * n_out + oi] = sum;
}

/* ================================================================ */
/*  Kernel launch wrappers (extern "C" for tsac-ng integration)      */
/* ================================================================ */

extern "C" {

void hip_launch_elem(const char *op, hipStream_t stream,
                     void *out, const void *in, int n, int dtype_size)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    
    if (dtype_size == 4) { /* f32 */
        if (!strcmp(op, "neg"))    neg_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)in, n);
        else if (!strcmp(op, "exp"))   exp_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)in, n);
        else if (!strcmp(op, "relu"))  relu_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)in, n);
        else if (!strcmp(op, "gelu"))  gelu_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)in, n);
        else if (!strcmp(op, "sigmoid")) sigmoid_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)in, n);
        else if (!strcmp(op, "tanh"))  tanh_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)in, n);
        else if (!strcmp(op, "swish")) swish_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)in, n);
        else if (!strcmp(op, "recip")) recip_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)in, n);
        else if (!strcmp(op, "log"))   log_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)in, n);
    }
}

void hip_launch_binary(const char *op, hipStream_t stream,
                        void *out, const void *a, const void *b, int n)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    if (!strcmp(op, "add"))  add_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)a, (const float*)b, n);
    else if (!strcmp(op, "mul"))  mul_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)a, (const float*)b, n);
    else if (!strcmp(op, "sub"))  sub_kernel<float><<<dim3(grid), dim3(block), 0, stream>>>((float*)out, (const float*)a, (const float*)b, n);
}

void hip_launch_layer_norm(hipStream_t stream, float *out, const float *in,
                            const float *weight, const float *bias,
                            float eps, int rows, int cols)
{
    int block = 256;
    int shmem = block * sizeof(float);
    void *args[] = { &out, &in, (void*)&weight, (void*)&bias, &eps, &cols };
    (void)hipLaunchKernel((const void*)layer_norm_kernel, dim3(rows), dim3(block), args, shmem, stream);
}

void hip_launch_softmax(hipStream_t stream, float *out, const float *in,
                         int rows, int cols)
{
    int block = 256;
    int shmem = block * sizeof(float);
    void *args[] = { &out, &in, &cols };
    (void)hipLaunchKernel((const void*)softmax_kernel, dim3(rows), dim3(block), args, shmem, stream);
}

} /* extern "C" */
