
/* ================================================================ */
/*  Element-wise binary ops                                         */
/* ================================================================ */

template<typename T>
__global__ void add_kernel(T *out, const T *a, const T *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)a[i] + (float)b[i];
}

template<typename T>
__global__ void mul_kernel(T *out, const T *a, const T *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)a[i] * (float)b[i];
}

template<typename T>
__global__ void sub_kernel(T *out, const T *a, const T *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)a[i] - (float)b[i];
}

template<typename T>
__global__ void div_kernel(T *out, const T *a, const T *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)a[i] / (float)b[i];
}

template<typename T>
__global__ void mul_scalar_kernel(T *out, const T *in, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i] * s;
}

template<typename T>
__global__ void add_scalar_kernel(T *out, const T *in, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i] + s;
}

/* ================================================================ */
/*  Conversions (f32 ↔ bf16 ↔ f16)                                 */
/* ================================================================ */

typedef uint16_t bf16_t;
typedef uint16_t f16_t;

__global__ void f32_to_bf16_kernel(bf16_t *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t u; memcpy(&u, &in[i], 4);
        uint32_t rounding = 0x7fff + ((u >> 16) & 1);
        out[i] = (bf16_t)((u + rounding) >> 16);
    }
}

__global__ void bf16_to_f32_kernel(float *out, const bf16_t *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t u = (uint32_t)in[i] << 16;
        memcpy(&out[i], &u, 4);
    }
}

/* ================================================================ */
/*  Reductions (block-level)                                        */
/* ================================================================ */

__global__ void reduce_sum_kernel(float *out, const float *in, int n) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = shared[0];
}

__global__ void reduce_max_kernel(float *out, const float *in, int n) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (i < n) ? in[i] : -1e10f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = shared[0];
}

/* ================================================================ */
/*  Normalization                                                    */
/* ================================================================ */

__global__ void layer_norm_kernel(float *out, const float *in,
                                   const float *weight, const float *bias,
