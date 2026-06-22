/*
 * nc_cuda_device.c — CUDA backend for LibNC.
 *
 * Reconstructed from libnc_cuda.so binary (TSAC package by Fabrice Bellard).
 * This is a clean-room reimplementation that uses the CUDA Driver API
 * directly (matching the original's approach), enabling fine-grained
 * control over kernel loading, memory management, and stream scheduling.
 *
 * Architecture Abstraction:
 *   The public entry points (nc_new_cuda_device, cuda_set_device_param)
 *   register into the NCArchOps vtable so the same host code path works
 *   across CUDA, HIP, Metal, and other backends.
 */

#include "../nc_internal.h"
#include <cuda.h>
#include <cublasLt.h>
#include <dlfcn.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call) do {                                         \
    CUresult _err = (call);                                           \
    if (_err != CUDA_SUCCESS) {                                       \
        const char *_err_str = "unknown";                             \
        cuGetErrorString(_err, &_err_str);                            \
        NC_LOG("CUDA error %d (%s) at %s:%d",                        \
               (int)_err, _err_str, __FILE__, __LINE__);              \
        return NULL;                                                  \
    }                                                                 \
} while (0)

#define CUDA_CHECK_VOID(call) do {                                    \
    CUresult _err = (call);                                           \
    if (_err != CUDA_SUCCESS) {                                       \
        const char *_err_str = "unknown";                             \
        cuGetErrorString(_err, &_err_str);                            \
        NC_LOG("CUDA error %d (%s) at %s:%d",                        \
               (int)_err, _err_str, __FILE__, __LINE__);              \
        return;                                                       \
    }                                                                 \
} while (0)

/* ------------------------------------------------------------------ */
/*  Embedded fatbin reference (from .rodata of original libnc_cuda.so) */
/* ------------------------------------------------------------------ */

/*
 * nc_cuda_ops_fatbin is expected to be linked from an external object file
 * generated from the extracted fatbin data (tools/gen_fatbin.c).
 * When the kernels are compiled as separate .cu files instead, this is NULL.
 */
extern const unsigned char nc_cuda_ops_fatbin[];
extern const unsigned int  nc_cuda_ops_fatbin_len;

/* ------------------------------------------------------------------ */
/*  CUDA architecture ops (NCArchOps vtable implementation)            */
/* ------------------------------------------------------------------ */

static NCArchDeviceImpl *cuda_arch_create(void *ctx, int device_index,
                                          uint32_t flags)
{
    fprintf(stderr, "[nc_cuda] cuda_arch_create(ctx=%p, device=%d, flags=%u)\n", ctx, device_index, flags);
    NCCudaDeviceState *s = (NCCudaDeviceState *)calloc(1, sizeof(NCCudaDeviceState));
    if (!s) return NULL;

    s->device_index = device_index;
    s->flags = flags;

    CUDA_CHECK(cuInit(0));

    int dev_count = 0;
    CUDA_CHECK(cuDeviceGetCount(&dev_count));
    if (device_index >= dev_count) {
        NC_LOG("No CUDA device available (requested %d, found %d)",
               device_index, dev_count);
        free(s);
        return NULL;
    }

    CUDA_CHECK(cuDeviceGet(&s->cu_device, device_index));
    CUDA_CHECK(cuCtxCreate(&s->cu_context, NULL, 0, s->cu_device));

    /* Query device properties */
    int val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                         s->cu_device);
    s->sm_count = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_WARP_SIZE, s->cu_device);
    s->warp_size = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                         s->cu_device);
    s->max_threads_per_block = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                         s->cu_device);
    s->max_shared_memory = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                         s->cu_device);
    s->compute_cap_major = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                         s->cu_device);
    s->compute_cap_minor = val;

    /* Create stream */
    CUDA_CHECK(cuStreamCreate(&s->stream, CU_STREAM_DEFAULT));

    /* Load kernel module from embedded fatbin */
    if (nc_cuda_ops_fatbin && nc_cuda_ops_fatbin_len > 0) {
        CUDA_CHECK(cuModuleLoadData(&s->module, nc_cuda_ops_fatbin));
    }

#ifdef NC_CUDA_ENABLED
    /* Initialize cuBLASLt for optimized matmul */
    {
        cublasStatus_t stat = cublasLtCreate(&s->cublaslt_handle);
        if (stat == CUBLAS_STATUS_SUCCESS) {
            s->cublaslt_lib_handle = (void*)1;
            NC_LOG("cuBLASLt initialized");
        } else {
            s->cublaslt_lib_handle = NULL;
            s->cublaslt_handle = NULL;
            NC_LOG("cuBLASLt init failed (%d), matmul will use custom kernels", (int)stat);
        }
    }
#endif

    /* Allocate device constants */
    float one_val = 1.0f, zero_val = 0.0f;
    CUDA_CHECK(cuMemAlloc(&s->one_ptr, sizeof(float)));
    CUDA_CHECK(cuMemAlloc(&s->zero_ptr, sizeof(float)));
    CUDA_CHECK(cuMemcpyHtoD(s->one_ptr, &one_val, sizeof(float)));
    CUDA_CHECK(cuMemcpyHtoD(s->zero_ptr, &zero_val, sizeof(float)));

    /* Initialize fast division table */
    fastdiv_init(s->fastdiv_table, 256);
    s->fastdiv_table_len = 256;

    /* Query memory info */
    size_t free_bytes, total_bytes;
    cuMemGetInfo(&free_bytes, &total_bytes);
    s->free_mem = free_bytes;
    s->total_mem = total_bytes;

    NC_LOG("CUDA device %d: %d SMs, CC %d.%d, %zu MB free",
           device_index, s->sm_count,
           s->compute_cap_major, s->compute_cap_minor,
           free_bytes / (1024 * 1024));

    return (NCArchDeviceImpl *)s;
}

static void cuda_arch_destroy(NCArchDeviceImpl *dev)
{
    NCCudaDeviceState *s = (NCCudaDeviceState *)dev;
    if (!s) return;

#ifdef NC_CUDA_ENABLED
    if (s->cublaslt_handle)
        cublasLtDestroy(s->cublaslt_handle);
#endif
    if (s->lt_workspace)    cuMemFree(s->lt_workspace);
    if (s->one_ptr)         cuMemFree(s->one_ptr);
    if (s->zero_ptr)        cuMemFree(s->zero_ptr);
    if (s->module)          cuModuleUnload(s->module);
    if (s->stream)          cuStreamDestroy(s->stream);
    if (s->cu_context)      cuCtxDestroy(s->cu_context);

    free(s);
}

static void *cuda_arch_alloc(NCArchDeviceImpl *dev, size_t size)
{
    NCCudaDeviceState *s = (NCCudaDeviceState *)dev;
    fprintf(stderr, "[nc_cuda] alloc size=%zu\n", size);
    CUdeviceptr ptr;
    CUresult err = cuMemAlloc(&ptr, size);
    if (err != CUDA_SUCCESS) {
        NC_LOG("libnc_cuda: could not allocate %zu bytes, exiting", size);
        return NULL;
    }
    return (void *)ptr;
}

static void cuda_arch_free(NCArchDeviceImpl *dev, void *ptr)
{
    cuMemFree((CUdeviceptr)(uintptr_t)ptr);
}

static void cuda_arch_memset(NCArchDeviceImpl *dev, void *ptr, int val, size_t count)
{
    NCCudaDeviceState *s = (NCCudaDeviceState *)dev;
    CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)ptr;

    switch (val & 0xff) {
    case 0:
        cuMemsetD8Async(dptr, 0, count, s->stream);
        break;
    default:
        /* Use 32-bit memset for larger patterns */
        cuMemsetD32Async(dptr, (uint32_t)(unsigned)val,
                         (count + 3) / 4, s->stream);
        break;
    }
}

static void cuda_arch_memcpy_htod(NCArchDeviceImpl *dev, void *dst,
                                   const void *src, size_t n)
{
    cuMemcpyHtoD((CUdeviceptr)(uintptr_t)dst, src, n);
}

static void cuda_arch_memcpy_dtoh(NCArchDeviceImpl *dev, void *dst,
                                   const void *src, size_t n)
{
    cuMemcpyDtoH(dst, (CUdeviceptr)(uintptr_t)src, n);
}

static int cuda_arch_load_kernel(NCArchDeviceImpl *dev, const char *name,
                                  const void *code, size_t code_size)
{
    NCCudaDeviceState *s = (NCCudaDeviceState *)dev;
    CUfunction func;

    /* Check cache first */
    for (int i = 0; i < s->n_cached_kernels; i++) {
        if (s->cached_kernels[i] == (CUfunction)(uintptr_t)0x1)
            continue; /* skip if we ever need invalidation */
    }

    CUresult err = cuModuleGetFunction(&func, s->module, name);
    if (err != CUDA_SUCCESS) {
        NC_LOG("Kernel '%s' not found in module", name);
        return -1;
    }

    /* Cache it */
    if (s->n_cached_kernels < 256) {
        s->cached_kernels[s->n_cached_kernels++] = func;
    }

    return 0;
}

static void cuda_arch_launch(NCArchDeviceImpl *dev, const char *kernel_name,
                              int grid_x, int grid_y, int grid_z,
                              int block_x, int block_y, int block_z,
                              int shared_mem, void **args)
{
    NCCudaDeviceState *s = (NCCudaDeviceState *)dev;
    fprintf(stderr, "[nc_cuda] launch kernel='%s' grid=<%d,%d,%d> block=<%d,%d,%d> shmem=%d\n",
            kernel_name ? kernel_name : "(null)",
            grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem);

    CUfunction func;
    CUresult err = cuModuleGetFunction(&func, s->module, kernel_name);
    if (err != CUDA_SUCCESS) {
        NC_LOG("Kernel '%s' not found", kernel_name);
        return;
    }

    /* Set max dynamic shared memory if needed */
    if (shared_mem > 0) {
        cuFuncSetAttribute(func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_mem);
    }

    err = cuLaunchKernel(func,
                         grid_x, grid_y, grid_z,
                         block_x, block_y, block_z,
                         shared_mem, s->stream, args, NULL);
    if (err != CUDA_SUCCESS) {
        const char *err_str = "unknown";
        cuGetErrorString(err, &err_str);
        NC_LOG("cuLaunchKernel(%s) failed: %s", kernel_name, err_str);
    }
}

static void cuda_arch_synchronize(NCArchDeviceImpl *dev)
{
    NCCudaDeviceState *s = (NCCudaDeviceState *)dev;
    cuStreamSynchronize(s->stream);
}

/* Architecture ops vtable instance */
static NCArchOps cuda_arch_ops = {
    .name        = "CUDA",
    .arch_type   = NC_ARCH_CUDA,
    .create      = cuda_arch_create,
    .destroy     = cuda_arch_destroy,
    .alloc       = cuda_arch_alloc,
    .free        = cuda_arch_free,
    .memset      = cuda_arch_memset,
    .memcpy_htod = cuda_arch_memcpy_htod,
    .memcpy_dtoh = cuda_arch_memcpy_dtoh,
    .load_kernel = cuda_arch_load_kernel,
    .launch      = cuda_arch_launch,
    .synchronize = cuda_arch_synchronize,
};

/* ------------------------------------------------------------------ */
/*  Public API: CUDA device creation                                   */
/* ------------------------------------------------------------------ */

/*
 * This function is called by the original libnc.so (CPU library) via dlsym.
 * It creates the CUDA backend state and returns an opaque pointer.
 * The caller stores this pointer in NCDevice.arch_dev.
 *
 * Original libnc_cuda.so allocates 0x80 bytes for the state struct.
 * We allocate NCCudaDeviceState which is larger but identical in layout
 * for the fields that the original libnc.so accesses through the vtable.
 */

void *nc_new_cuda_device_internal(void *ctx_ptr, int device_index,
                                   uint32_t flags)
{
    /*
     * ctx_ptr is the NCContext* from the original libnc.so.
     * We forward it to cuda_arch_create which expects void* ctx.
     */
    void *state = cuda_arch_ops.create(ctx_ptr, device_index, flags);
    if (!state) {
        fprintf(stderr, "[libnc] Failed to create CUDA device\n");
        return NULL;
    }
    return state;
}

void cuda_set_device_param(const char *key, const char *value)
{
    /*
     * Original libnc_cuda.so accepted parameters like:
     *   "device" -> CUDA device index
     *   "batch_size" -> force batch size
     * These are stored globally and applied on next nc_new_cuda_device call.
     */
    (void)key;
    (void)value;
}

/* ------------------------------------------------------------------ */
/*  Kernel name resolution                                             */
/* ------------------------------------------------------------------ */

/*
 * Returns the CUDA kernel function name for a given operation and dtype.
 * The original libnc_cuda.so fatbin uses naming convention:
 *   cu_<op>_<dtype_suffix>
 * where dtype_suffix is one of: f32, f16, bf16, i32, i8, u8, u16, u32.
 */
const char *nc_cuda_kernel_name(const char *op_prefix, NCType dtype)
{
    static char name_buf[128];

    const char *suffix;
    switch (dtype) {
    case NC_TYPE_F32:  suffix = "f32";  break;
    case NC_TYPE_F16:  suffix = "f16";  break;
    case NC_TYPE_BF16: suffix = "bf16"; break;
    case NC_TYPE_I32:  suffix = "i32";  break;
    case NC_TYPE_I8:   suffix = "i8";   break;
    default:           suffix = "f32";  break;
    }

    snprintf(name_buf, sizeof(name_buf), "cu_%s_%s", op_prefix, suffix);
    return name_buf;
}

/* ------------------------------------------------------------------ */
/*  Tensor operation dispatchers (CUDA backend)                       */
/* ------------------------------------------------------------------ */

/*
 * Each op dispatches to the appropriate CUDA kernel.
 * In the full implementation, these would be auto-generated from
 * kernel templates. Here we provide the dispatch skeleton.
 */

static NCCudaDeviceState *get_cuda_state(NCTensor *x)
{
    if (!x || !x->device || x->device->kind != NC_DEVICE_CUDA)
        return NULL;
    return (NCCudaDeviceState *)x->device->arch_dev;
}

/* ------------------------------------------------------------------ */
/*  Element-wise operations                                            */
/* ------------------------------------------------------------------ */

NCTensor *nc_add_cuda(NCTensor *a, NCTensor *b)
{
    NCCudaDeviceState *s = get_cuda_state(a);
    if (!s) return NULL;

    /* Launch cu_add_<dtype> kernel */
    const char *kname = nc_cuda_kernel_name("add", a->dtype);
    int n_elements = (int)a->n_elements;

    int block_size = 256;
    int grid_size = (n_elements + block_size - 1) / block_size;

    CUdeviceptr d_a = (CUdeviceptr)(uintptr_t)a->buffer->backend_ptr;
    CUdeviceptr d_b = (CUdeviceptr)(uintptr_t)b->buffer->backend_ptr;

    /* Create output tensor */
    NCTensor *out = nc_new_tensor(a->device, a->dtype,
                                   a->n_dims, a->dims);
    CUdeviceptr d_out = (CUdeviceptr)(uintptr_t)out->buffer->backend_ptr;

    void *args[] = { &d_out, &d_a, &d_b, &n_elements };
    cuda_arch_launch((NCArchDeviceImpl *)s, kname,
                     grid_size, 1, 1,
                     block_size, 1, 1,
                     0, args);

    /* Consume inputs (refcounting) */
    nc_free_tensor(a);
    nc_free_tensor(b);

    return out;
}

/* ------------------------------------------------------------------ */
/*  Matrix multiplication                                              */
/* ------------------------------------------------------------------ */

NCTensor *nc_matmul_add_cuda(NCTensor *a, NCTensor *b, NCTensor *bias)
{
    NCCudaDeviceState *s = get_cuda_state(a);
    if (!s) return NULL;

    /*
     * Dispatch to appropriate cu_matmul_nn_* kernel based on dtype
     * and matrix dimensions. The original binary has ~24 specialized
     * matmul kernels for different shapes and quantization formats.
     */
    (void)bias;

    size_t M = a->dims[1];  /* inner dim (column-first) */
    size_t K = a->dims[0];
    size_t N = b->dims[1];

    /* TODO: select optimal kernel variant based on shape heuristics */
    (void)M;
    (void)K;
    (void)N;

    return NULL;  /* placeholder */
}

/* ------------------------------------------------------------------ */
/*  Fused attention                                                   */
/* ------------------------------------------------------------------ */

NCTensor *nc_fused_attention_cuda(NCTensor *q, NCTensor *k, NCTensor *v,
                                   NCTensor *mask)
{
    NCCudaDeviceState *s = get_cuda_state(q);
    if (!s) return NULL;

    /*
     * The original has fused attention kernels for shapes:
     *   64x16x128, 128x16x64, 256x16x16 (with f16 and bf16 variants)
     * as well as 8-head variants (64x16x8, 128x16x8).
     */
    (void)k;
    (void)v;
    (void)mask;

    return NULL;  /* placeholder */
}

/* ------------------------------------------------------------------ */
/*  Normalization operations                                           */
/* ------------------------------------------------------------------ */

NCTensor *nc_layer_norm_cuda(NCTensor *x, NCTensor *weight, NCTensor *bias)
{
    NCCudaDeviceState *s = get_cuda_state(x);
    if (!s) return NULL;

    const char *kname = nc_cuda_kernel_name("layer_norm1", x->dtype);
    (void)kname;
    (void)weight;
    (void)bias;

    return NULL;  /* placeholder */
}

/* ------------------------------------------------------------------ */
/*  Device transfer helpers                                            */
/* ------------------------------------------------------------------ */

NCTensor *nc_tensor_to_cuda(NCTensor *x, NCDevice *dst)
{
    if (!x || !dst || dst->kind != NC_DEVICE_CUDA)
        return x;

    if (x->device == dst)
        return nc_dup_tensor(x);

    /* Allocate buffer on CUDA device and copy */
    NCCudaDeviceState *s = (NCCudaDeviceState *)dst->arch_dev;
    size_t size = nc_dtype_size(x->dtype) * x->n_elements;

    NCTensorBuffer *buf = nc_buffer_new(dst, size);
    if (!buf) return NULL;

    CUdeviceptr d_ptr = (CUdeviceptr)(uintptr_t)buf->backend_ptr;
    cuMemcpyHtoD(d_ptr, x->buffer->data, size);

    NCTensor *y = nc_dup_tensor(x);
    y->device = dst;
    nc_buffer_free(y->buffer);

    y->buffer = buf;
    return y;
}

/* ------------------------------------------------------------------ */
/*  Fast integer division (extracted from libnc_cuda.so)               */
/* ------------------------------------------------------------------ */

void fastdiv_init(uint32_t *table, int table_len)
{
    for (int i = 0; i < table_len; i++) {
        if (i == 0) {
            table[i] = 0;
        } else {
            /*
             * Compute magic multiplier for fast unsigned division by i.
             * This matches the original binary's approach for computing
             * block indices in grid-stride loops.
             */
            table[i] = (uint32_t)(((1ULL << 32) + i - 1) / i);
        }
    }
}
