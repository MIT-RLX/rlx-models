#ifndef LIBNC_ARCH_H
#define LIBNC_ARCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Architecture abstraction layer for LibNC GPU backends.
 *
 * This header defines the portability interface that allows the same
 * host code to target CUDA (NVIDIA), HIP (AMD), Metal (Apple),
 * SYCL (Intel), and Vulkan Compute (cross-platform).
 *
 * To add a new backend:
 *   1. Implement all functions in NCArchOps
 *   2. Create a device creation function that fills NCArchOps
 *   3. Register in nc_new_device() dispatch
 */

/* Supported architecture types */
typedef enum {
    NC_ARCH_CUDA   = 0,
    NC_ARCH_HIP    = 1,
    NC_ARCH_METAL  = 2,
    NC_ARCH_SYCL   = 3,
    NC_ARCH_VULKAN = 4
} NCArch;

/* Architecture capability flags */
typedef enum {
    NC_CAP_FP16           = 1 << 0,
    NC_CAP_BF16           = 1 << 1,
    NC_CAP_INT8_GEMM      = 1 << 2,
    NC_CAP_TENSOR_CORES   = 1 << 3,
    NC_CAP_COOPERATIVE_GROUPS = 1 << 4,
    NC_CAP_ASYNC_COPY     = 1 << 5,
    NC_CAP_DP4A           = 1 << 6,
    NC_CAP_WGMMA          = 1 << 7,
    NC_CAP_FP8            = 1 << 8,
} NCCapabilityFlags;

/* Memory type hints for allocation */
typedef enum {
    NC_MEM_DEFAULT   = 0,
    NC_MEM_DEVICE    = 1,
    NC_MEM_HOST      = 2,
    NC_MEM_HOST_COHERENT = 3,
} NCMemType;

/* Kernel launch configuration */
typedef struct {
    int grid_x, grid_y, grid_z;
    int block_x, block_y, block_z;
    int shared_mem_bytes;
    int n_arguments;
    void **arguments;
} NCLaunchConfig;

/* Device properties (queried at runtime) */
typedef struct {
    char     name[128];
    NCArch   arch;
    int      compute_capability_major;
    int      compute_capability_minor;
    int      multi_processor_count;
    int      warp_size;
    int      max_threads_per_block;
    int      max_shared_memory_per_block;
    int      max_grid_dim_x;
    int      max_grid_dim_y;
    int      max_grid_dim_z;
    size_t   global_memory_bytes;
    size_t   shared_memory_per_multiprocessor;
    uint32_t capabilities;    /* NCCapabilityFlags */
    int      pci_bus_id;
    int      pci_device_id;
} NCArchDeviceProps;

/* Opaque device handle */
typedef struct NCArchDeviceImpl NCArchDeviceImpl;

/*
 * Architecture ops vtable — every port must implement these.
 *
 * Dev holds an opaque handle (NCArchDeviceImpl *) allocated by create().
 * ctx is the NCContext pointer, passed as void* to avoid circular deps.
 */
typedef struct {
    const char *name;          /* human-readable backend name */
    NCArch      arch_type;

    /* Lifecycle: device_index = GPU ordinal, flags = reserved (0) */
    NCArchDeviceImpl *(*create)(void *ctx, int device_index, uint32_t flags);
    void              (*destroy)(NCArchDeviceImpl *dev);

    /* Memory management */
    void *(*alloc)(NCArchDeviceImpl *dev, size_t size);
    void  (*free)(NCArchDeviceImpl *dev, void *ptr);
    void  (*memset)(NCArchDeviceImpl *dev, void *ptr, int val, size_t count);
    void  (*memcpy_htod)(NCArchDeviceImpl *dev, void *dst, const void *src, size_t n);
    void  (*memcpy_dtoh)(NCArchDeviceImpl *dev, void *dst, const void *src, size_t n);

    /* Kernel management */
    int   (*load_kernel)(NCArchDeviceImpl *dev, const char *name,
                          const void *code, size_t code_size);
    void  (*launch)(NCArchDeviceImpl *dev, const char *kernel_name,
                    int grid_x, int grid_y, int grid_z,
                    int block_x, int block_y, int block_z,
                    int shared_mem, void **args);

    /* Synchronization */
    void  (*synchronize)(NCArchDeviceImpl *dev);

} NCArchOps;

/* Get architecture name string */
static inline const char *nc_arch_name(NCArch arch) {
    switch (arch) {
        case NC_ARCH_CUDA:   return "CUDA";
        case NC_ARCH_HIP:    return "HIP/ROCm";
        case NC_ARCH_METAL:  return "Metal";
        case NC_ARCH_SYCL:   return "SYCL/oneAPI";
        case NC_ARCH_VULKAN: return "Vulkan Compute";
        default:             return "Unknown";
    }
}

/* Portable div/mod utilities */
static inline uint32_t nc_ceil_div(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
}

static inline uint32_t nc_round_up(uint32_t x, uint32_t alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}

#ifdef __cplusplus
}
#endif

#endif /* LIBNC_ARCH_H */
