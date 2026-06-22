/*
 * hook_cumoduleload.c — LD_PRELOAD interceptor for cuModuleLoadData.
 *
 * Intercepts cuModuleLoadData calls and replaces the embedded fatbin
 * from libnc_cuda.so with our custom fatbin data, while keeping all
 * other original library code (struct layouts, cuBLASLt integration,
 * memory management) unchanged.
 *
 * Build:
 *   gcc -shared -fPIC -o hook_cumoduleload.so hook_cumoduleload.c -ldl
 *
 * Use:
 *   LD_PRELOAD=./hook_cumoduleload.so tsac --cuda ...
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* Typedef CUDA types to avoid depending on cuda.h */
typedef int CUresult;
typedef void *CUmodule;
#define CUDA_SUCCESS 0

/* Our custom fatbin data — defined in nc_cuda_ops_fatbin_data.o */
extern const unsigned char nc_cuda_ops_fatbin[];
extern const unsigned int  nc_cuda_ops_fatbin_len;

/* Original cuModuleLoadData */
typedef CUresult (*orig_cuModuleLoadData_t)(CUmodule *module, const void *image);
static orig_cuModuleLoadData_t orig_cuModuleLoadData = NULL;

CUresult cuModuleLoadData(CUmodule *module, const void *image)
{
    if (!orig_cuModuleLoadData) {
        orig_cuModuleLoadData = (orig_cuModuleLoadData_t)dlsym(RTLD_NEXT, "cuModuleLoadData");
        fprintf(stderr, "[hook] cuModuleLoadData interceptor initialized\n");
    }

    /* Check if this is the nc_cuda_ops_fatbin being loaded */
    /* The original fatbin has a specific magic — check if this is it */
    if (image != NULL && nc_cuda_ops_fatbin != NULL && nc_cuda_ops_fatbin_len > 0) {
        const unsigned char *img = (const unsigned char *)image;
        /* Check for our custom fatbin magic: 0xba55ed50 */
        uint32_t magic;
        memcpy(&magic, img, 4);
        if (magic == 0xba55ed50) {
            fprintf(stderr, "[hook] Intercepted cuModuleLoadData: replacing fatbin "
                    "(original=%p, replacement=%p, size=%u)\n",
                    image, nc_cuda_ops_fatbin, nc_cuda_ops_fatbin_len);
            return orig_cuModuleLoadData(module, nc_cuda_ops_fatbin);
        }
    }

    return orig_cuModuleLoadData(module, image);
}
