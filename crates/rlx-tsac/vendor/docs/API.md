# tsac-ng API Reference

## Core API (`include/tsac.h`)

### Types

```c
typedef enum {
    TSAC_BACKEND_CPU    = 0,
    TSAC_BACKEND_CUDA   = 1,
    TSAC_BACKEND_HIP    = 2,
    TSAC_BACKEND_VULKAN = 3,
    TSAC_BACKEND_LLVM   = 4,
} TSACBackend;

typedef struct TSACContext TSACContext;  // opaque
```

### Return Codes

```c
#define TSAC_OK             0
#define TSAC_ERR_MEMORY    -1
#define TSAC_ERR_FILE      -2
#define TSAC_ERR_FORMAT    -3
#define TSAC_ERR_MODEL     -4
#define TSAC_ERR_CODEC     -5
#define TSAC_ERR_PARAM     -6
#define TSAC_ERR_BACKEND   -7
#define TSAC_ERR_INTERNAL  -8
```

### Lifecycle

```c
TSACContext *tsac_init(TSACBackend backend, int n_threads, const char *model_path);
void         tsac_free(TSACContext *ctx);
const char  *tsac_version(void);
```

### Compression

```c
int tsac_compress(TSACContext *ctx,
                  const float *pcm, int n_samples, int channels,
                  uint8_t **out_data, size_t *out_size,
                  int n_codebooks);

int tsac_compress_file(TSACContext *ctx, const char *in_wav,
                       const char *out_txc, int n_codebooks);
```

### Decompression

```c
int tsac_decompress(TSACContext *ctx,
                    const uint8_t *txc_data, size_t txc_size,
                    float **out_pcm, int *out_samples, int *out_channels);

int tsac_decompress_file(TSACContext *ctx, const char *in_txc,
                         const char *out_wav);
```

### Memory

```c
void tsac_free_buffer(void *ptr);
```

## Backend API (`src/tsac_codec.h`)

Each backend implements:

```c
int  tsac_<backend>_init(void **priv);
int  tsac_<backend>_encode(void *priv, void *model, ...);
int  tsac_<backend>_decode(void *priv, void *model, ...);
void tsac_<backend>_shutdown(void *priv);
```

Where `<backend>` is one of: `cuda`, `hip`, `vk`, `llvm`.

Backends receive:
- `priv`: Backend-specific state (GPU context, buffers, etc.)
- `model`: `DACModel*` pointer (tensor array)

## Model Format API (`src/model_loader.h`)

```c
typedef struct {
    char     name[128];
    int      ndims;
    int      dims[8];
    uint8_t *data;        // raw tensor data
    int      data_size;   // bytes
    int      elem_size;   // 1=uint8(BF8), 4=float32
    void    *dev;         // GPU device pointer (optional)
    float   *dev_f32;     // dequantized GPU pointer (optional)
} DACTensor;

typedef struct {
    DACTensor *tensors;
    int        n_tensors;
} DACModel;

int  model_loader_load(const char *path, DACModel *model);
void model_loader_free(DACModel *model);
```

## .txc Container API (`src/txc_format.h`)

```c
typedef struct {
    char     magic[4];       // "FBAZ"
    uint16_t version;
    uint16_t n_codebooks;
    uint32_t block_len;
    uint32_t n_blocks;
    uint32_t sample_rate;
    uint32_t flags;          // bit0 = stereo
    uint32_t data_offset;
} TSCHeader;

void txc_header_init(TSCHeader *hdr, int stereo, int n_codebooks, int sample_rate);
int  txc_write(const TSCHeader *hdr, const int *codebook_indices, int n_frames,
               uint8_t **out_data, size_t *out_size);
int  txc_read(const uint8_t *data, size_t data_size, TSCHeader *hdr,
              int **codebook_indices, int *n_frames);
```

## Usage Example

```c
#include "tsac.h"

int main() {
    // Initialize with CUDA backend
    TSACContext *ctx = tsac_init(TSAC_BACKEND_CUDA, 1, "/usr/share/tsac");
    if (!ctx) return 1;

    // Read .txc file
    FILE *f = fopen("audio.txc", "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *txc = malloc(sz);
    fread(txc, 1, sz, f); fclose(f);

    // Decompress
    float *pcm = NULL;
    int samples = 0, channels = 0;
    int ret = tsac_decompress(ctx, txc, sz, &pcm, &samples, &channels);
    if (ret == TSAC_OK) {
        // pcm is interleaved float32, [-1.0, 1.0]
        // samples per channel × channels
    }

    tsac_free_buffer(pcm);
    tsac_free_buffer(txc);
    tsac_free(ctx);
    return 0;
}
```
