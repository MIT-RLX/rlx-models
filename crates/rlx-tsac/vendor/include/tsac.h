#ifndef TSAC_H
#define TSAC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version string */
#define TSAC_NG_VERSION "0.1.0"

/* .txc container header (reconstructed from binary analysis) */
/* Magic: FBAZ (0x5A414246) */
typedef struct {
    char     magic[4];      /* "FBAZ" */
    uint16_t version;       /* container format version */
    uint16_t n_codebooks;   /* number of RVQ codebooks */
    uint32_t block_len;     /* samples per encoded block */
    uint32_t n_blocks;      /* total blocks */
    uint32_t sample_rate;   /* 44100 typical */
    uint32_t flags;         /* stereo=1, mono=0 */
    uint32_t data_offset;   /* offset to encoded data */
} TSCHeader;

/* Opaque TSAC codec handle */
typedef struct TSACContext TSACContext;

/* Channel mode */
typedef enum {
    TSAC_CODEC_MONO   = 0,
    TSAC_CODEC_STEREO = 1,
} TSACChannelMode;

/* Backend type */
typedef enum {
    TSAC_BACKEND_CPU    = 0,
    TSAC_BACKEND_CUDA   = 1,
    TSAC_BACKEND_HIP    = 2,
    TSAC_BACKEND_VULKAN = 3,
    TSAC_BACKEND_LLVM   = 4,
} TSACBackend;

/* Return codes */
#define TSAC_OK             0
#define TSAC_ERR_MEMORY    -1
#define TSAC_ERR_FILE      -2
#define TSAC_ERR_FORMAT    -3
#define TSAC_ERR_MODEL     -4
#define TSAC_ERR_CODEC     -5
#define TSAC_ERR_PARAM     -6
#define TSAC_ERR_BACKEND   -7
#define TSAC_ERR_INTERNAL  -8

/*
 * Create/destroy codec context.
 *
 * backend:  TSAC_BACKEND_CPU, TSAC_BACKEND_CUDA, or TSAC_BACKEND_HIP
 * n_threads: number of worker threads (0 = auto)
 * model_path: path to the .bin model file
 *
 * Returns NULL on failure.
 */
TSACContext *tsac_init(TSACBackend backend, int n_threads, const char *model_path);

/* Free all resources associated with ctx. Safe to call with NULL. */
void tsac_free(TSACContext *ctx);

/*
 * Compress raw PCM samples to .txc format.
 *
 * pcm:     interleaved float samples in [-1.0, 1.0]
 * n_samples: number of samples per channel
 * channels: 1 (mono) or 2 (stereo)
 * out_data:  on success, set to newly allocated buffer with .txc data
 * out_size:  on success, size of output data
 * n_codebooks: number of RVQ codebooks (1-12, higher = better quality)
 *
 * Returns TSAC_OK on success, negative on error.
 * Caller must free out_data with tsac_free_buffer().
 */
int tsac_compress(TSACContext *ctx,
                  const float *pcm, int n_samples, int channels,
                  uint8_t **out_data, size_t *out_size,
                  int n_codebooks);

/*
 * Decompress .txc data to raw PCM.
 *
 * txc_data:   pointer to .txc container data
 * txc_size:   size of txc_data
 * out_pcm:    on success, set to newly allocated float buffer
 * out_samples: on success, set to number of samples per channel
 * out_channels: on success, set to number of channels
 *
 * Returns TSAC_OK on success, negative on error.
 * Caller must free out_pcm with tsac_free_buffer().
 */
int tsac_decompress(TSACContext *ctx,
                    const uint8_t *txc_data, size_t txc_size,
                    float **out_pcm, int *out_samples, int *out_channels);

/*
 * Compress a WAV file to .txc file.
 */
int tsac_compress_file(TSACContext *ctx, const char *in_wav, const char *out_txc, int n_codebooks);

/*
 * Decompress a .txc file to WAV file.
 */
int tsac_decompress_file(TSACContext *ctx, const char *in_txc, const char *out_wav);

/* Free buffers allocated by the library. */
void tsac_free_buffer(void *ptr);

/* Return version string. */
const char *tsac_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TSAC_H */
