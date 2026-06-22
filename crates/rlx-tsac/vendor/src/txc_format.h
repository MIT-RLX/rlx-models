#ifndef TXC_FORMAT_H
#define TXC_FORMAT_H

#include "tsac.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize a TSCHeader with defaults */
void txc_header_init(TSCHeader *hdr, int stereo, int n_codebooks, int sample_rate);

/* Write .txc container from codebook indices */
int txc_write(const TSCHeader *hdr,
              const int *codebook_indices, int n_frames,
              uint8_t **out_data, size_t *out_size);

/* Parse .txc container to codebook indices */
int txc_read(const uint8_t *data, size_t data_size,
             TSCHeader *hdr,
             int **codebook_indices, int *n_frames);

#ifdef __cplusplus
}
#endif

#endif /* TXC_FORMAT_H */
