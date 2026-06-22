#ifndef RANGE_CODER_H
#define RANGE_CODER_H

/* Range coder: get_freq (15-bit adaptive) + cumul symbol decode. RE from arith.c. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Range coder state.
 *  Stores the current decoding window (low, range) and buffer position. */

#define RC_MIN_VALUE      0x0000FF00U
#define RC_INIT_RANGE     0xFFFFFFFFU
#define RC_INIT_CODE_BYTES 4
#define RC_MAX_FREQ       32767  /* 15-bit frequency precision */

typedef struct {
    uint32_t        low;
    uint32_t        range;
    const uint8_t  *buf;
    size_t          buf_pos;
    size_t          buf_len;
} RangeCoder;

/** Initialize a range coder from a byte buffer.
 *  Reads RC_INIT_CODE_BYTES (4) bytes to seed the low value.
 *  Returns 0 on success, negative on error. */
int  rc_decoder_init(RangeCoder *rc, const uint8_t *buf, size_t len);

/** Decode a single bit with 50/50 equal probability.
 *  Used in fast TXC mode as a fallback (confirmed dead code in original). */
int  rc_decoder_direct_bit(RangeCoder *rc);

/** Decode a single bit with adaptive frequency.
 *  freq is a 15-bit probability value (1..32767).
 *  Used in normal TXC mode with binary search decoder. */
int  rc_decoder_get_freq(RangeCoder *rc, uint32_t freq);

/** Decode a symbol from a cumulative frequency table.
 *  cum_freq: array of size n_syms+1 (cum_freq[0]=0, cum_freq[n_syms]=total)
 *  n_syms: number of symbols
 *  total: total frequency (cum_freq[n_syms])
 *  Returns symbol index 0..n_syms-1, or -1 on error. */
int  rc_decode_cumul(RangeCoder *rc, const uint32_t *cum_freq, int n_syms, uint32_t total);

/** Decode n_bits using equal probability (direct bits).
 *  Convenience for decoding fixed-width values.
 *  Returns decoded value, or -1 on error. */
int  rc_decode_bits(RangeCoder *rc, int n_bits);

/** Get remaining bytes in the range coder buffer. */
static inline size_t rc_remaining(RangeCoder *rc) {
    return rc->buf_len - rc->buf_pos;
}

#ifdef __cplusplus
}
#endif

#endif /* RANGE_CODER_H */
