/*
 * range_coder.c — Arithmetic range coder implementation.
 *
 * Reverse-engineered from original tsac binary's arith.c.
 * The original uses get_freq (adaptive 15-bit probability) for
 * codebook index decoding, NOT equal-probability direct bits.
 *
 * Extended with cumulative frequency symbol decode (rc_decode_cumul)
 * for normal TXC mode with adaptive probability tables.
 */

#include "range_coder.h"

/* Ensure range is above minimum threshold after bit consumption.
 * If buffer is exhausted, range left-shifts alone (low fills with zeros). */
static void rc_normalize(RangeCoder *rc)
{
    while (rc->range <= RC_MIN_VALUE) {
        rc->range <<= 8;
        rc->low   <<= 8;
        if (rc->buf_pos < rc->buf_len)
            rc->low |= rc->buf[rc->buf_pos++];
        /* If buffer exhausted, low left-shifts with zero fill - 
         * range continues to grow, eventually exceeding RC_MIN_VALUE */
    }
}

int rc_decoder_init(RangeCoder *rc, const uint8_t *buf, size_t len)
{
    if (!rc || !buf) return -1;
    if (len < RC_INIT_CODE_BYTES) return -2;

    rc->low     = 0;
    rc->range   = RC_INIT_RANGE;
    rc->buf     = buf;
    rc->buf_pos = 0;
    rc->buf_len = len;

    for (int i = 0; i < RC_INIT_CODE_BYTES; i++)
        rc->low = (rc->low << 8) | rc->buf[rc->buf_pos++];

    return 0;
}

int rc_decoder_direct_bit(RangeCoder *rc)
{
    rc_normalize(rc);
    uint32_t r0 = rc->range >> 1;
    if (rc->low >= r0) { rc->low -= r0; rc->range -= r0; return 1; }
    rc->range = r0;
    return 0;
}

int rc_decoder_get_freq(RangeCoder *rc, uint32_t freq)
{
    rc_normalize(rc);
    uint32_t r0 = ((uint64_t)rc->range * freq) >> 15;
    if (r0 < 1) r0 = 1;
    if (r0 >= rc->range) r0 = rc->range - 1;

    if (rc->low >= r0) {
        rc->low   -= r0;
        rc->range -= r0;
        return 1;
    }
    rc->range = r0;
    return 0;
}

int rc_decode_cumul(RangeCoder *rc, const uint32_t *cum_freq, int n_syms, uint32_t total)
{
    if (!rc || !cum_freq || n_syms < 1 || total < 1) return -1;

    rc_normalize(rc);

    /* Scale: split range by total frequency */
    uint32_t rng_per_freq = rc->range / total;
    if (rng_per_freq < 1) rng_per_freq = 1;

    /* Find which cumulative frequency slot contains low */
    uint32_t value = rc->low / rng_per_freq;
    if (value >= total) value = total - 1;

    /* Binary search for symbol */
    int lo = 0, hi = n_syms;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cum_freq[mid] <= value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int sym = lo - 1;
    if (sym < 0) sym = 0;
    if (sym >= n_syms) sym = n_syms - 1;

    /* Update range to the symbol's interval */
    uint32_t sym_lo = (uint64_t)cum_freq[sym] * rng_per_freq;
    uint32_t sym_hi = (uint64_t)(sym + 1 < n_syms ? cum_freq[sym + 1] : total) * rng_per_freq;

    rc->low   -= sym_lo;
    rc->range  = sym_hi - sym_lo;
    if (rc->range < 1) rc->range = 1;

    rc_normalize(rc);
    return sym;
}

int rc_decode_bits(RangeCoder *rc, int n_bits)
{
    if (!rc || n_bits < 0 || n_bits > 31) return -1;

    int value = 0;
    for (int i = 0; i < n_bits; i++) {
        value = (value << 1) | rc_decoder_direct_bit(rc);
    }
    return value;
}
