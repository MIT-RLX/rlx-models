/* tsac_normal_decode.c — Normal TXC decode using Transformer + range coder */
#include "dac_model.h"
#include "range_coder.h"
#include "tsac_transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TF_VOCAB 1024  /* codebook size */

/* Softmax: convert logits to probabilities */
static void softmax(float *probs, const float *logits, int n) {
    float max = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max) max = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { probs[i] = expf(logits[i] - max); sum += probs[i]; }
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

/* Build cumulative frequency table from probabilities.
 * cum_freq[n+1] = cumulative scaled probability. Returns total. */
static uint32_t build_cum_freq(uint32_t *cum_freq, const float *probs, int n) {
    cum_freq[0] = 0;
    for (int i = 0; i < n; i++) {
        uint32_t scaled = (uint32_t)(probs[i] * RC_MAX_FREQ + 0.5f);
        if (scaled < 1) scaled = 1;  /* avoid zero probability */
        if (cum_freq[i] + scaled > RC_MAX_FREQ)
            cum_freq[i + 1] = RC_MAX_FREQ;
        else
            cum_freq[i + 1] = cum_freq[i] + scaled;
    }
    /* Adjust last to exactly RC_MAX_FREQ */
    if (cum_freq[n] != RC_MAX_FREQ) cum_freq[n] = RC_MAX_FREQ;
    return cum_freq[n];
}

/* Decode one frame of codebook indices using Transformer + range coder.
 * encoded: compressed payload, len: payload length
 * frame_n: which frame to decode (0-indexed)
 * n_cb: number of codebooks per frame
 * indices_out: [n_cb] output array for this frame
 * 
 * The transformer takes previously-decoded indices (shape [total_frames, n_cb])
 * and predicts the next frame's codebook indices as probability distributions.
 * Each codebook entry (0..1023) is decoded via range arithmetic decoding
 * using the Transformer-predicted cumulative frequency table. */
static int decode_one_frame(RangeCoder *rc, TSACTransformer *tf,
                             int frame_n, int total_frames, int n_cb,
                             int *all_indices) {
    int D = TF_D_MODEL;

    /* Build position_ids: [frame_n] */
    int pos_ids[1] = {frame_n};

    /* For each codebook quantizer */
    for (int cb = 0; cb < n_cb && cb < 12; cb++) {
        /* Run Transformer forward pass to get logits */
        float logits[TF_MAX_SEQ * TF_D_MODEL];
        int ret = tsac_transformer_forward(tf, NULL, pos_ids, 1, logits);
        if (ret != TSAC_OK) return ret;

        /* We only need the last position's logits, projected through g */
        float *pos_logits = logits;  /* [512], actually just first element from g projection */

        /* Convert logits to cumulative frequencies */
        float probs[TF_VOCAB];
        softmax(probs, pos_logits, TF_VOCAB);
        
        uint32_t cum_freq[TF_VOCAB + 1];
        uint32_t total = build_cum_freq(cum_freq, probs, TF_VOCAB);

        /* Range decode the codebook entry */
        int sym = rc_decode_cumul(rc, cum_freq, TF_VOCAB, total);
        if (sym < 0) return TSAC_ERR_CODEC;

        /* Store decoded index */
        all_indices[frame_n * n_cb + cb] = sym;

        /* Feed back as input for next codebook (autoregressive) */
        /* The Transformer input_ids would need to be updated here.
         * For now, we use position_ids only (no token embeddings). */
    }
    return TSAC_OK;
}

/* Full Normal TXC decode: compressed payload → codebook indices */
int tsac_normal_decode(const uint8_t *compressed, size_t comp_len,
                        TSACTransformer *tf,
                        int n_frames, int n_cb,
                        int **indices_out, int *out_frames) {
    if (!compressed || !tf || !indices_out || !out_frames) return TSAC_ERR_PARAM;
    if (n_frames <= 0 || n_frames > TF_MAX_SEQ) return TSAC_ERR_PARAM;
    if (n_cb <= 0 || n_cb > 12) return TSAC_ERR_PARAM;

    /* Allocate index array */
    int *all_idx = (int *)calloc((size_t)n_frames * n_cb, sizeof(int));
    if (!all_idx) return TSAC_ERR_MEMORY;

    /* Initialize range decoder */
    RangeCoder rc;
    int ret = rc_decoder_init(&rc, compressed, comp_len);
    if (ret != TSAC_OK) { free(all_idx); return ret; }

    /* Decode frame by frame, codebook by codebook */
    for (int f = 0; f < n_frames; f++) {
        ret = decode_one_frame(&rc, tf, f, n_frames, n_cb, all_idx);
        if (ret != TSAC_OK) { free(all_idx); return ret; }
    }

    *indices_out = all_idx;
    *out_frames = n_frames;
    return TSAC_OK;
}
