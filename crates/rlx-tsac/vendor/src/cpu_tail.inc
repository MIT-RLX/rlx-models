    int current_C = m0_Co;
    int cur_frames = ctx_frames;

#include "cpu_blocks.inc"
    /* model.5 snake */
    DACTensor *m5_alpha = tf(ts, nt, "decoder.model.5.alpha");
    if (m5_alpha && current_C == 96) {
        const float *alpha = (const float *)m5_alpha->data;
        int alpha_C = m5_alpha->dims[0];
        snake_parallel(current, current, alpha,
                       current_C * cur_frames, alpha_C, n_threads);
    }

    /* model.6 output conv1d */
    DACTensor *m6_wv = tf(ts, nt, "decoder.model.6.weight_v");
    DACTensor *m6_wg = tf(ts, nt, "decoder.model.6.weight_g");
    DACTensor *m6_b  = tf(ts, nt, "decoder.model.6.bias");

    int m6_Ci, m6_K, m6_Co;
    float *m6_w = dequant_weights(m6_wv, m6_wg, m6_b, &m6_Ci, &m6_K, &m6_Co, NULL);
    const float *m6_b_data = m6_b ? (const float *)m6_b->data : NULL;

    DBG("[DEBUG] m6: Ci=%d, K=%d, Co=%d, cur_frames=%d\n", m6_Ci, m6_K, m6_Co, cur_frames);

    float *output = (float *)malloc((size_t)m6_Co * cur_frames * sizeof(float));

    if (m6_w && output) {
        ops.conv1d(output, current, m6_w, m6_b_data,
                   cur_frames, m6_K, m6_Ci, m6_Co);

        DBG("[DEBUG] After m6 conv1d NaN: %d/%d\n", count_nan(output, m6_Co*cur_frames), m6_Co*cur_frames);
        DUMP_ACT(output, m6_Co*cur_frames, "m6_pre_tanh");

        /* Compute how many output samples to discard from the beginning
         * due to context padding. code_offset is the start of the context
         * window, batch_start is the start of the actual batch data. */
        int discard_frames = batch_start - code_offset;
        if (discard_frames < 0) discard_frames = 0;
        int discard_samples = discard_frames * total_upscale;

        int out_start = batch_start * total_upscale;
        int out_count = batch_frames * total_upscale;

        /* Bounds check: ensure we don't read past output buffer */
        int max_s = cur_frames - discard_samples;
        if (max_s > out_count) max_s = out_count;
        if (max_s < 0) max_s = 0;

        int clipped = 0;
        /* DEBUG: print output value range before tanh */
        float max_abs_pre = 0;
        for (int i = 0; i < m6_Co * cur_frames; i++) {
            float a = fabsf(output[i]);
            if (a > max_abs_pre) max_abs_pre = a;
        }
        DBG("[DEBUG] m6 output pre-tanh: max_abs=%.2f\n", max_abs_pre);
        for (int c = 0; c < ch && c < m6_Co; c++) {
            for (int s = 0; s < max_s; s++) {
                int final_s = out_start + s;
                if (final_s >= n_samples) break;
                float val = tanhf(output[c * cur_frames + discard_samples + s]);
                if (val > 1.0f) { val = 1.0f; clipped++; }
                if (val < -1.0f) { val = -1.0f; clipped++; }
                pcm[c * n_samples + final_s] = val;
            }
        }
        DBG("[DEBUG] Clipped samples: %d/%d\n", clipped, max_s * ch);
    }

    free(current);
    free(m6_w);
    free(output);
    return TSAC_OK;
}

/* CPU decoder entry point: decode TXC indices to float32 PCM.
 * Handles batching, threading, and the full DAC decode graph. */
int cpu_decoder_run(DACTensor *ts, int nt,
                     const int *codes, int n_frames, int n_cb,
                     float *pcm, int n_samples, int ch,
                     int n_threads)
{
    (void)n_samples;
    int total_batches = (n_frames + BATCH_FRAMES - 1) / BATCH_FRAMES;
    fprintf(stderr, "[cpu_dac] SIMD=%s frames=%d cb=%d ch=%d threads=%d\n",
            cpu_simd_name(), n_frames, n_cb, ch, n_threads);
    fprintf(stderr, "batch_size=%d bs=%d n_cb=%d block_len=%d\n",
            BATCH_FRAMES, 1, n_cb, total_batches);

    CPUOps ops = get_ops();

    int total_upscale = 1;
    for (int block = 1; block <= 4; block++) {
        char wv_name[128];
        snprintf(wv_name, sizeof(wv_name), "decoder.model.%d.block.1.weight_v", block);
        DACTensor *wv = tf(ts, nt, wv_name);
        if (wv && wv->ndims >= 2) {
            int K = wv->dims[1];
            total_upscale *= (K / 2);
        }
    }

    int processed_frames = 0;
    int batch_idx = 0;

    for (int batch_start = 0; batch_start < n_frames; batch_start += BATCH_FRAMES) {
        int batch_end = batch_start + BATCH_FRAMES;
        if (batch_end > n_frames) batch_end = n_frames;
        int batch_frames = batch_end - batch_start;
        batch_idx++;

        int ctx_start = batch_start - CONTEXT_PAD;
        if (ctx_start < 0) ctx_start = 0;
        int ctx_end = batch_end + CONTEXT_PAD;
        if (ctx_end > n_frames) ctx_end = n_frames;
        int ctx_frames = ctx_end - ctx_start;

        int ret = decode_batch(ts, nt, codes, n_cb, ctx_start,
                               ctx_frames, n_threads,
                               pcm, n_samples, ch,
                               batch_start, batch_frames, total_upscale, ops);
        if (ret != TSAC_OK) return ret;

        processed_frames += batch_frames;
        int pct = (int)((float)batch_idx / (float)total_batches * 100.0f);
        if (pct > 100) pct = 100;
        fprintf(stderr, "%3u%%\r", pct);
        fflush(stderr);
    }
    fprintf(stderr, "\n");

    return TSAC_OK;
}

/* ================================================================ */
/*  CPU Encoder                                                      */
/* ================================================================ */

#include "cpu_encoder.inc"
