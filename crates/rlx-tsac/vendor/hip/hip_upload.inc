        b->cb_dim = 1024;
        b->cb_entries = 8;

        size_t cb_total = (size_t)total_entries * sizeof(float);
        hipMalloc(&b->d_cb_data, cb_total);
        hipMalloc(&b->d_cb_offsets, 13 * sizeof(int));

        size_t copied = 0;
        for (int cb = 0; cb < 12; cb++) {
            char cb_name[160];
            snprintf(cb_name, sizeof(cb_name), "quantizer.quantizers.%d.codebook.weight", cb);
            DACTensor *cb_t = F(ts, nt, cb_name);
            if (!cb_t) { b->cb_offsets[cb] = (int)copied / (int)sizeof(float); continue; }
            size_t sz = (size_t)cb_t->dims[0] * cb_t->dims[1] * sizeof(float);
            hipMemcpy((char *)b->d_cb_data + copied, cb_t->data, sz, hipMemcpyHostToDevice);
            b->cb_offsets[cb] = (int)(copied / sizeof(float));
            copied += sz;
        }
        b->cb_offsets[12] = (int)(copied / sizeof(float));
        hipMemcpy(b->d_cb_offsets, b->cb_offsets, 13 * sizeof(int), hipMemcpyHostToDevice);
    }

    b->weights_uploaded = 1;
    fprintf(stderr, "[hip] uploaded %d weights + %d snake alphas + codebooks\n",
            n_wl, (int)(sizeof(snake_names)/sizeof(snake_names[0])));

    return TSAC_OK;
}

/* ================================================================ */
/*  Main decoder run function (ported from CUDA backend)             */
/* ================================================================ */

extern "C" int dac_decoder_run(DACTensor *ts, int nt,
    const int *codes, int n_frames, int n_cb,
    float *pcm, int n_samples, int ch)
{
    fprintf(stderr, "[hip] dac_decoder_run: nt=%d frames=%d cb=%d samples=%d ch=%d\n",
            nt, n_frames, n_cb, n_samples, ch);

    /* Static backend storage for weight caching across calls */
    static HipBackend *b = NULL;
    if (!b) {
        b = (HipBackend *)calloc(1, sizeof(HipBackend));
        if (!b) return TSAC_ERR_MEMORY;
        hipSetDevice(0);
        hipStreamCreate(&b->stream);
        b->initialized = 1;
    }

    hipStream_t s = b->stream;

    /* Lazy weight upload on first call */
    if (!b->weights_uploaded) {
        int ret = hip_upload_weights(b, ts, nt);
        if (ret != TSAC_OK) return ret;
        /* Weight sanity check */
        GpuWeight *w0 = hip_find_weight(b, "decoder.model.0");
        if (w0) {
            float test[4];
            hipMemcpy(test, w0->d_data, 4*sizeof(float), hipMemcpyDeviceToHost);
            fprintf(stderr, "[hip] w0[0..3]=%f %f %f %f\n", test[0], test[1], test[2], test[3]);
        } else {
            fprintf(stderr, "[hip] WARNING: decoder.model.0 weight not found!\n");
        }
    }

    if (n_frames < 1) return TSAC_OK;

    int rvq_dim = 1024;

    /* Upload codebook indices */
    size_t codes_bytes = (size_t)n_frames * n_cb * sizeof(int);
    hipMalloc(&b->d_codes, codes_bytes);
    hipMemcpy(b->d_codes, codes, codes_bytes, hipMemcpyHostToDevice);

    /* RVQ lookup -> features [1024, n_frames] */
    size_t feat_bytes = (size_t)rvq_dim * n_frames * sizeof(float);
    float *d_feat = hip_backend_get_buf(b, 0, feat_bytes);
    if (!d_feat) return TSAC_ERR_MEMORY;
#include "hip_decode.inc"
    int cur_T = nf;

    for (int blk = 4; blk >= 1; blk--) {
        int idx = 4 - blk;

        /* Snake before conv */
        char sname[128];
        snprintf(sname, sizeof(sname), "encoder.model.%d.block.0.alpha", blk);
        GpuWeight *sa = hip_find_enc_weight(b, sname);
        if (sa) {
            snake_k<<<(cur_C * cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
                d_cur, d_cur, sa->d_data, cur_C * cur_T, sa->Ci);
        }

        /* Conv1d (no downsampling in DAC encoder, stride=1) */
        char wname[128];
        snprintf(wname, sizeof(wname), "encoder.model.%d.block.1", blk);
        GpuWeight *cw = hip_find_enc_weight(b, wname);
        int next_C = enc_c_out[idx];
        int next_T = cur_T;

        float *d_next = hip_backend_get_buf(b, idx + 1, next_C * next_T * sizeof(float));
        if (!d_next) return TSAC_ERR_MEMORY;
        hipMemsetAsync(d_next, 0, next_C * next_T * sizeof(float), s);

        if (cw) {
            conv1d_k<<<dim3(next_C, (next_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                d_next, d_cur, cw->d_data, NULL,
                next_T, cw->K, cw->Ci, cw->Co, 1);
            char bname[128];
            snprintf(bname, sizeof(bname), "encoder.model.%d.block.1.bias", blk);
            GpuWeight *cb = hip_find_enc_weight(b, bname);
            if (cb) {
                add_bias_k<<<dim3(next_T, (next_C+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                    d_next, cb->d_data, next_T, next_C);
            }
        }

        d_cur = d_next;
        cur_C = next_C;
        cur_T = next_T;

        /* Inner blocks (snake->conv1d K=7->snake->conv1d K=1) x 3 */
        for (int inner = 2; inner <= 4; inner++) {
            /* Snake */
            char isname[160];
            snprintf(isname, sizeof(isname), "encoder.model.%d.block.%d.block.0.alpha", blk, inner);
            GpuWeight *gis = hip_find_enc_weight(b, isname);
            if (gis) {
                snake_k<<<(cur_C * cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
                    d_cur, d_cur, gis->d_data, cur_C * cur_T, gis->Ci);
            }

            /* Conv1d K=7 */
            char iwname[160], ibname[160];
            snprintf(iwname, sizeof(iwname), "encoder.model.%d.block.%d.block.1", blk, inner);
            snprintf(ibname, sizeof(ibname), "encoder.model.%d.block.%d.block.1.bias", blk, inner);
            GpuWeight *giw = hip_find_enc_weight(b, iwname);
            GpuWeight *gib = hip_find_enc_weight(b, ibname);
            if (giw) {
                float *d_tmp = hip_backend_get_buf(b, 6, cur_C * cur_T * sizeof(float));
                if (!d_tmp) return TSAC_ERR_MEMORY;
                hipMemsetAsync(d_tmp, 0, cur_C * cur_T * sizeof(float), s);
                conv1d_k<<<dim3(cur_C, (cur_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                    d_tmp, d_cur, giw->d_data, NULL,
                    cur_T, giw->K, giw->Ci, giw->Co, 1);
                if (gib) {
                    add_bias_k<<<dim3(cur_T, (cur_C+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                        d_tmp, gib->d_data, cur_T, cur_C);
                }
