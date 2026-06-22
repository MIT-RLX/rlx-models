    hipMemsetAsync(d_feat, 0, feat_bytes, s);

    /* Use rvq_lookup_simple_k for each codebook */
    for (int cb = 0; cb < n_cb && cb < 12; cb++) {
        char cb_name[160];
        snprintf(cb_name, sizeof(cb_name), "quantizer.quantizers.%d.codebook.weight", cb);
        DACTensor *cb_t = F(ts, nt, cb_name);
        if (!cb_t) continue;

        int entries = cb_t->dims[0];
        int dim = cb_t->dims[1];

        float *d_cb = NULL;
        if (cb_t->dev_f32) {
            d_cb = cb_t->dev_f32;
        } else {
            hipMalloc(&cb_t->dev_f32, cb_t->data_size);
            hipMemcpyAsync(cb_t->dev_f32, cb_t->data, cb_t->data_size, hipMemcpyHostToDevice, s);
            d_cb = cb_t->dev_f32;
        }

        rvq_lookup_simple_k<<<(n_frames + BLK - 1) / BLK, BLK, 0, s>>>(
            d_feat, b->d_codes, d_cb, cb, entries, n_frames, n_cb, dim);
    }
    hipStreamSynchronize(s);

    /* model.0: Conv1d(1024->1536, K=7) */
    GpuWeight *w0 = hip_find_weight(b, "decoder.model.0");
    int T0 = n_frames;
    int C0 = w0 ? w0->Co : 1536;
    size_t b0_bytes = (size_t)C0 * T0 * sizeof(float);
    float *d_buf0 = hip_backend_get_buf(b, 1, b0_bytes);
    hipMemsetAsync(d_buf0, 0, b0_bytes, s);
    if (w0) {
        GpuWeight *b0_bias = hip_find_weight(b, "decoder.model.0.bias");
        conv1d_k<<<dim3(C0, (T0+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
            d_buf0, d_feat, w0->d_data, b0_bias ? b0_bias->d_data : NULL,
            T0, w0->K, w0->Ci, w0->Co, 1);
    }
    hipStreamSynchronize(s);

    float *d_cur = d_buf0;
    int cur_C = C0;
    int cur_T = T0;

    /* Blocks 1-4 */
    int c_out[4] = {768, 384, 192, 96};
    int convt_K[4] = {16, 16, 8, 4};

    for (int blk = 1; blk <= 4; blk++) {
        char wname[160], sname[160], bname[160];
        snprintf(wname, sizeof(wname), "decoder.model.%d.block.1", blk);
        snprintf(sname, sizeof(sname), "decoder.model.%d.block.0.alpha", blk);
        snprintf(bname, sizeof(bname), "decoder.model.%d.block.1.bias", blk);

        /* Snake before conv */
        GpuWeight *gs = hip_find_weight(b, sname);
        if (gs) {
            snake_k<<<(cur_C * cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
                d_cur, d_cur, gs->d_data, cur_C * cur_T, gs->Ci);
        }

        /* ConvTranspose1d - 2x upsampling */
        GpuWeight *gw = hip_find_weight(b, wname);
        int next_C = c_out[blk - 1];
        int next_T = cur_T * 2;
        int K = convt_K[blk - 1];

        size_t nb = (size_t)next_C * next_T * sizeof(float);
        float *d_next = hip_backend_get_buf(b, blk + 1, nb);
        hipMemsetAsync(d_next, 0, nb, s);

        if (gw) {
            /* Use convt_k with grid config: dim3(Ci, (Ti+BLK-1)/BLK) */
            convt_k<<<dim3(gw->Ci, (cur_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                d_next, d_cur, gw->d_data, cur_T, next_T, K, gw->Ci, gw->Co, 2);

            /* Add bias */
            GpuWeight *gbias = hip_find_weight(b, bname);
            if (gbias) {
                add_bias_k<<<dim3(next_C, (next_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                    d_next, gbias->d_data, next_T, next_C);
            }
        }

        d_cur = d_next;
        cur_C = next_C;
        cur_T = next_T;

        /* Inner blocks (3x per residual block) */
        for (int inner = 2; inner <= 4; inner++) {
            char iname[160], isname[160], ibname[160];
            snprintf(iname, sizeof(iname), "decoder.model.%d.block.%d.block.1", blk, inner);
            snprintf(isname, sizeof(isname), "decoder.model.%d.block.%d.block.0.alpha", blk, inner);
            snprintf(ibname, sizeof(ibname), "decoder.model.%d.block.%d.block.1.bias", blk, inner);

            GpuWeight *gis = hip_find_weight(b, isname);
            if (gis) {
                snake_k<<<(cur_C * cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
                    d_cur, d_cur, gis->d_data, cur_C * cur_T, gis->Ci);
            }

            /* Use temp buffer for conv1d */
            GpuWeight *giw = hip_find_weight(b, iname);
            GpuWeight *gbias = hip_find_weight(b, ibname);
            if (giw) {
                float *d_tmp = hip_backend_get_buf(b, 6, (size_t)cur_C * cur_T * sizeof(float));
                hipMemsetAsync(d_tmp, 0, (size_t)cur_C * cur_T * sizeof(float), s);
                conv1d_k<<<dim3(cur_C, (cur_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                    d_tmp, d_cur, giw->d_data, gbias ? gbias->d_data : NULL,
                    cur_T, giw->K, giw->Ci, giw->Co, 1);
                d_cur = d_tmp;
            }
        }
        hipStreamSynchronize(s);
    }

    /* model.5: Snake(96) */
    GpuWeight *gm5 = hip_find_weight(b, "decoder.model.5.alpha");
    if (gm5) {
        snake_k<<<(cur_C * cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
            d_cur, d_cur, gm5->d_data, cur_C * cur_T, gm5->Ci);
    }
    hipStreamSynchronize(s);

    /* model.6: Conv1d(96->2, K=7) */
    GpuWeight *gm6 = hip_find_weight(b, "decoder.model.6");
    if (gm6) {
        int out_C = gm6->Co;
        int out_T = cur_T;
        size_t ob = (size_t)out_C * out_T * sizeof(float);
        float *d_out = hip_backend_get_buf(b, 7, ob);
        hipMemsetAsync(d_out, 0, ob, s);
        GpuWeight *b6 = hip_find_weight(b, "decoder.model.6.bias");
        conv1d_k<<<dim3(out_C, (out_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
            d_out, d_cur, gm6->d_data, b6 ? b6->d_data : NULL,
            out_T, gm6->K, gm6->Ci, gm6->Co, 1);
        tanh_clip_k<<<(out_C * out_T + BLK - 1) / BLK, BLK, 0, s>>>(d_out, out_C * out_T);

        /* Copy to host PCM */
        int to_copy = out_T;
        if (to_copy > n_samples) to_copy = n_samples;
        if (to_copy > 0) {
            hipMemcpyAsync(pcm, d_out, (size_t)to_copy * sizeof(float), hipMemcpyDeviceToHost, s);
        }
    }

    hipStreamSynchronize(s);
    hipFree(b->d_codes);
    b->d_codes = NULL;

    fprintf(stderr, "[hip] completed: output samples=%d\n", cur_T);
    return TSAC_OK;
}

/* ================================================================ */
/*  Encoder run function (mirrors CUDA encoder)                      */
/* ================================================================ */

extern "C" int dac_encoder_run(DACTensor *ts, int nt,
    const float *pcm, int n_samples, int channels,
    int n_codebooks, int *codes, int *n_frames)
{
    fprintf(stderr, "[hip] dac_encoder_run: nt=%d samples=%d ch=%d cb=%d\n",
            nt, n_samples, channels, n_codebooks);

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
    }

    if (n_samples < 1) return TSAC_OK;

    /* Calculate frame count */
    int block_len = 512;  /* DAC default block length */
    int nf = (n_samples + block_len - 1) / block_len;
    if (nf < 1) nf = 1;
    *n_frames = nf;

    /* Upload PCM to GPU */
    size_t pcm_bytes = (size_t)n_samples * channels * sizeof(float);
    float *d_pcm;
    hipMalloc(&d_pcm, pcm_bytes);
    hipMemcpy(d_pcm, pcm, pcm_bytes, hipMemcpyHostToDevice);

    /* encoder.model.6: Conv1d(channels->96, K=7) */
    GpuWeight *e6 = hip_find_enc_weight(b, "encoder.model.6");
    float *d_cur = hip_backend_get_buf(b, 0, 96 * nf * sizeof(float));
    if (!d_cur) { hipFree(d_pcm); return TSAC_ERR_MEMORY; }
    hipMemsetAsync(d_cur, 0, 96 * nf * sizeof(float), s);
    if (e6) {
        GpuWeight *b6 = hip_find_enc_weight(b, "encoder.model.6.bias");
        conv1d_k<<<dim3(96, (nf+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
            d_cur, d_pcm, e6->d_data, b6 ? b6->d_data : NULL,
            nf, e6->K, e6->Ci, e6->Co, 1);
    }
    hipFree(d_pcm);

    /* encoder.model.5: Snake(96) */
    GpuWeight *e5 = hip_find_enc_weight(b, "encoder.model.5.alpha");
    if (e5) {
        snake_k<<<(96 * nf + BLK - 1) / BLK, BLK, 0, s>>>(
            d_cur, d_cur, e5->d_data, 96 * nf, 96);
    }

    /* Blocks 4->3->2->1 (reverse order: 96->192->384->768->1536) */
    int enc_c_out[4] = {192, 384, 768, 1536};
    int cur_C = 96;
