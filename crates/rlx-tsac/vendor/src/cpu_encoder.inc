/* CPU encoder entry point: encode float32 PCM to TXC indices.
 * NOTE: strided convs not yet implemented — partial functionality. */
int cpu_encoder_run(DACTensor *ts, int nt,
                    const float *pcm, int n_samples, int channels,
                    int n_codebooks, int block_len,
                    int **codebook_indices, int *n_frames) {
    int nf = (n_samples + block_len - 1) / block_len;
    if (nf < 1) nf = 1;
    *n_frames = nf;

    CPUOps ops = get_ops();
    int rvq_dim = 1024;

    int padded_samples = nf * block_len;
    float *padded = (float *)calloc((size_t)padded_samples * channels, sizeof(float));
    if (!padded) return TSAC_ERR_MEMORY;
    memcpy(padded, pcm, (size_t)n_samples * channels * sizeof(float));

    fprintf(stderr, "[cpu_enc] frames=%d cb=%d ch=%d blk=%d\n", nf, n_codebooks, channels, block_len);
    fprintf(stderr, "batch_size=%d bs=%d n_cb=%d block_len=%d\n", 1, 1, n_codebooks, nf);

    /* Encoder tensor naming: encoder.block.X (NOT encoder.model.X).
     * Internal block mapping differs from decoder:
     *   block.3.alpha = snake before conv  (we call block.0)
     *   block.4.*     = conv_transpose      (we call block.1)
     *   block.{0,1,2}.block.* = inner residual units (we call inner={2,3,4}) */
    DACTensor *e6_wv = tf(ts, nt, "encoder.block.0.weight_v");
    DACTensor *e6_wg = tf(ts, nt, "encoder.block.0.weight_g");
    DACTensor *e6_b  = tf(ts, nt, "encoder.block.0.bias");
    int e6_Ci, e6_K, e6_Co;
    float *e6_w = dequant_weights(e6_wv, e6_wg, e6_b, &e6_Ci, &e6_K, &e6_Co, NULL);
    const float *e6_b_data = e6_b ? (const float *)e6_b->data : NULL;
    int blk_T = block_len; /* temporal dimension: process all block samples through encoder */

    /* Allocate per-block feature buffer and per-frame output */
    float *features = (float *)calloc((size_t)rvq_dim * nf, sizeof(float));
    if (!features) { free(padded); free(e6_w); return TSAC_ERR_MEMORY; }

    if (!e6_w) { free(padded); free(e6_w); free(features); return TSAC_ERR_MEMORY; }

    /* Process each block independently through full encoder, keeping temporal
     * context (block_len time steps) throughout all layers. Only reduce to
     * a single feature vector at the output via center-frame selection. */
    float *blk_in  = (float *)malloc((size_t)e6_Ci * blk_T * sizeof(float));
    if (!blk_in) { free(padded); free(e6_w); free(features); return TSAC_ERR_MEMORY; }

    for (int f = 0; f < nf; f++) {
        int start = f * block_len;
        for (int t = 0; t < blk_T; t++) {
            for (int c = 0; c < e6_Ci; c++) {
                int si = start + t;
                int src_c = (c < channels) ? c : 0;
                blk_in[c * blk_T + t] = (si < padded_samples)
                    ? padded[si * channels + src_c] : 0.0f;
            }
        }

        /* model.6: Conv1d(ch→64, K=7) over block_len samples */
        float *cur = (float *)malloc((size_t)e6_Co * blk_T * sizeof(float));
        conv1d_s(cur, blk_in, e6_w, e6_b_data, blk_T, e6_K, e6_Ci, e6_Co);
        int cur_C = e6_Co;
        int cur_T = blk_T;

        /* Blocks 4→1: strided convs for temporal reduction */
        int enc_c_out[4] = {128, 256, 512, 1024};
        /* Stride = K/2 from weight tensor dims. Each block.4 weight_v has
         * dims=[Ci, K, Co] with K=4/8/16/16. Compute stride dynamically. */
        for (int blk = 4; blk >= 1; blk--) {
            int idx = 4 - blk;

            /* Snake before conv */
            char sn[128]; snprintf(sn, sizeof(sn), "encoder.block.%d.block.3.alpha", blk);
            DACTensor *sa = tf(ts, nt, sn);
            if (sa) snake_s(cur, cur, (const float *)sa->data, cur_C * cur_T, sa->dims[0]);

            /* Strided conv1d for channel expansion + temporal reduction.
             * Stride = K/2 computed from weight tensor dims. */
            char wvn[128]; snprintf(wvn, sizeof(wvn), "encoder.block.%d.block.4.weight_v", blk);
            char wgn[128]; snprintf(wgn, sizeof(wgn), "encoder.block.%d.block.4.weight_g", blk);
            char bn[128];  snprintf(bn, sizeof(bn), "encoder.block.%d.block.4.bias", blk);
            DACTensor *wv = tf(ts, nt, wvn), *wg = tf(ts, nt, wgn), *b = tf(ts, nt, bn);
            int c_Ci, c_K, c_Co;
            float *cw = dequant_weights(wv, wg, b, &c_Ci, &c_K, &c_Co, NULL);
            int c_stride = c_K / 2;
            int nxt_T = (cur_T + c_stride - 1) / c_stride;
            float *nxt = (float *)malloc((size_t)enc_c_out[idx] * nxt_T * sizeof(float));
            if (cw && c_Ci <= cur_C)
                conv1d_strided_s(nxt, cur, cw, b?(const float*)b->data:NULL, cur_T, nxt_T, c_K, c_Ci, c_Co, c_stride);
            free(cur); free(cw); cur = nxt; cur_C = enc_c_out[idx]; cur_T = nxt_T;

            /* Inner residual units on reduced temporal dimension */
            for (int inner = 2; inner <= 4; inner++) {
                int io = inner - 2;
                char isn[128]; snprintf(isn, sizeof(isn), "encoder.block.%d.block.%d.block.0.alpha", blk, io);
                DACTensor *isa = tf(ts, nt, isn);
                float *res = (float *)malloc((size_t)cur_C * cur_T * sizeof(float));
                if (!res) continue;
                memcpy(res, cur, (size_t)cur_C * cur_T * sizeof(float));
                if (isa) snake_s(cur, cur, (const float *)isa->data, cur_C * cur_T, isa->dims[0]);
                char iwn[128]; snprintf(iwn, sizeof(iwn), "encoder.block.%d.block.%d.block.1.weight_v", blk, io);
                char igw[128]; snprintf(igw, sizeof(igw), "encoder.block.%d.block.%d.block.1.weight_g", blk, io);
                char ibn[128]; snprintf(ibn, sizeof(ibn), "encoder.block.%d.block.%d.block.1.bias", blk, io);
                DACTensor *iwv = tf(ts, nt, iwn), *iwg = tf(ts, nt, igw), *ib = tf(ts, nt, ibn);
                int ic_Ci, ic_K, ic_Co;
                float *iw = dequant_weights(iwv, iwg, ib, &ic_Ci, &ic_K, &ic_Co, NULL);
                if (!iw) { free(res); continue; }
                float *co = (float *)malloc((size_t)ic_Co * cur_T * sizeof(float));
                if (co) {
                    conv1d_s(co, cur, iw, ib?(const float*)ib->data:NULL, cur_T, ic_K, ic_Ci, ic_Co);
                    char s2n[128]; snprintf(s2n, sizeof(s2n), "encoder.block.%d.block.%d.block.2.alpha", blk, io);
                    DACTensor *s2a = tf(ts, nt, s2n);
                    if (s2a) snake_s(co, co, (const float *)s2a->data, ic_Co * cur_T, s2a->dims[0]);
                    char w2n[128]; snprintf(w2n, sizeof(w2n), "encoder.block.%d.block.%d.block.3.weight_v", blk, io);
                    char g2n[128]; snprintf(g2n, sizeof(g2n), "encoder.block.%d.block.%d.block.3.weight_g", blk, io);
                    char b2n[128]; snprintf(b2n, sizeof(b2n), "encoder.block.%d.block.%d.block.3.bias", blk, io);
                    DACTensor *w2v=tf(ts,nt,w2n),*w2g=tf(ts,nt,g2n),*b2=tf(ts,nt,b2n);
                    int c2_Ci,c2_K,c2_Co; float *w2=dequant_weights(w2v,w2g,b2,&c2_Ci,&c2_K,&c2_Co,NULL);
                    float *uo=co;
                    if(w2){uo=(float*)malloc((size_t)c2_Co*cur_T*sizeof(float));
                     if(uo)conv1d_s(uo,co,w2,b2?(const float*)b2->data:NULL,cur_T,c2_K,c2_Ci,c2_Co);
                     else uo=co; free(w2);}
                    int nn=cur_C*cur_T;
                    for(int i=0;i<nn;i++)cur[i]=uo[i]+res[i];
                    if(uo!=co)free(uo); free(co);
                }
                free(iw); free(res);
            }
        }

        /* model.5 snake + feature extraction (cur_T should be ~1 after strides) */
        DACTensor *e5_a = tf(ts, nt, "encoder.block.5.alpha");
        if (e5_a) snake_s(cur, cur, (const float *)e5_a->data, cur_C * cur_T, e5_a->dims[0]);

        /* Take center frame as single feature vector per block */
        int center = cur_T / 2;
         for (int d = 0; d < rvq_dim && d < cur_C; d++)
            features[d * nf + f] = cur[d * cur_T + center];
        free(cur);
        int pct = (int)((float)(f + 1) / (float)nf * 100.0f);
        if (pct > 100) pct = 100;
        fprintf(stderr, "%3u%%\r", pct);
        fflush(stderr);
    }
    fprintf(stderr, "\n");
    free(blk_in); free(padded); free(e6_w);

    /* RVQ quantization on [rvq_dim, nf] features */

    /* RVQ quantization */
    int *indices = (int *)calloc((size_t)nf * n_codebooks, sizeof(int));
    if (!indices) { free(features); return TSAC_ERR_MEMORY; }
    float *residual_buf = (float *)malloc((size_t)rvq_dim * nf * sizeof(float));
    if (!residual_buf) { free(indices); free(features); return TSAC_ERR_MEMORY; }
    memcpy(residual_buf, features, (size_t)rvq_dim * nf * sizeof(float));

    for (int cb = 0; cb < n_codebooks && cb < 12; cb++) {
        char cb_name[128];
        snprintf(cb_name, sizeof(cb_name), "quantizer.quantizers.%d.codebook.weight", cb);
        DACTensor *codebook = tf(ts, nt, cb_name);
        if (!codebook) continue;
        int entries = codebook->dims[0];
        const float *cb_data = (const float *)codebook->data;
        for (int f = 0; f < nf; f++) {
            float best = 1e30f; int best_e = 0;
            for (int e = 0; e < entries; e++) {
                float d = 0;
                const float *c = cb_data + e * rvq_dim;
                for (int i = 0; i < rvq_dim; i++) {
                    float df = residual_buf[i * nf + f] - c[i];
                    d += df * df;
                }
                if (d < best) { best = d; best_e = e; }
            }
            indices[f * n_codebooks + cb] = best_e;
            const float *ch = cb_data + best_e * rvq_dim;
            for (int i = 0; i < rvq_dim; i++)
                residual_buf[i * nf + f] -= ch[i];
        }
    }
    free(features); free(residual_buf);
    *codebook_indices = indices;
    return TSAC_OK;
}
