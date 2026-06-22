static GpuWeight *hip_find_enc_weight(HipBackend *b, const char *name) {
    char fullname[256];
    snprintf(fullname, sizeof(fullname), "enc:%s", name);
    return hip_find_weight(b, fullname);
}

/* CPU-side dequantization helper for weight cache */
/* Dequantize BF8/float32 weight tensor (HIP copy of CPU version). */
static float *dequant_weights(const DACTensor *weight_v, const DACTensor *weight_g,
                               const DACTensor *bias,
                               int *out_Ci, int *out_K, int *out_Co, int *is_conv_transpose) {
    if (!weight_v || !weight_g) return NULL;

    int d0 = weight_v->dims[0];
    int d1 = weight_v->dims[1];
    int d2 = weight_v->dims[2];

    int Co = bias ? bias->dims[0] : d2;
    int K = d1;
    int Ci = (d0 == Co) ? d2 : d0;

    *out_Ci = Ci;
    *out_K = K;
    *out_Co = Co;
    *is_conv_transpose = (d0 == Co) ? 1 : 0;

    int per_input = (Ci == (int)weight_g->dims[2]);
    int nel = Ci * K * Co;

    float *cpu_buf = (float*)malloc(nel * sizeof(float));
    if (!cpu_buf) return NULL;

    const float *g_scales = (const float*)weight_g->data;
    const uint8_t *v_data = (const uint8_t*)weight_v->data;

    for (int co = 0; co < Co; co++) {
        for (int ci = 0; ci < Ci; ci++) {
            float g_val = per_input ? g_scales[ci] : g_scales[co];
            for (int k = 0; k < K; k++) {
                int src_idx;
                if (per_input) {
                    src_idx = co * K * Ci + k * Ci + ci;
                } else {
                    src_idx = ci * K * Co + k * Co + co;
                }
                int8_t v_val = (int8_t)v_data[src_idx];
                int dst_idx = co * Ci * K + ci * K + k;
                cpu_buf[dst_idx] = g_val * ((float)v_val - 128.0f) / 127.0f;
            }
        }
    }

    return cpu_buf;
}

/* Upload all model weights to GPU memory for fast decoding. */
static int hip_upload_weights(HipBackend *b, DACTensor *ts, int nt) {
    if (!ts) return TSAC_ERR_PARAM;

    /* Upload dequantized conv weights */
    const char *weight_names[] = {
        "decoder.model.0",
        "decoder.model.1.block.1",
        "decoder.model.2.block.1",
        "decoder.model.3.block.1",
        "decoder.model.4.block.1",
        "decoder.model.6",
        /* Inner blocks (model.1-4, inner 2-4) */
        "decoder.model.1.block.2.block.1",
        "decoder.model.1.block.3.block.1",
        "decoder.model.1.block.4.block.1",
        "decoder.model.2.block.2.block.1",
        "decoder.model.2.block.3.block.1",
        "decoder.model.2.block.4.block.1",
        "decoder.model.3.block.2.block.1",
        "decoder.model.3.block.3.block.1",
        "decoder.model.3.block.4.block.1",
        "decoder.model.4.block.2.block.1",
        "decoder.model.4.block.3.block.1",
        "decoder.model.4.block.4.block.1",
    };
    int n_wl = sizeof(weight_names) / sizeof(weight_names[0]);

    b->n_gpu_weights = 0;
    for (int i = 0; i < n_wl && b->n_gpu_weights < MAX_GPU_WEIGHTS; i++) {
        char wv[160], wg[160], bi[160];
        snprintf(wv, sizeof(wv), "%s.weight_v", weight_names[i]);
        snprintf(wg, sizeof(wg), "%s.weight_g", weight_names[i]);
        snprintf(bi, sizeof(bi), "%s.bias",      weight_names[i]);

        DACTensor *twv = F(ts, nt, wv);
        DACTensor *twg = F(ts, nt, wg);
        DACTensor *tbi = F(ts, nt, bi);

        if (!twv) continue;

        int Ci, K, Co, is_convt;
        float *w_f32 = dequant_weights(twv, twg, tbi, &Ci, &K, &Co, &is_convt);
        if (!w_f32) continue;

        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
        snprintf(gw->name, sizeof(gw->name), "%s", weight_names[i]);
        gw->Ci = Ci;
        gw->K  = K;
        gw->Co = Co;
        gw->is_convt = is_convt;

        size_t w_bytes = (size_t)Ci * K * Co * sizeof(float);
        hipMalloc(&gw->d_data, w_bytes);
        hipMemcpy(gw->d_data, w_f32, w_bytes, hipMemcpyHostToDevice);
        free(w_f32);

        /* Upload bias tensor separately */
        if (tbi && b->n_gpu_weights < MAX_GPU_WEIGHTS) {
            GpuWeight *gbias = &b->gpu_weights[b->n_gpu_weights++];
            snprintf(gbias->name, sizeof(gbias->name), "%s.bias", weight_names[i]);
            gbias->Ci = tbi->dims[0];
            gbias->K = 1;
            gbias->Co = 1;
            gbias->is_convt = 0;
            size_t bias_bytes = (size_t)tbi->dims[0] * sizeof(float);
            hipMalloc(&gbias->d_data, bias_bytes);
            hipMemcpy(gbias->d_data, tbi->data, bias_bytes, hipMemcpyHostToDevice);
        }
    }

    /* Upload encoder weights with "enc:" prefix */
    const char *enc_weight_names[] = {
        "encoder.model.0",
        "encoder.model.1.block.1",
        "encoder.model.2.block.1",
        "encoder.model.3.block.1",
        "encoder.model.4.block.1",
        "encoder.model.6",
        /* Inner blocks */
        "encoder.model.1.block.2.block.1",
        "encoder.model.1.block.3.block.1",
        "encoder.model.1.block.4.block.1",
        "encoder.model.2.block.2.block.1",
        "encoder.model.2.block.3.block.1",
        "encoder.model.2.block.4.block.1",
        "encoder.model.3.block.2.block.1",
        "encoder.model.3.block.3.block.1",
        "encoder.model.3.block.4.block.1",
        "encoder.model.4.block.2.block.1",
        "encoder.model.4.block.3.block.1",
        "encoder.model.4.block.4.block.1",
    };
    int n_enc_wl = sizeof(enc_weight_names) / sizeof(enc_weight_names[0]);
    for (int i = 0; i < n_enc_wl && b->n_gpu_weights < MAX_GPU_WEIGHTS; i++) {
        char wv[160], wg[160], bi[160];
        snprintf(wv, sizeof(wv), "%s.weight_v", enc_weight_names[i]);
        snprintf(wg, sizeof(wg), "%s.weight_g", enc_weight_names[i]);
        snprintf(bi, sizeof(bi), "%s.bias",      enc_weight_names[i]);

        DACTensor *twv = F(ts, nt, wv);
        DACTensor *twg = F(ts, nt, wg);
        DACTensor *tbi = F(ts, nt, bi);

        if (!twv) continue;

        int Ci, K, Co, is_convt;
        float *w_f32 = dequant_weights(twv, twg, tbi, &Ci, &K, &Co, &is_convt);
        if (!w_f32) continue;

        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
        snprintf(gw->name, sizeof(gw->name), "enc:%s", enc_weight_names[i]);
        gw->Ci = Ci;
        gw->K  = K;
        gw->Co = Co;
        gw->is_convt = is_convt;

        size_t w_bytes = (size_t)Ci * K * Co * sizeof(float);
        hipMalloc(&gw->d_data, w_bytes);
        hipMemcpy(gw->d_data, w_f32, w_bytes, hipMemcpyHostToDevice);
        free(w_f32);

        if (tbi && b->n_gpu_weights < MAX_GPU_WEIGHTS) {
            GpuWeight *gbias = &b->gpu_weights[b->n_gpu_weights++];
            snprintf(gbias->name, sizeof(gbias->name), "enc:%s.bias", enc_weight_names[i]);
            gbias->Ci = tbi->dims[0];
            gbias->K = 1;
            gbias->Co = 1;
