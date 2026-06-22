    /* decoder blocks 1-4 */
    for (int block = 1; block <= 4; block++) {
        int target_C;
        switch (block) {
            case 1: target_C = 768; break;
            case 2: target_C = 384; break;
            case 3: target_C = 192; break;
            case 4: target_C = 96; break;
            default: target_C = current_C / 2;
        }

        /* snake before conv */
        char snake_name[128];
        snprintf(snake_name, sizeof(snake_name), "decoder.model.%d.block.0.alpha", block);
        DACTensor *snake_alpha = tf(ts, nt, snake_name);
        if (snake_alpha) {
            const float *alpha = (const float *)snake_alpha->data;
            int alpha_C = snake_alpha->dims[0];
            snake_parallel(current, current, alpha,
                           current_C * cur_frames, alpha_C, n_threads);
        }

        DBG("[DEBUG] After block%d snake NaN: %d/%d\n", block, count_nan(current, current_C*cur_frames), current_C*cur_frames);

        /* conv_transpose upsampling */
        char wv_name[128], wg_name[128], b_name[128];
        snprintf(wv_name, sizeof(wv_name), "decoder.model.%d.block.1.weight_v", block);
        snprintf(wg_name, sizeof(wg_name), "decoder.model.%d.block.1.weight_g", block);
        snprintf(b_name, sizeof(b_name), "decoder.model.%d.block.1.bias", block);

        DACTensor *wv = tf(ts, nt, wv_name);
        DACTensor *wg = tf(ts, nt, wg_name);
        DACTensor *b  = tf(ts, nt, b_name);

        int conv_Ci, conv_K, conv_Co, is_convt;
        float *w = dequant_weights(wv, wg, b, &conv_Ci, &conv_K, &conv_Co, &is_convt);
        const float *b_data = b ? (const float *)b->data : NULL;

        int conv_stride = conv_K / 2;
        int n_frames_out = cur_frames * conv_stride;

        float *next_buf = (float *)malloc((size_t)target_C * n_frames_out * sizeof(float));
        if (!next_buf) { free(current); free(w); return TSAC_ERR_MEMORY; }

        if (w) {
            convt1d_parallel(ops.conv_transpose1d, next_buf, current, w,
                             cur_frames, n_frames_out,
                             conv_K, conv_Ci, conv_Co, conv_stride, n_threads);
            if (b_data) {
                for (int c = 0; c < conv_Co; c++) {
                    for (int t = 0; t < n_frames_out; t++) {
                        next_buf[c * n_frames_out + t] += b_data[c];
                    }
                }
            }
        }

        char dump_name[64]; snprintf(dump_name, sizeof(dump_name), "block%d_convt", block);
        DUMP_ACT(next_buf, target_C*n_frames_out, dump_name);
        DBG("[DEBUG] After block%d convt NaN: %d/%d\n", block, count_nan(next_buf, target_C*n_frames_out), target_C*n_frames_out);
        {
            float mv = 0; for (int i = 0; i < target_C*n_frames_out; i++) { float a = fabsf(next_buf[i]); if (a > mv) mv = a; }
            DBG("[DEBUG] After block%d convt max_abs=%.2f [0]=%.4f\n", block, mv, next_buf[0]);
        }

        free(current);
        free(w);
        current = next_buf;
        current_C = target_C;
        cur_frames = n_frames_out;

        /* DAC residual units: Snake -> dilated Conv1d(K=7) -> Snake -> Conv1d(K=1) -> skip */
        for (int inner = 2; inner <= 4; inner++) {
            char inner_snake[128], inner_wv[128], inner_wg[128], inner_b[128];
            snprintf(inner_snake, sizeof(inner_snake),
                     "decoder.model.%d.block.%d.block.0.alpha", block, inner);
            snprintf(inner_wv, sizeof(inner_wv),
                     "decoder.model.%d.block.%d.block.1.weight_v", block, inner);
            snprintf(inner_wg, sizeof(inner_wg),
                     "decoder.model.%d.block.%d.block.1.weight_g", block, inner);
            snprintf(inner_b, sizeof(inner_b),
                     "decoder.model.%d.block.%d.block.1.bias", block, inner);

            DACTensor *is_alpha = tf(ts, nt, inner_snake);
            DACTensor *iwv = tf(ts, nt, inner_wv);
            DACTensor *iwg = tf(ts, nt, inner_wg);
            DACTensor *ib  = tf(ts, nt, inner_b);

            float *residual = (float *)malloc((size_t)current_C * cur_frames * sizeof(float));
            if (!residual) continue;
            memcpy(residual, current, (size_t)current_C * cur_frames * sizeof(float));

            if (is_alpha) {
                const float *alpha = (const float *)is_alpha->data;
                int alpha_C = is_alpha->dims[0];
                snake_parallel(current, current, alpha,
                               current_C * cur_frames, alpha_C, n_threads);
            }

            int ic_Ci, ic_K, ic_Co;
            float *iw = dequant_weights(iwv, iwg, ib, &ic_Ci, &ic_K, &ic_Co, NULL);
            const float *ib_data = ib ? (const float *)ib->data : NULL;

            if (iw) {
                float *conv_out = (float *)malloc((size_t)ic_Co * cur_frames * sizeof(float));
                if (conv_out) {
                    int dilation = (inner == 2) ? 1 : (inner == 3 ? 3 : 9);
                    if (dilation == 1)
                        conv1d_parallel(ops.conv1d, conv_out, current, iw, ib_data,
                                        cur_frames, ic_K, ic_Ci, ic_Co, n_threads);
                    else
                        conv1d_dilated_s(conv_out, current, iw, ib_data,
                                         cur_frames, ic_K, ic_Ci, ic_Co, dilation);

                    char snake2_name[128], wv2_name[128], wg2_name[128], b2_name[128];
                    snprintf(snake2_name, sizeof(snake2_name),
                             "decoder.model.%d.block.%d.block.2.alpha", block, inner);
                    snprintf(wv2_name, sizeof(wv2_name),
                             "decoder.model.%d.block.%d.block.3.weight_v", block, inner);
                    snprintf(wg2_name, sizeof(wg2_name),
                             "decoder.model.%d.block.%d.block.3.weight_g", block, inner);
                    snprintf(b2_name, sizeof(b2_name),
                             "decoder.model.%d.block.%d.block.3.bias", block, inner);
                    DACTensor *snake2 = tf(ts, nt, snake2_name);
                    if (snake2) {
                        snake_parallel(conv_out, conv_out, (const float *)snake2->data,
                                       ic_Co * cur_frames, snake2->dims[0], n_threads);
                    }

                    DACTensor *wv2 = tf(ts, nt, wv2_name);
                    DACTensor *wg2 = tf(ts, nt, wg2_name);
                    DACTensor *b2 = tf(ts, nt, b2_name);
                    int c2_Ci, c2_K, c2_Co;
                    float *w2 = dequant_weights(wv2, wg2, b2, &c2_Ci, &c2_K, &c2_Co, NULL);
                    float *unit_out = conv_out;
                    if (w2) {
                        unit_out = (float *)malloc((size_t)c2_Co * cur_frames * sizeof(float));
                        if (unit_out) {
                            conv1d_parallel(ops.conv1d, unit_out, conv_out, w2,
                                            b2 ? (const float *)b2->data : NULL,
                                            cur_frames, c2_K, c2_Ci, c2_Co, n_threads);
                        } else {
                            unit_out = conv_out;
                        }
                        free(w2);
                    }

                    int n = current_C * cur_frames;
                    if (ops.add) add_parallel(current, unit_out, residual, n, n_threads);
                    else for (int i = 0; i < n; i++) current[i] = unit_out[i] + residual[i];

                    if (unit_out != conv_out) free(unit_out);
                    free(conv_out);
                }
                free(iw);
            }
            free(residual);
        }
    }

