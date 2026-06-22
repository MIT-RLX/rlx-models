        if (is_file_path)
            snprintf(dac_path, sizeof(dac_path), "%s", model_path);
        else
            snprintf(dac_path, sizeof(dac_path),
                     "%s/dac_stereo_q8.bin", model_path);
        int ret = model_loader_load(dac_path, ctx->model);
        if (ret != TSAC_OK) {
            fprintf(stderr, "[tsac-ng] Warning: model load failed (%d), "
                    "falling back to original tsac binary\n", ret);
        }
    }

    ctx->initialized = 1;
    return ctx;
}

/* Release all resources held by a TSAC context.
 * Safe to call with NULL. */
void tsac_free(TSACContext *ctx)
{
#include "tsac_wrap.inc"
    uint8_t *out_data = NULL;
    size_t out_size = 0;
    int ret = tsac_compress(ctx, pcm, n_samples, channels, &out_data, &out_size, n_codebooks);
    free(pcm);

    if (ret == TSAC_OK) {
        FILE *fo = fopen(out_txc, "wb");
        if (!fo) {
            tsac_free_buffer(out_data);
            return TSAC_ERR_FILE;
        }
        if (fwrite(out_data, 1, out_size, fo) != out_size) {
            fclose(fo);
            tsac_free_buffer(out_data);
            return TSAC_ERR_FILE;
        }
        fclose(fo);
        tsac_free_buffer(out_data);

        uint32_t sr = wav_hdr[24] | ((uint32_t)wav_hdr[25] << 8)
                    | ((uint32_t)wav_hdr[26] << 16) | ((uint32_t)wav_hdr[27] << 24);
        if (sr == 0) sr = 44100;
        double duration = (double)n_samples / (double)sr;
        double bitrate = duration > 0.0 ? (double)out_size * 8.0 / duration / 1000.0 : 0.0;
        double model_mb = 0.0;
        if (ctx->model && ctx->model->tensors) {
            for (int t = 0; t < ctx->model->n_tensors; t++)
                model_mb += (double)ctx->model->tensors[t].data_size;
        }
        double max_mem = (model_mb * 3.0) / (1024.0 * 1024.0 * 1024.0);
