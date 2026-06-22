    if (!ctx) return;

    if (ctx->backend == TSAC_BACKEND_CUDA)
        tsac_cuda_shutdown(ctx->backend_priv);
    else if (ctx->backend == TSAC_BACKEND_HIP)
        tsac_hip_shutdown(ctx->backend_priv);
    else if (ctx->backend == TSAC_BACKEND_VULKAN)
        tsac_vk_shutdown(ctx->backend_priv);
    else if (ctx->backend == TSAC_BACKEND_LLVM)
        tsac_llvm_shutdown(ctx->backend_priv);

    dac_model_destroy(ctx->model);
    free(ctx->model_path);
    memset(ctx, 0, sizeof(*ctx));
    free(ctx);
}

/* Compress float32 PCM audio to TXC codebook indices. */
int tsac_compress(TSACContext *ctx,
                  const float *pcm, int n_samples, int channels,
                  uint8_t **out_data, size_t *out_size,
                  int n_codebooks)
{
    if (!ctx || !ctx->initialized || !pcm || !out_data || !out_size)
        return TSAC_ERR_PARAM;

    if (n_codebooks < 1 || n_codebooks > 12)
        return TSAC_ERR_PARAM;

    if (channels != 1 && channels != 2)
        return TSAC_ERR_PARAM;

    if (n_samples < 1)
        return TSAC_ERR_PARAM;

    TSCHeader hdr;
    txc_header_init(&hdr, channels == 2 ? 1 : 0, n_codebooks, 44100);

    hdr.block_len  = 320;
    hdr.n_blocks   = (n_samples + hdr.block_len - 1) / hdr.block_len;

    int total_samples = hdr.n_blocks * hdr.block_len;
    int n_frames = hdr.n_blocks;

    int *codebook_indices = NULL;

    if (ctx->backend == TSAC_BACKEND_CUDA) {
        int ret = tsac_cuda_encode(ctx->backend_priv, ctx->model,
                                   pcm, n_samples, channels,
                                   n_codebooks, hdr.block_len,
                                   &codebook_indices, &n_frames);
        if (ret != TSAC_OK) return ret;
    } else if (ctx->backend == TSAC_BACKEND_HIP) {
        int ret = tsac_hip_encode(ctx->backend_priv, ctx->model,
                                  pcm, n_samples, channels,
                                  n_codebooks, hdr.block_len,
                                  &codebook_indices, &n_frames);
        if (ret != TSAC_OK) return ret;
    } else {
        int ret = cpu_encoder_run(ctx->model->tensors, ctx->model->n_tensors,
                                  pcm, n_samples, channels,
                                  n_codebooks, hdr.block_len,
                                  &codebook_indices, &n_frames);
        if (ret != TSAC_OK) return ret;
    }

    int ret = txc_write(&hdr, codebook_indices, n_frames,
                        out_data, out_size);
    free(codebook_indices);

    return ret;
}

/* Decode a TXC byte buffer into float32 PCM.
 * Returns TSAC_OK on success, error code otherwise.
 * Caller must free *out_pcm. */
int tsac_decompress(TSACContext *ctx,
                    const uint8_t *txc_data, size_t txc_size,
                    float **out_pcm, int *out_samples, int *out_channels)
{
    if (!ctx || !ctx->initialized || !txc_data || !out_pcm || !out_samples || !out_channels)
        return TSAC_ERR_PARAM;

    TSCHeader hdr;
    int *codebook_indices = NULL;
    int n_frames = 0;

    int ret = txc_read(txc_data, txc_size, &hdr, &codebook_indices, &n_frames);
    if (ret != TSAC_OK) return ret;

    *out_channels = hdr.flags ? 2 : 1;
    *out_samples  = hdr.n_blocks * hdr.block_len;

    size_t total_samples = (size_t)(*out_samples) * (*out_channels);
    float *pcm = (float *)calloc(total_samples, sizeof(float));
    if (!pcm) {
        free(codebook_indices);
        return TSAC_ERR_MEMORY;
    }

    if (ctx->backend == TSAC_BACKEND_CUDA) {
        ret = tsac_cuda_decode(ctx->backend_priv, ctx->model,
                               codebook_indices, n_frames,
                               hdr.n_codebooks, hdr.block_len,
                               *out_channels,
                               pcm, *out_samples);
    } else if (ctx->backend == TSAC_BACKEND_HIP) {
        ret = tsac_hip_decode(ctx->backend_priv, ctx->model,
                               codebook_indices, n_frames,
                               hdr.n_codebooks, hdr.block_len,
                               *out_channels,
                               pcm, *out_samples);
    } else if (ctx->backend == TSAC_BACKEND_VULKAN) {
        ret = tsac_vk_decode(ctx->backend_priv, ctx->model,
                              codebook_indices, n_frames,
                              hdr.n_codebooks, hdr.block_len,
                              *out_channels,
                              pcm, *out_samples);
    } else if (ctx->backend == TSAC_BACKEND_LLVM) {
        ret = tsac_llvm_decode(ctx->backend_priv, ctx->model,
                                codebook_indices, n_frames,
                                hdr.n_codebooks, hdr.block_len,
                                *out_channels,
                                pcm, *out_samples);
    } else {
        ret = dac_model_decode(ctx->model, codebook_indices, n_frames,
                               hdr.n_codebooks, hdr.block_len,
                               *out_channels,
                               pcm, *out_samples,
                               ctx->n_threads);
    }

    if (ret != TSAC_OK) {
        free(pcm);
        free(codebook_indices);
        return ret;
    }

    free(codebook_indices);
    *out_pcm = pcm;
    return TSAC_OK;
}

/* Compress a WAV file to TXC format.
 * Reads PCM from in_wav, encodes with n_codebooks, writes to out_txc. */
int tsac_compress_file(TSACContext *ctx, const char *in_wav, const char *out_txc, int n_codebooks)
{
    if (!ctx || !in_wav || !out_txc)
        return TSAC_ERR_PARAM;

    FILE *f = fopen(in_wav, "rb");
