    float *pcm = NULL;
    int out_samples = 0;
    int out_channels = 0;
    ret = tsac_decompress(ctx, txc_data, (size_t)file_size, &pcm, &out_samples, &out_channels);
    free(txc_data);
    txc_data = NULL;

    if (ret != TSAC_OK) {
        return ret;
    }

    /* Write WAV file with float PCM data (format 3 = IEEE float) */
    FILE *fo = fopen(out_wav, "wb");
    if (!fo) {
        free(pcm);
        return TSAC_ERR_FILE;
    }

    uint32_t pcm_data_size = (uint32_t)(out_samples * out_channels * sizeof(float));
    uint32_t riff_size = 36 + pcm_data_size;

    uint8_t wav_hdr[44];
    /* RIFF chunk */
    memcpy(wav_hdr + 0, "RIFF", 4);
    wav_hdr[4] = (uint8_t)(riff_size);
    wav_hdr[5] = (uint8_t)(riff_size >> 8);
    wav_hdr[6] = (uint8_t)(riff_size >> 16);
    wav_hdr[7] = (uint8_t)(riff_size >> 24);
    memcpy(wav_hdr + 8, "WAVE", 4);
    /* fmt chunk */
    memcpy(wav_hdr + 12, "fmt ", 4);
    wav_hdr[16] = 16; /* fmt chunk size (little-endian) */
    wav_hdr[17] = 0;
    wav_hdr[18] = 0;
    wav_hdr[19] = 0;
    wav_hdr[20] = 3; /* format = IEEE float */
    wav_hdr[21] = 0;
    wav_hdr[22] = (uint8_t)out_channels;
    wav_hdr[23] = 0;
    wav_hdr[24] = (uint8_t)(sample_rate);
    wav_hdr[25] = (uint8_t)(sample_rate >> 8);
    wav_hdr[26] = (uint8_t)(sample_rate >> 16);
    wav_hdr[27] = (uint8_t)(sample_rate >> 24);
    uint32_t byte_rate = sample_rate * out_channels * sizeof(float);
    wav_hdr[28] = (uint8_t)(byte_rate);
    wav_hdr[29] = (uint8_t)(byte_rate >> 8);
    wav_hdr[30] = (uint8_t)(byte_rate >> 16);
    wav_hdr[31] = (uint8_t)(byte_rate >> 24);
    uint16_t block_align = out_channels * sizeof(float);
    wav_hdr[32] = (uint8_t)(block_align);
    wav_hdr[33] = (uint8_t)(block_align >> 8);
    wav_hdr[34] = 32; /* bits per sample */
    wav_hdr[35] = 0;
    /* data chunk */
    memcpy(wav_hdr + 36, "data", 4);
    wav_hdr[40] = (uint8_t)(pcm_data_size);
    wav_hdr[41] = (uint8_t)(pcm_data_size >> 8);
    wav_hdr[42] = (uint8_t)(pcm_data_size >> 16);
    wav_hdr[43] = (uint8_t)(pcm_data_size >> 24);

    if (fwrite(wav_hdr, 1, 44, fo) != 44) {
        fclose(fo);
        free(pcm);
        return TSAC_ERR_FILE;
    }

    /* Write interleaved float PCM samples */
    for (int i = 0; i < out_samples * out_channels; i++) {
        uint32_t bits;
        memcpy(&bits, &pcm[i], 4);
        uint8_t b[4];
        b[0] = (uint8_t)(bits);
        b[1] = (uint8_t)(bits >> 8);
        b[2] = (uint8_t)(bits >> 16);
        b[3] = (uint8_t)(bits >> 24);
        if (fwrite(b, 1, 4, fo) != 4) {
            fclose(fo);
            free(pcm);
            return TSAC_ERR_FILE;
        }
    }

    fclose(fo);
    free(pcm);

    if (out_samples > 0 && sample_rate > 0) {
        double duration = (double)out_samples / (double)sample_rate;
        double bitrate = duration > 0.0 ? (double)file_size * 8.0 / duration / 1000.0 : 0.0;
        double model_mb = 0.0;
        if (ctx->model && ctx->model->tensors) {
            for (int t = 0; t < ctx->model->n_tensors; t++)
