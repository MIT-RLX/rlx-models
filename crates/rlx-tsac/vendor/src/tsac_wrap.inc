#include "tsac_helpers.inc"
    if (!f) return TSAC_ERR_FILE;

    /* Read WAV header to get format info */
    uint8_t wav_hdr[44];
    if (fread(wav_hdr, 1, 44, f) != 44) {
        fclose(f);
        return TSAC_ERR_FORMAT;
    }

    if (memcmp(wav_hdr, "RIFF", 4) != 0 || memcmp(wav_hdr + 8, "WAVE", 4) != 0) {
        fclose(f);
        return TSAC_ERR_FORMAT;
    }

    uint16_t audio_fmt = wav_hdr[20] | ((uint16_t)wav_hdr[21] << 8);
    uint16_t channels  = wav_hdr[22] | ((uint16_t)wav_hdr[23] << 8);
    uint32_t data_size = 0;

    if (audio_fmt != 1 && audio_fmt != 3) { /* PCM int16 or IEEE float */
        fclose(f);
        return TSAC_ERR_FORMAT;
    }

    /* Find data chunk — start searching after fmt chunk */
    uint32_t fmt_size = wav_hdr[16] | ((uint32_t)wav_hdr[17] << 8)
                      | ((uint32_t)wav_hdr[18] << 16) | ((uint32_t)wav_hdr[19] << 24);
    long data_search_start = 20 + fmt_size;
    fseek(f, data_search_start, SEEK_SET);
    uint8_t chunk_hdr[8];
    while (fread(chunk_hdr, 1, 8, f) == 8) {
        if (memcmp(chunk_hdr, "data", 4) == 0) {
            data_size = (uint32_t)chunk_hdr[4]
                      | ((uint32_t)chunk_hdr[5] << 8)
                      | ((uint32_t)chunk_hdr[6] << 16)
                      | ((uint32_t)chunk_hdr[7] << 24);
            break;
        }
        uint32_t chunk_size = (uint32_t)chunk_hdr[4]
                            | ((uint32_t)chunk_hdr[5] << 8)
                            | ((uint32_t)chunk_hdr[6] << 16)
                            | ((uint32_t)chunk_hdr[7] << 24);
        fseek(f, chunk_size, SEEK_CUR);
    }

    if (data_size == 0) {
        fclose(f);
        return TSAC_ERR_FORMAT;
    }

    int n_samples = data_size / (channels * (audio_fmt == 3 ? 4 : 2));

    /* Read PCM samples */
    float *pcm = (float *)calloc((size_t)n_samples * channels, sizeof(float));
    if (!pcm) {
        fclose(f);
        return TSAC_ERR_MEMORY;
    }

    if (audio_fmt == 3) {
        /* IEEE float */
        for (int i = 0; i < n_samples * channels; i++) {
            uint8_t b[4];
            if (fread(b, 1, 4, f) != 4) break;
            uint32_t bits = (uint32_t)b[0] | ((uint32_t)b[1] << 8)
                          | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
            float val;
            memcpy(&val, &bits, 4);
            pcm[i] = val;
        }
    } else {
        /* 16-bit PCM */
        for (int i = 0; i < n_samples * channels; i++) {
            uint8_t b[2];
            if (fread(b, 1, 2, f) != 2) break;
            int16_t sample = (int16_t)(b[0] | ((uint16_t)b[1] << 8));
            pcm[i] = (float)sample / 32768.0f;
        }
    }
    fclose(f);

