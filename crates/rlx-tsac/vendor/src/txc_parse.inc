              uint8_t **out_data, size_t *out_size)
{
    if (!hdr || !codebook_indices || !out_data || !out_size)
        return TSAC_ERR_PARAM;

    /* Use uint8 encoding: 1 byte per codebook index */
    size_t hdr_size = sizeof(TSCHeader);
    size_t idx_count = (size_t)n_frames * hdr->n_codebooks;
    size_t idx_bytes = idx_count; /* 1 byte per index */
    size_t total_size = hdr_size + idx_bytes;

    uint8_t *buf = (uint8_t *)calloc(total_size, 1);
    if (!buf) return TSAC_ERR_MEMORY;

    TSCHeader *out_hdr = (TSCHeader *)buf;
    buf[0] = hdr->magic[0]; buf[1] = hdr->magic[1];
    buf[2] = hdr->magic[2]; buf[3] = hdr->magic[3];
    buf[4] = (uint8_t)(hdr->version >> 8);
    buf[5] = (uint8_t)(hdr->version & 0xFF);
    buf[6] = (uint8_t)(hdr->flags & 0xFF);
    buf[7] = (uint8_t)(hdr->n_codebooks & 0xFF);
    uint32_t be32;
    be32 = hdr->block_len;   buf[ 8]=(be32>>24); buf[ 9]=(be32>>16); buf[10]=(be32>>8); buf[11]=be32;
    be32 = hdr->sample_rate; buf[12]=(be32>>24); buf[13]=(be32>>16); buf[14]=(be32>>8); buf[15]=be32;
    out_hdr->n_blocks = (uint32_t)n_frames;
    out_hdr->data_offset = (uint32_t)hdr_size;

    uint8_t *idx_data = buf + hdr_size;
    for (size_t i = 0; i < idx_count; i++)
        idx_data[i] = (uint8_t)(codebook_indices[i] & 0xFF);

    *out_data = buf;
    *out_size = total_size;
    return TSAC_OK;
}

/* Parse a complete TXC byte buffer into header + codebook index array. */
int txc_read(const uint8_t *data, size_t data_size,
             TSCHeader *hdr,
             int **codebook_indices, int *n_frames)
{
    if (!data || !hdr || !codebook_indices || !n_frames)
        return TSAC_ERR_PARAM;

    if (data_size < 8)
        return TSAC_ERR_FORMAT;

    /* Validate magic */
    if (data[0] != 'F' || data[1] != 'B' ||
        data[2] != 'A' || data[3] != 'Z')
        return TSAC_ERR_FORMAT;

    /* Parse fixed header fields */
    memset(hdr, 0, sizeof(TSCHeader));
    hdr->magic[0] = 'F'; hdr->magic[1] = 'B';
    hdr->magic[2] = 'A'; hdr->magic[3] = 'Z';

    uint16_t ver_be = (uint16_t)((data[4] << 8) | data[5]);
    uint16_t ver_le = (uint16_t)(data[4] | (data[5] << 8));
    hdr->version = (ver_be <= 255) ? ver_be : ver_le;

    if (hdr->version > 255)
        return TSAC_ERR_FORMAT;

    hdr->flags       = data[6];
    hdr->n_codebooks = data[7];
    hdr->sample_rate = 44100;

    if (hdr->n_codebooks < 1 || hdr->n_codebooks > 12)
        return TSAC_ERR_FORMAT;

    if (data_size >= 16) {
        hdr->n_blocks = read_be32(data + 8);
        hdr->block_len = 512;
    }

    if ((hdr->flags & 0x80U)) {
        /* Fast TXC: either raw uint8 indices (tsac-ng encoder) or
         * range-coded indices (original tsac -f mode).  Detect which
         * by checking whether the payload starts with valid raw
         * codebook index values (all < n_codebooks). */
