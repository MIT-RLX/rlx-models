/*
 * txc_format.c — .txc container parser (original TSAC format)
 *
 * Verified format (int16 encoding variant):
 *   Offset  Description
 *   ------  -----------
 *   0-3     "FBAZ" magic (ASCII)
 *   4-5     version (BE u16)
 *   6       flags (u8): bit0=stereo
 *   7       n_codebooks (u8): 1-12
 *   8-11    param1 u32 BE (batch_size * block_len)
 *   12-15   param2 u32 BE
 *   16-19   param3 u32 BE
 *   20-23   param4 u32 BE (possibly sample_rate LE)
 *   24+     int16 codebook_indices[n_frames * n_codebooks]
 *
 * Fallback (uint8 encoding):
 *   Header end auto-detected. Each index occupies 1 byte.
 */
#include "txc_format.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define TXC_MAGIC_BYTES { 'F', 'B', 'A', 'Z' }

/* CRC-32 lookup table (polynomial 0x04C11DB7, big-endian non-reflected),
 * reverse-engineered from original tsac binary at address 0x43dda0. */
static const uint32_t crc32_table[256] = {
    0x00000000, 0x04C11DB7, 0x09823B6E, 0x0D4326D9, 0x130476DC, 0x17C56B6B, 0x1A864DB2, 0x1E475005,
    0x2608EDB8, 0x22C9F00F, 0x2F8AD6D6, 0x2B4BCB61, 0x350C9B64, 0x31CD86D3, 0x3C8EA00A, 0x384FBDBD,
    0x4C11DB70, 0x48D0C6C7, 0x4593E01E, 0x4152FDA9, 0x5F15ADAC, 0x5BD4B01B, 0x569796C2, 0x52568B75,
    0x6A1936C8, 0x6ED82B7F, 0x639B0DA6, 0x675A1011, 0x791D4014, 0x7DDC5DA3, 0x709F7B7A, 0x745E66CD,
    0x9823B6E0, 0x9CE2AB57, 0x91A18D8E, 0x95609039, 0x8B27C03C, 0x8FE6DD8B, 0x82A5FB52, 0x8664E6E5,
    0xBE2B5B58, 0xBAEA46EF, 0xB7A96036, 0xB3687D81, 0xAD2F2D84, 0xA9EE3033, 0xA4AD16EA, 0xA06C0B5D,
    0xD4326D90, 0xD0F37027, 0xDDB056FE, 0xD9714B49, 0xC7361B4C, 0xC3F706FB, 0xCEB42022, 0xCA753D95,
    0xF23A8028, 0xF6FB9D9F, 0xFBB8BB46, 0xFF79A6F1, 0xE13EF6F4, 0xE5FFEB43, 0xE8BCCD9A, 0xEC7DD02D,
    0x34867077, 0x30476DC0, 0x3D044B19, 0x39C556AE, 0x278206AB, 0x23431B1C, 0x2E003DC5, 0x2AC12072,
    0x128E9DCF, 0x164F8078, 0x1B0CA6A1, 0x1FCDBB16, 0x018AEB13, 0x054BF6A4, 0x0808D07D, 0x0CC9CDCA,
    0x7897AB07, 0x7C56B6B0, 0x71159069, 0x75D48DDE, 0x6B93DDDB, 0x6F52C06C, 0x6211E6B5, 0x66D0FB02,
    0x5E9F46BF, 0x5A5E5B08, 0x571D7DD1, 0x53DC6066, 0x4D9B3063, 0x495A2DD4, 0x44190B0D, 0x40D816BA,
    0xACA5C697, 0xA864DB20, 0xA527FDF9, 0xA1E6E04E, 0xBFA1B04B, 0xBB60ADFC, 0xB6238B25, 0xB2E29692,
    0x8AAD2B2F, 0x8E6C3698, 0x832F1041, 0x87EE0DF6, 0x99A95DF3, 0x9D684044, 0x902B669D, 0x94EA7B2A,
    0xE0B41DE7, 0xE4750050, 0xE9362689, 0xEDF73B3E, 0xF3B06B3B, 0xF771768C, 0xFA325055, 0xFEF34DE2,
    0xC6BCF05F, 0xC27DEDE8, 0xCF3ECB31, 0xCBFFD686, 0xD5B88683, 0xD1799B34, 0xDC3ABDED, 0xD8FBA05A,
    0x690CE0EE, 0x6DCDFD59, 0x608EDB80, 0x644FC637, 0x7A089632, 0x7EC98B85, 0x738AAD5C, 0x774BB0EB,
    0x4F040D56, 0x4BC510E1, 0x46863638, 0x42472B8F, 0x5C007B8A, 0x58C1663D, 0x558240E4, 0x51435D53,
    0x251D3B9E, 0x21DC2629, 0x2C9F00F0, 0x285E1D47, 0x36194D42, 0x32D850F5, 0x3F9B762C, 0x3B5A6B9B,
    0x0315D626, 0x07D4CB91, 0x0A97ED48, 0x0E56F0FF, 0x1011A0FA, 0x14D0BD4D, 0x19939B94, 0x1D528623,
    0xF12F560E, 0xF5EE4BB9, 0xF8AD6D60, 0xFC6C70D7, 0xE22B20D2, 0xE6EA3D65, 0xEBA91BBC, 0xEF68060B,
    0xD727BBB6, 0xD3E6A601, 0xDEA580D8, 0xDA649D6F, 0xC423CD6A, 0xC0E2D0DD, 0xCDA1F604, 0xC960EBB3,
    0xBD3E8D7E, 0xB9FF90C9, 0xB4BCB610, 0xB07DABA7, 0xAE3AFBA2, 0xAAFBE615, 0xA7B8C0CC, 0xA379DD7B,
    0x9B3660C6, 0x9FF77D71, 0x92B45BA8, 0x9675461F, 0x8832161A, 0x8CF30BAD, 0x81B02D74, 0x857130C3,
    0x5D8A9099, 0x594B8D2E, 0x5408ABF7, 0x50C9B640, 0x4E8EE645, 0x4A4FFBF2, 0x470CDD2B, 0x43CDC09C,
    0x7B827D21, 0x7F436096, 0x7200464F, 0x76C15BF8, 0x68860BFD, 0x6C47164A, 0x61043093, 0x65C52D24,
    0x119B4BE9, 0x155A565E, 0x18197087, 0x1CD86D30, 0x029F3D35, 0x065E2082, 0x0B1D065B, 0x0FDC1BEC,
    0x3793A651, 0x3352BBE6, 0x3E119D3F, 0x3AD08088, 0x2497D08D, 0x2056CD3A, 0x2D15EBE3, 0x29D4F654,
    0xC5A92679, 0xC1683BCE, 0xCC2B1D17, 0xC8EA00A0, 0xD6AD50A5, 0xD26C4D12, 0xDF2F6BCB, 0xDBEE767C,
    0xE3A1CBC1, 0xE760D676, 0xEA23F0AF, 0xEEE2ED18, 0xF0A5BD1D, 0xF464A0AA, 0xF9278673, 0xFDE69BC4,
    0x89B8FD09, 0x8D79E0BE, 0x803AC667, 0x84FBDBD0, 0x9ABC8BD5, 0x9E7D9662, 0x933EB0BB, 0x97FFAD0C,
    0xAFB010B1, 0xAB710D06, 0xA6322BDF, 0xA2F33668, 0xBCB4666D, 0xB8757BDA, 0xB5365D03, 0xB1F740B4
};

/* CRC32 computation: polynomial 0x04C11DB7, shift-left, non-reflected. */
static uint32_t crc32(const uint8_t *data, size_t len, uint32_t crc) {
    for (size_t i = 0; i < len; i++) {
        uint8_t idx = (uint8_t)(crc >> 24) ^ data[i];
        crc = (crc << 8) ^ crc32_table[idx];
    }
    return crc;
}

/* Read a big-endian uint32 from a byte buffer. */
static uint32_t read_be32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

/* Locate TXC header boundary in 16-bit format. */
static int find_header_end_int16(const uint8_t *data, size_t data_size, int n_codebooks)
{
    int stride = n_codebooks * 2;
    for (int h = 16; h < 256 && h < (int)data_size; h++) {
        if ((data_size - (size_t)h) % (size_t)stride == 0)
            return h;
    }
    return 0;
}

/* Locate TXC header boundary in 8-bit format. */
static int find_header_end_uint8(const uint8_t *data, size_t data_size, int n_codebooks)
{
    for (int h = 8; h < 256 && h < (int)data_size; h++) {
        if ((data_size - (size_t)h) % (size_t)n_codebooks == 0)
            return h;
    }
    return 8;
}

void txc_header_init(TSCHeader *hdr, int stereo, int n_codebooks, int sample_rate)
{
    if (!hdr) return;
    hdr->magic[0] = 'F';
    hdr->magic[1] = 'B';
    hdr->magic[2] = 'A';
    hdr->magic[3] = 'Z';
    hdr->version     = 1;
    hdr->n_codebooks = (uint16_t)(n_codebooks & 0xFFFF);
    hdr->block_len   = 320;
    hdr->n_blocks    = 0;
    hdr->sample_rate = (uint32_t)sample_rate;
    hdr->flags       = stereo ? 1U : 0U;
    hdr->data_offset = sizeof(TSCHeader);
}

/* Serialize header + codebook indices into a TXC byte buffer. */
int txc_write(const TSCHeader *hdr,
              const int *codebook_indices, int n_frames,
#include "txc_parse.inc"
        /* Normal TXC (version>=1): 16-byte header, range-coded payload */
        if (hdr->version >= 1) {
            hdr->data_offset = 16;

            /* CRC32 verification (last 4 bytes of payload) */
            size_t payload_bytes = data_size - (size_t)hdr->data_offset;
            if (payload_bytes >= 4) {
                uint32_t stored = read_be32(data + data_size - 4);
                uint32_t computed = crc32(data + hdr->data_offset,
                                           payload_bytes - 4, 0xFFFFFFFF);
                if (stored != computed) {
                    fprintf(stderr, "tsac-ng: CRC32 mismatch (stored=0x%08x computed=0x%08x)\n",
                            stored, computed);
                    return TSAC_ERR_CODEC;
                }
            }

            fprintf(stderr,
                    "tsac-ng: normal TXC decode not yet implemented "
                    "(version=%u, frames=%u, codebooks=%u)\n",
                    hdr->version, hdr->n_blocks, hdr->n_codebooks);
            return TSAC_ERR_CODEC;
        }

        /* Fast TXC (version=0): detect raw uint8 vs 10-bit packed */
        int header_end = 8;
        while (header_end < 256 && header_end < (int)data_size &&
               ((data_size - (size_t)header_end) % (size_t)hdr->n_codebooks) != 0) {
            header_end++;
        }
        if (header_end >= (int)data_size || header_end >= 256)
            return TSAC_ERR_FORMAT;

        size_t idx_count = data_size - (size_t)header_end;
        int total_frames = (int)(idx_count / (size_t)hdr->n_codebooks);
        if (total_frames < 1 || (size_t)total_frames * (size_t)hdr->n_codebooks != idx_count)
            return TSAC_ERR_FORMAT;

        /* Validate: sample first bytes to detect range-coded format.
         * Raw indices are [0, cb_size-1] where cb_size is the
         * smallest power of two >= n_codebooks (e.g. nc=6 \u2192 cb=8). */
        int cb_size = 1;
        while (cb_size < (int)hdr->n_codebooks)
            cb_size *= 2;

        const uint8_t *src = data + header_end;
        int is_bitpacked = 0;
        size_t sample_count = idx_count < 64 ? idx_count : (size_t)64;
        for (size_t i = 0; i < sample_count; i++) {
            if ((int)src[i] >= cb_size) {
                is_bitpacked = 1;
                break;
            }
        }

        if (is_bitpacked) {
            const uint8_t *buf = data + 8;
            size_t bm_idx_count = data_size - 8;
            int total_bits = (int)bm_idx_count * 8;
            int total_indices = total_bits / 10;
            total_frames = total_indices / (int)hdr->n_codebooks;
            if (total_frames < 1)
                return TSAC_ERR_FORMAT;

            int bit_pos = 0;

            int *indices = (int *)calloc((size_t)total_indices, sizeof(int));
            if (!indices) return TSAC_ERR_MEMORY;

            int decoded = 0;
            while (decoded < total_indices) {
                int byte_off = bit_pos >> 3;
                uint32_t val = 0;
                int avail = (int)bm_idx_count - byte_off;
                if (avail >= 4)
                    memcpy(&val, buf + byte_off, 4);
                else if (avail > 0)
                    memcpy(&val, buf + byte_off, (size_t)avail);
                val = ((val & 0xFF) << 24) | ((val & 0xFF00) << 8)
                    | ((val >> 8) & 0xFF00) | (val >> 24);
                int shift = 22 - (bit_pos & 7);
                int sym = (int)((val >> shift) & 0x3FF);
                indices[decoded++] = sym;
                bit_pos += 10;
            }

            total_frames = decoded / (int)hdr->n_codebooks;
            if (total_frames < 1) {
                free(indices);
                return TSAC_ERR_FORMAT;
            }

            hdr->flags &= 0x7fU;
            hdr->data_offset = 8;
            hdr->block_len = 512;
            hdr->sample_rate = 44100;
            hdr->n_blocks = (uint32_t)total_frames;
            *codebook_indices = indices;
            *n_frames = total_frames;
            return TSAC_OK;
        }

        int *indices = (int *)calloc(idx_count, sizeof(int));
        if (!indices) return TSAC_ERR_MEMORY;

        for (size_t i = 0; i < idx_count; i++)
            indices[i] = (int)src[i];

        hdr->flags &= 0x7fU;
        hdr->data_offset = (uint32_t)header_end;
        hdr->block_len = 512;
        hdr->sample_rate = 44100;
        hdr->n_blocks = (uint32_t)total_frames;
        *codebook_indices = indices;
        *n_frames = total_frames;
        return TSAC_OK;
    }

    if (hdr->flags & 0x80U) {
        /* CRC32 verification for normal-mode compressed payload.
         * The last 4 bytes of the payload are the CRC32 (big-endian,
         * byte-swapped from computed value). Algorithm reverse-engineered
         * from original tsac binary: polynomial 0x04C11DB7, shift-left,
         * initial value 0xFFFFFFFF. */
        size_t payload_bytes = data_size - (size_t)hdr->data_offset;
        if (payload_bytes >= 4) {
            uint32_t stored = read_be32(data + data_size - 4);
            uint32_t computed = crc32(data + hdr->data_offset,
                                       payload_bytes - 4, 0xFFFFFFFF);
            if (stored != computed) {
                fprintf(stderr, "tsac-ng: CRC32 mismatch (stored=0x%08x computed=0x%08x)\n",
                        stored, computed);
                return TSAC_ERR_CODEC;
            }
        }
        fprintf(stderr,
                "tsac-ng: compressed/transformer-coded .txc payload is not decoded yet "
                "(frames=%u, codebooks=%u). Raw RVQ fallback would be invalid.\n",
                hdr->n_blocks, hdr->n_codebooks);
        return TSAC_ERR_CODEC;
    }

    int codebooks = (int)hdr->n_codebooks;

    int int16_header = find_header_end_int16(data, data_size, codebooks);
    int uint8_header = find_header_end_uint8(data, data_size, codebooks);
    int is_int16 = 0;
    int header_end;

    /* Prefer header closest to sizeof(TSCHeader)=28 for our encoder output */
    if (uint8_header >= 8 && (int16_header <= 0 || uint8_header >= int16_header)) {
        header_end = uint8_header;
    } else if (int16_header > 0) {
        header_end = int16_header;
        is_int16 = 1;
    } else {
        return TSAC_ERR_FORMAT;
    }

    hdr->data_offset = (uint32_t)header_end;

    size_t payload_size = data_size - (size_t)header_end;
    size_t idx_count = payload_size / (is_int16 ? 2 : 1);
    int total_frames = (int)(idx_count / (size_t)codebooks);

    if (total_frames < 1 || (size_t)total_frames * (size_t)codebooks != idx_count)
        return TSAC_ERR_FORMAT;

    int *indices = (int *)calloc(idx_count, sizeof(int));
    if (!indices) return TSAC_ERR_MEMORY;

    const uint8_t *src = data + header_end;

    if (is_int16) {
        for (size_t i = 0; i < idx_count; i++) {
            int16_t val = (int16_t)((uint16_t)src[i*2] | ((uint16_t)src[i*2+1] << 8));
            indices[i] = (int)val;
        }
    } else {
        for (size_t i = 0; i < idx_count; i++)
            indices[i] = (int)src[i];
    }

    *codebook_indices = indices;
    *n_frames = total_frames;

    hdr->block_len = 512;
    hdr->sample_rate = 44100;

    if (data_size >= 24 && is_int16) {
        uint32_t p3_le = (uint32_t)data[16] | ((uint32_t)data[17] << 8)
                       | ((uint32_t)data[18] << 16) | ((uint32_t)data[19] << 24);
        uint32_t p4_le = (uint32_t)data[20] | ((uint32_t)data[21] << 8)
                       | ((uint32_t)data[22] << 16) | ((uint32_t)data[23] << 24);
        if (p3_le == 44100 || p3_le == 48000) hdr->sample_rate = p3_le;
        if (p4_le == 44100 || p4_le == 48000) hdr->sample_rate = p4_le;
    }

    hdr->n_blocks = (uint32_t)total_frames;

    return TSAC_OK;
}
