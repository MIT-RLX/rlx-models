/* model_loader.c — model loader for tsac-ng. */
#include "model_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAGIC_HDR 0x23f4aefb
#define MAGIC_TNS 0x23f4aefa

/* Check for libnc override: if /tmp/libnc_OVR_<layer_name>.bin exists,
 * replace weight_v data with the float32 reference.
 * Uses tensor name with '.' replaced by '_' as filename.
 * Example: decoder.model.0.weight_v → /tmp/libnc_OVR_decoder_model_0_weight_v.bin */
#define LIBNC_OVR_DIR "/tmp/libnc_OVR_"

/* Load a .bin model file into a DACModel structure. */
int model_loader_load(const char *path, DACModel *model)
{
    if (!path || !model) return TSAC_ERR_PARAM;

    FILE *f = fopen(path, "rb");
    if (!f) return TSAC_ERR_FILE;

    /* Read all file contents for efficient scanning */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *buf = (uint8_t *)malloc(fsize);
    if (!buf) { fclose(f); return TSAC_ERR_MEMORY; }
    fread(buf, 1, fsize, f);
    fclose(f);

    /* Verify header */
    uint32_t magic, type;
    memcpy(&magic, buf, 4);
    memcpy(&type, buf + 4, 4);
    if (magic != MAGIC_HDR) { free(buf); return TSAC_ERR_FORMAT; }

    /* Find JSON end */
    long json_end = 8;
    while (json_end < fsize && buf[json_end] != '}') json_end++;
    if (json_end >= fsize) { free(buf); return TSAC_ERR_FORMAT; }
    json_end++;

    /* Scan for all tensor magic offsets */
    long *tensor_offsets = NULL;
    int n_tensors = 0;
    for (long pos = json_end; pos < fsize - 4; pos++) {
        uint32_t m;
        memcpy(&m, buf + pos, 4);
        if (m == MAGIC_TNS) {
            tensor_offsets = realloc(tensor_offsets, (n_tensors + 1) * sizeof(long));
            tensor_offsets[n_tensors++] = pos;
            pos += 3; /* skip known magic bytes */
        }
    }

    if (n_tensors == 0) { free(buf); return TSAC_ERR_FORMAT; }

    /* Allocate tensors */
    model->n_tensors = n_tensors;
    model->tensors = (DACTensor *)calloc(n_tensors, sizeof(DACTensor));

    /* Read each tensor */
    for (int i = 0; i < n_tensors; i++) {
        DACTensor *t = &model->tensors[i];
        long pos = tensor_offsets[i];

        uint32_t m, f1, nd, nl;
        memcpy(&m, buf + pos, 4); pos += 4;
        memcpy(&f1, buf + pos, 4); pos += 4;
        memcpy(&nd, buf + pos, 4); pos += 4;
        memcpy(&nl, buf + pos, 4); pos += 4;
        t->ndims = nd;

        for (int d = 0; d < (int)nd; d++) {
            uint32_t v; memcpy(&v, buf + pos, 4); pos += 4;
            t->dims[d] = v;
        }

        int name_bytes = nl < 128 ? nl : 127;
        memcpy(t->name, buf + pos, name_bytes);
        t->name[name_bytes] = '\0';
        pos += nl;

        /* Data size = next tensor start - current data position */
        long next_start = (i + 1 < n_tensors) ? tensor_offsets[i + 1] : fsize;
        t->data_size = (int)(next_start - pos);
        if (t->data_size > 0) {
            t->data = (uint8_t *)malloc(t->data_size);
            memcpy(t->data, buf + pos, t->data_size);
        }
        if (strstr(t->name, "weight_v") != NULL) {
            int dims_product = 1;
            for (int d = 0; d < (int)nd; d++) dims_product *= t->dims[d];
            int as_uint8  = dims_product;
            int as_float32 = dims_product * 4;
            if (t->data_size == as_float32) t->elem_size = 4;
            else if (t->data_size == as_uint8) t->elem_size = 1;
            else t->elem_size = 1;
        } else {
            t->elem_size = 4;
        }

        /* LibNC override: check for float32 weight override file */
        if (strstr(t->name, "weight_v")) {
            char ovr_path[512];
            /* Build path from tensor name: replace '.' with '_' */
            int name_len = (int)strlen(t->name);
            snprintf(ovr_path, sizeof(ovr_path), "%s%s.bin", LIBNC_OVR_DIR, t->name);
            /* Replace dots only in the name portion, not the appended .bin */
            for (char *p = ovr_path + strlen(LIBNC_OVR_DIR); *p && (p - ovr_path) < (int)(strlen(LIBNC_OVR_DIR) + name_len); p++)
                if (*p == '.') *p = '_';
            FILE *ovf = fopen(ovr_path, "rb");
            if (ovf) {
                fseek(ovf, 0, SEEK_END);
                long ov_size = ftell(ovf);
                fseek(ovf, 0, SEEK_SET);
                int dims_prod = 1;
                for (int d = 0; d < (int)nd; d++) dims_prod *= t->dims[d];
                if (ov_size == dims_prod * 4) {
                    float *ov_f32 = (float *)malloc(ov_size);
                    fread(ov_f32, 1, ov_size, ovf);
                    /* Runtime format is [Co][Ci][K]. Convert to flat [d0][K][d2] order
                     * so dequant_weights' rearrangement produces correct [Co][Ci][K].
                     * For conv1d (d0!=d2): d0=Ci, d1=K, d2=Co. flat [Ci][K][Co].
                     * For convt (d0==d2):  d0=Co, d1=K, d2=Ci. flat [Co][K][Ci]. */
                    /* Runtime format IS [Co][Ci][K] float32 (nc_convert output).
                     * Store directly as-is. Use elem_size=0 sentinel to signal
                     * dequant_weights that data is already in [Co][Ci][K] order,
                     * skipping the standard flat→[Co][Ci][K] rearrangement. */
                    free(t->data);
                    t->data = (uint8_t *)ov_f32;
                    t->data_size = (int)ov_size;
                    t->elem_size = 0;  /* sentinel: data is [Co][Ci][K], skip rearrange */
                    fprintf(stderr, "[model_loader] LIBNC OVR: %s (%ld bytes) [Co][Ci][K] direct\n",
                            t->name, ov_size);
                }
                fclose(ovf);
            }
        }
    }

    free(tensor_offsets);
    free(buf);
    return TSAC_OK;
}

/* Release all memory held by a DACModel. */
void model_loader_free(DACModel *model)
{
    if (!model || !model->tensors) return;
    for (int i = 0; i < model->n_tensors; i++)
        free(model->tensors[i].data);
    free(model->tensors);
    model->tensors = NULL;
    model->n_tensors = 0;
}
