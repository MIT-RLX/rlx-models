#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "tsac.h"
#include "dac_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Load a .bin model file into a DACModel */
int model_loader_load(const char *path, DACModel *model);

/* Free model resources */
void model_loader_free(DACModel *model);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_LOADER_H */
