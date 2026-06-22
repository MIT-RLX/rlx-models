/*
 * cpu_arm.h — ARM64 NEON/SVE CPU acceleration header for TSAC.
 */

#ifndef CPU_ARM_H
#define CPU_ARM_H

/* Architecture initialization and detection */
int cpu_arch_init(void);
const char *cpu_arch_name(void);
int cpu_arch_has_sve(void);

#ifdef __aarch64__

/* NEON-optimized kernels */
void add_neon(float *o, const float *a, const float *b, int n);
void conv1d_neon(float *o, const float *x, const float *w, const float *b, 
                 int T, int K, int Ci, int Co);
void snake_neon(float *o, const float *x, const float *a, int n, int C);
void group_norm_neon(float *o, const float *x, const float *w, const float *b, 
                     int N, int G, float eps);

/* SVE-optimized kernels (only available when compiled with SVE support) */
#ifdef __ARM_FEATURE_SVE
void add_sve(float *o, const float *a, const float *b, int n);
void conv1d_sve(float *o, const float *x, const float *w, const float *b, 
                int T, int K, int Ci, int Co);
void snake_sve(float *o, const float *x, const float *a, int n, int C);
#endif

#endif /* __aarch64__ */

#endif /* CPU_ARM_H */
