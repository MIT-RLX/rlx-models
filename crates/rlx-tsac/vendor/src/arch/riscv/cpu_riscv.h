#ifndef CPU_RISCV_H
#define CPU_RISCV_H

#ifdef __riscv

/* Architecture init and detection */
int cpu_arch_init(void);
const char *cpu_arch_name(void);
int riscv_rvv_available(void);

/* Runtime-dispatchable functions (automatically use RVV if available) */
void conv1d_riscv(float *o, const float *x, const float *w, const float *b,
                  int T, int K, int Ci, int Co);
void snake_riscv(float *o, const float *x, const float *a, int n, int C);
void add_riscv(float *o, const float *a, const float *b, int n);

/* RVV-specific implementations (for explicit RVV dispatch) */
#ifdef __riscv_v
void conv1d_rvv(float *o, const float *x, const float *w, const float *b,
                int T, int K, int Ci, int Co);
void snake_rvv(float *o, const float *x, const float *a, int n, int C);
void add_rvv(float *o, const float *a, const float *b, int n);
#endif

#endif /* __riscv */

#endif /* CPU_RISCV_H */
