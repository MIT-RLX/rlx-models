
/* Thread worker: conv1d over a slice of output channels. */
static void *conv1d_thread(void *arg) {
    Conv1dJob *j = (Conv1dJob *)arg;
    int Co_sub = j->oc_end - j->oc_start;
    if (Co_sub <= 0) return NULL;
    j->kernel(
        j->o + j->oc_start * j->T,
        j->x,
        j->w + j->oc_start * j->Ci * j->K,
        j->b ? j->b + j->oc_start : NULL,
        j->T, j->K, j->Ci, Co_sub
    );
    return NULL;
}

/* Parallel conv1d: split output channels across threads */
/* Parallel conv1d: split output channels across threads.
 * Calls kernel function pointer for actual computation. */
static void conv1d_parallel(void (*kern)(float*,const float*,const float*,const float*,int,int,int,int),
                            float *o, const float *x, const float *w, const float *b,
                            int T, int K, int Ci, int Co, int n_threads)
{
    if (n_threads <= 1 || Co < n_threads * 2) {
        kern(o, x, w, b, T, K, Ci, Co);
        return;
    }
    int nt = (n_threads < Co) ? n_threads : Co;
    pthread_t *tid = (pthread_t *)alloca(nt * sizeof(pthread_t));
    Conv1dJob jobs[nt];
    int per = (Co + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        jobs[t] = (Conv1dJob){
            .oc_start = t * per,
            .oc_end   = (t + 1) * per < Co ? (t + 1) * per : Co,
            .o = o, .x = x, .w = w, .b = b,
            .T = T, .K = K, .Ci = Ci, .Co = Co,
            .kernel = kern
        };
        pthread_create(&tid[t], NULL, conv1d_thread, &jobs[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
}

/* Thread worker: conv_transpose parallel over output channels */
typedef struct {
    int oc_start, oc_end;
    float *o;
    const float *x, *w;
    int Ti, To, K, Ci, Co, stride;
    void (*kernel)(float*,const float*,const float*,int,int,int,int,int,int);
} Convt1dJob;

/* Thread worker: convtranspose1d over a slice of output channels. */
static void *convt1d_thread(void *arg) {
    Convt1dJob *j = (Convt1dJob *)arg;
    int Co_sub = j->oc_end - j->oc_start;
    if (Co_sub <= 0) return NULL;
    j->kernel(
        j->o + j->oc_start * j->To,
        j->x,
        j->w + j->oc_start * j->Ci * j->K,
        j->Ti, j->To, j->K, j->Ci, Co_sub, j->stride
    );
    return NULL;
}

/* Parallel convtranspose1d: split output channels across threads. */
static void convt1d_parallel(void (*kern)(float*,const float*,const float*,int,int,int,int,int,int),
                             float *o, const float *x, const float *w,
                             int Ti, int To, int K, int Ci, int Co, int stride, int n_threads)
{
    memset(o, 0, (size_t)Co * To * sizeof(float));
    if (n_threads <= 1 || Co < n_threads * 2) {
        kern(o, x, w, Ti, To, K, Ci, Co, stride);
        return;
    }
    int nt = (n_threads < Co) ? n_threads : Co;
    pthread_t *tid = (pthread_t *)alloca(nt * sizeof(pthread_t));
    Convt1dJob jobs[nt];
    int per = (Co + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        jobs[t] = (Convt1dJob){
            .oc_start = t * per,
            .oc_end   = (t + 1) * per < Co ? (t + 1) * per : Co,
            .o = o, .x = x, .w = w,
            .Ti = Ti, .To = To, .K = K, .Ci = Ci, .Co = Co, .stride = stride,
            .kernel = kern
        };
        pthread_create(&tid[t], NULL, convt1d_thread, &jobs[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
}

/* Thread worker: snake parallel over elements */
typedef struct {
    int start, end;
    float *o;
    const float *x, *a;
    int C;
} SnakeJob;

/* Thread worker: snake activation over a slice of channels. */
static void *snake_thread(void *arg) {
    SnakeJob *j = (SnakeJob *)arg;
    for (int i = j->start; i < j->end; i++) {
        float v = j->x[i], al = j->a[i % j->C];
        if (al < 1e-6f) al = 1e-6f;
        float sa = sinf(al * v);
        j->o[i] = v + sa * sa / al;
    }
    return NULL;
}

/* Parallel snake activation: x + sin^2(alpha*x)/alpha per element. */
static void snake_parallel(float *o, const float *x, const float *a, int n, int C, int n_threads) {
    if (n_threads <= 1 || n < n_threads * 1024) {
        snake_s(o, x, a, n, C);
        return;
    }
    int nt = n_threads;
    pthread_t *tid = (pthread_t *)alloca(nt * sizeof(pthread_t));
    SnakeJob jobs[nt];
    int per = (n + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        jobs[t] = (SnakeJob){
            .start = t * per,
            .end   = (t + 1) * per < n ? (t + 1) * per : n,
            .o = o, .x = x, .a = a, .C = C
        };
        pthread_create(&tid[t], NULL, snake_thread, &jobs[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
}

/* Thread worker: add parallel over elements */
typedef struct {
    int start, end;
    float *o;
    const float *x, *y;
} AddJob;

/* Thread worker: element-wise addition (used for residual connections). */
static void *add_thread(void *arg) {
    AddJob *j = (AddJob *)arg;
    for (int i = j->start; i < j->end; i++)
        j->o[i] = j->x[i] + j->y[i];
    return NULL;
}

/* Parallel element-wise addition: o = x + y. */
static void add_parallel(float *o, const float *x, const float *y, int n, int n_threads) {
    if (n_threads <= 1 || n < n_threads * 2048) {
        for (int i = 0; i < n; i++) o[i] = x[i] + y[i];
        return;
    }
    int nt = n_threads;
    pthread_t *tid = (pthread_t *)alloca(nt * sizeof(pthread_t));
    AddJob jobs[nt];
    int per = (n + nt - 1) / nt;
