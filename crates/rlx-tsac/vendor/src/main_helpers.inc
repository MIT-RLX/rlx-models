        if (tmp_fd < 0) {
            fprintf(stderr, "Error: cannot create temp file\n");
            return 1;
        }
        if (close(tmp_fd) != 0) {}
        ret = tsac_compress_file(ctx, infile, tmp_path, n_codebooks);
        if (ret == TSAC_OK)
            ret = tsac_decompress_file(ctx, tmp_path, outfile);
        if (remove(tmp_path) != 0 && verbose)
            fputs("Warning: failed to remove temp file\n", stderr);
    }
    return ret;
}

int main(int argc, char **argv)
{
    int use_cuda   = 0;
    int use_hip    = 0;
    int use_vulkan = 0;
    int use_llvm   = 0;
    int n_threads  = 1;
    int n_codebooks = 6;
    int verbose    = 0;
    int separate_channels = 0;
    int force_channels = 0;
    int fast_mode = 0;
    const char *model_path = NULL;
    const char *trf_model_path = NULL;
    int batch_size = 0;

    static const struct option long_opts[] = {
        {"cuda",              no_argument,       0, 0},
        {"hip",               no_argument,       0, 0},
        {"vulkan",            no_argument,       0, 0},
        {"llvm",              no_argument,       0, 0},
        {"help",              no_argument,       0, 'h'},
        {"n_codebooks",       required_argument, 0, 'q'},
        {"separate_channels", no_argument,       0, 's'},
        {"channels",          required_argument, 0, 'c'},
        {"fast",              no_argument,       0, 'f'},
        {"model",             required_argument, 0, 'm'},
        {"trf_model",         required_argument, 0, 'M'},
        {"batch_size",        required_argument, 0, 1},
        {0, 0, 0, 0}
    };

    int opt_idx = 0;
    int optc;
    while ((optc = getopt_long(argc, argv, "+T:q:vhsc:fm:M:", long_opts, &opt_idx)) != -1) {
        switch (optc) {
        case 0:
            if (strcmp(long_opts[opt_idx].name, "cuda") == 0)
                use_cuda = 1;
            else if (strcmp(long_opts[opt_idx].name, "hip") == 0)
                use_hip = 1;
            else if (strcmp(long_opts[opt_idx].name, "vulkan") == 0)
                use_vulkan = 1;
            else if (strcmp(long_opts[opt_idx].name, "llvm") == 0)
                use_llvm = 1;
            break;
        case 1: /* --batch_size */
