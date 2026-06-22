/*
 * tsac-ng main — compatible with original tsac CLI.
 *
 * usage: tsac [options] c|d|t infile outfile
 *
 * Supported options (original tsac compat):
 *   --cuda / --hip / --vulkan / --llvm   GPU/accelerator backends
 *   -q n       n_codebooks (1-12 stereo, 1-9 mono, default=max)
 *   -T n       number of threads (default=1)
 *   -v         verbose mode
 *   -h         show help
 *   -s         separate channels (stereo as dual mono)
 *   -c n       force channel count
 *   -f         fast mode (no transformer)
 *   -m path    model filename
 *   -M path    transformer model filename
 *   --batch_size n   force batch size (default=auto)
 */

#include "tsac.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "tsac-ng version %s\n"
        "Very low bitrate audio compression (compatible with original tsac)\n"
        "\n"
        "usage: %s [options] c infile outfile     audio compression\n"
        "                    d infile outfile     audio decompression\n"
        "                    t infile outfile     audio compression then decompression\n"
        "\n"
        "Available options:\n"
        "-h --help               show the help\n"
        "--cuda                  enable CUDA support\n"
        "--hip                   enable HIP/ROCm support\n"
        "--vulkan                enable Vulkan compute support\n"
        "--llvm                  enable LLVM JIT backend (experimental)\n"
        "-T n                    number of threads (default=1)\n"
        "-q --n_codebooks n      number of codebooks, which sets the quality\n"
        "                        and bit rate (from 1 to 12 for joint stereo\n"
        "                        otherwise from 1 to 9, default = max value)\n"
        "-s --separate_channels  encode stereo as separate channels\n"
        "-c --channels n         force the number of input channels\n"
        "-f --fast               faster encoding and decoding\n"
        "-m --model filename     model filename\n"
        "-M --trf_model filename transformer model filename\n"
        "--batch_size n          force the batch size (default=auto)\n"
        "-v                      verbose mode\n"
        "\n",
        tsac_version(), prog);
}


/* Find the first directory containing dac_stereo_q8.bin.
 * Searches common install locations including Termux paths. */
/* Search common paths for the DAC model binary file. */
static const char *find_model_dir_(void) {
    static const char *search_paths[] = {
        "models/tsac",
        "/usr/share/tsac",
        "/usr/lib/tsac",
        "/data/data/com.termux/files/usr/share/tsac",  /* Termux */
        "/data/data/com.termux/files/home/develop/tsac-ng/models",
        NULL
    };
    for (int i = 0; search_paths[i]; i++) {
        char test[512];
        snprintf(test, sizeof(test), "%s/dac_stereo_q8.bin", search_paths[i]);
        FILE *f = fopen(test, "rb");
        if (f) { if (fclose(f) != 0) {} return search_paths[i]; }
    }
    return "/usr/share/tsac";  /* fallback */
}



/* Execute the requested command (compress/decompress/roundtrip).
 * Returns TSAC_OK on success, error code otherwise. */
/* Execute compress/decompress/roundtrip based on command character. */
static int execute_command(TSACContext *ctx, const char *cmd,
                           const char *infile, const char *outfile,
                           int n_codebooks, int verbose) {
    int ret = 0;
    if (cmd[0] == 'c') {
        if (verbose) fprintf(stderr, "Compressing %s -> %s\n", infile, outfile);
        ret = tsac_compress_file(ctx, infile, outfile, n_codebooks);
    } else if (cmd[0] == 'd') {
        if (verbose) fprintf(stderr, "Decompressing %s -> %s\n", infile, outfile);
        ret = tsac_decompress_file(ctx, infile, outfile);
    } else if (cmd[0] == 't') {
        if (verbose) fprintf(stderr, "Round-trip test: %s -> (compress) -> (decompress) -> %s\n",
                             infile, outfile);
        char tmp_path[] = "/tmp/tsac_rt_XXXXXX";
        int tmp_fd = mkstemp(tmp_path);
#include "main_helpers.inc"
            batch_size = (int)strtol(optarg, NULL, 10);
            break;
        case 'T':
            n_threads = (int)strtol(optarg, NULL, 10);
            if (n_threads < 1) n_threads = 1;
            break;
        case 'q':
            n_codebooks = (int)strtol(optarg, NULL, 10);
            if (n_codebooks < 1) n_codebooks = 1;
            if (n_codebooks > 12) n_codebooks = 12;
            break;
        case 'v':
            verbose = 1;
            break;
        case 's':
            separate_channels = 1;
            break;
        case 'c':
            force_channels = (int)strtol(optarg, NULL, 10);
            break;
        case 'f':
            fast_mode = 1;
            break;
        case 'm':
            model_path = optarg;
            break;
        case 'M':
            trf_model_path = optarg;
            break;
        case 'h':
        default:
            print_usage(argv[0]);
            return (optc == 'h') ? 0 : 1;
        }
    }

    if (argc - optind < 3) {
        fprintf(stderr, "Error: missing command or file arguments\n");
        print_usage(argv[0]);
        return 1;
    }

    const char *cmd     = argv[optind + 0];
    const char *infile  = argv[optind + 1];
    const char *outfile = argv[optind + 2];

    if (strlen(cmd) != 1 || (cmd[0] != 'c' && cmd[0] != 'd' && cmd[0] != 't')) {
        fprintf(stderr, "Error: unknown command '%s' (expected c, d, or t)\n", cmd);
        return 1;
    }

    /* Auto-detect conflicts */
    int gpu_count = use_cuda + use_hip + use_vulkan + use_llvm;
    if (gpu_count > 1) {
        fprintf(stderr, "Error: --cuda, --hip, --vulkan, --llvm are mutually exclusive\n");
        return 1;
    }

    TSACBackend backend = TSAC_BACKEND_CPU;
    const char *backend_name = "CPU";
    if (use_cuda)   { backend = TSAC_BACKEND_CUDA;   backend_name = "CUDA"; }
    if (use_hip)    { backend = TSAC_BACKEND_HIP;    backend_name = "HIP"; }
    if (use_vulkan) { backend = TSAC_BACKEND_VULKAN; backend_name = "Vulkan"; }
    if (use_llvm)   { backend = TSAC_BACKEND_LLVM;   backend_name = "LLVM JIT"; }

    /* Determine model directory */
    const char *model_dir = find_model_dir_();


    if (verbose) {
        fprintf(stderr, "TSAC-ng v%s\n", tsac_version());
        fprintf(stderr, "Backend: %s\n", backend_name);
        fprintf(stderr, "Threads: %d\n", n_threads);
        fprintf(stderr, "Quality: %d codebooks\n", n_codebooks);
        fprintf(stderr, "Model:   %s/dac_stereo_q8.bin\n", model_dir);
        if (model_path)      fprintf(stderr, "Model file: %s\n", model_path);
        if (trf_model_path)  fprintf(stderr, "TRF model:  %s\n", trf_model_path);
        if (fast_mode)       fprintf(stderr, "Mode: fast (no transformer)\n");
        if (separate_channels) fprintf(stderr, "Channels: separate\n");
        if (force_channels)  fprintf(stderr, "Force channels: %d\n", force_channels);
        if (batch_size)      fprintf(stderr, "Batch size: %d\n", batch_size);
    }

    TSACContext *ctx = tsac_init(backend, n_threads, model_path ? model_path : model_dir);
    if (!ctx) {
        fprintf(stderr, "Error: failed to initialize codec\n");
        return 1;
    }

    if (separate_channels)  fputs("Warning: --separate-channels not yet implemented\n", stderr);
    if (force_channels)     fputs("Warning: --force-channels not yet implemented\n", stderr);
    if (fast_mode)          fputs("Warning: --fast (transformer) mode not yet implemented\n", stderr);
    if (trf_model_path)     fputs("Warning: --trf-model not yet implemented\n", stderr);
    if (batch_size)         fputs("Warning: --batch-size not yet implemented\n", stderr);

    int ret = execute_command(ctx, cmd, infile, outfile, n_codebooks, verbose);

    if (ret != TSAC_OK) {
        fprintf(stderr, "Error: operation failed (code %d)\n", ret);
    } else if (verbose) {
        fprintf(stderr, "Success.\n");
    }

    tsac_free(ctx);
    return (ret == TSAC_OK) ? 0 : 1;
}
