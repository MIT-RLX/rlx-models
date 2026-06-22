# Reference TSAC binaries only (Bellard original + standalone tsac-ng).
# RLX native codec runs on the host — see `examples/perf_bench.rs`.
#
# Build (from repo root):
#   bash crates/rlx-tsac/docker/run.sh build

# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 debian:bookworm-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY vendor/ vendor/

# Avoid AVX-512 in the reference image (linux/amd64 under QEMU on Apple hosts).
RUN perl -pi -e 's/ -mavx512f -mavx512vl -mavx512bw//g' vendor/CMakeLists.txt \
    && cmake -S vendor -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)"

FROM --platform=linux/amd64 debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gawk python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/tsac-ng /usr/local/bin/tsac-ng
COPY --from=builder /src/build/libtsac-ng.so* /usr/local/lib/
RUN ldconfig

RUN mkdir -p /opt/tsac \
    && curl -fsSL "https://bellard.org/tsac/tsac-2024-04-08.tar.gz" \
        | tar xzf - -C /opt/tsac --strip-components=1 \
    && chmod +x /opt/tsac/tsac

COPY docker/bench_ref.sh /usr/local/bin/bench_ref.sh
RUN chmod +x /usr/local/bin/bench_ref.sh

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/bench_ref.sh"]
