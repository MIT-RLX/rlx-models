"""A deterministic, name-keyed PRNG both the Python reference and the Rust port
can reproduce exactly: FNV-1a over the parameter name seeds splitmix64, and each
draw is a uniform in [-s, s)."""
import numpy as np

MASK = (1 << 64) - 1


def fnv1a(name: str) -> int:
    h = 0xcbf29ce484222325
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001b3) & MASK
    return h


def splitmix64_stream(seed: int, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.uint64)
    s = seed & MASK
    for i in range(n):
        s = (s + 0x9E3779B97F4A7C15) & MASK
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z = z ^ (z >> 31)
        out[i] = z
    return out


def param_values(name: str, shape) -> np.ndarray:
    n = int(np.prod(shape)) if len(shape) else 1
    u = splitmix64_stream(fnv1a(name), n)
    # top 53 bits -> [0,1)
    f = (u >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    scale = (1.0 / shape[-1]) ** 0.5 if len(shape) >= 2 else 0.2
    return ((f - 0.5) * 2.0 * scale).reshape(shape).astype(np.float32)
