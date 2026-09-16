"""Score OpenImageDenoise on a dataset, with this crate's metric.

    python -m venv env && env/bin/pip install oidn numpy
    env/bin/python scripts/oidn_reference.py denoise_val.bin

OIDN is the reference a learned denoiser has to be measured against — it is the
one that ships in Blender, and unlike the OptiX denoiser it is Apache-2.0 down
to the weights, so it can be run and quoted.

The dataset stores colour compressed by `x / (1 + x)`; OIDN wants linear HDR. So
each tile is expanded back to radiance, denoised, and recompressed before the
error is taken, which puts the number in the same space as every other row of
the benchmark. The `unfiltered` line it prints should match the one
`rlx-denoise train` reports for the same file — if it does not, the two are not
measuring the same thing and neither number means anything.

One caveat on reading the result: OIDN is given 128x128 tiles independently,
which is smaller than the whole frames it is designed for, so this understates
it somewhat.
"""
import sys
import ctypes
import glob
import numpy as np
import oidn

# The Python binding does not wrap `oidnSetFilter1b`, and HDR mode is not
# optional here: the renderer's output is unbounded radiance, and OIDN in LDR
# mode assumes [0, 1] and would clip every highlight. Bind it directly.
_lib_path = glob.glob(
    __import__("os").path.join(__import__("os").path.dirname(oidn.__file__), "lib*", "libOpenImageDenoise.so*")
)[0]
_lib = ctypes.CDLL(_lib_path)
_lib.oidnSetFilter1b.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
_lib.oidnSetFilter1b.restype = None
print(f"OpenImageDenoise {oidn.oidn_version} at {_lib_path}")

EPS = 0.01

path = sys.argv[1]
raw = open(path, "rb").read()
# The plane counts come from the header rather than being assumed: the set has
# had 9 and 11 input planes at different times, and reading it wrong silently
# feeds OIDN the depth plane as a normal.
if raw[:8] == b"RLXDN002":
    tile = int.from_bytes(raw[8:12], "little")
    count = int.from_bytes(raw[12:16], "little")
    IN_C = int.from_bytes(raw[16:20], "little")
    OUT_C = int.from_bytes(raw[20:24], "little")
    header = 24
elif raw[:8] == b"RLXDN001":
    tile = int.from_bytes(raw[8:12], "little")
    count = int.from_bytes(raw[12:16], "little")
    IN_C, OUT_C, header = 9, 3, 16
else:
    raise SystemExit(f"{path}: not an RLXDN dataset")
a = np.frombuffer(raw[header:], dtype="<f4").reshape(count, IN_C + OUT_C, tile, tile)
print(f"{count} tiles of {tile}, {IN_C}+{OUT_C} planes")

dev = oidn.NewDevice(oidn.DEVICE_TYPE_DEFAULT)
oidn.CommitDevice(dev)

def chw_to_hwc(x):
    return np.ascontiguousarray(np.transpose(x, (1, 2, 0)).astype(np.float32))

sum_raw = sum_oidn = 0.0
n = 0
for i in range(count):
    colour = a[i, 0:3]
    albedo = a[i, 3:6]
    normal = a[i, 6:9]
    target = a[i, IN_C : IN_C + 3]

    # Compressed -> radiance. Clip below 1 so the inverse stays finite.
    lin = np.clip(colour, 0.0, 0.999999)
    lin = lin / (1.0 - lin)

    src = chw_to_hwc(lin)
    alb = chw_to_hwc(np.clip(albedo, 0.0, 1.0))
    nrm = chw_to_hwc(normal)
    out = np.zeros_like(src)

    f = oidn.NewFilter(dev, "RT")
    oidn.SetSharedFilterImage(f, "color", src, oidn.FORMAT_FLOAT3, tile, tile)
    oidn.SetSharedFilterImage(f, "albedo", alb, oidn.FORMAT_FLOAT3, tile, tile)
    oidn.SetSharedFilterImage(f, "normal", nrm, oidn.FORMAT_FLOAT3, tile, tile)
    oidn.SetSharedFilterImage(f, "output", out, oidn.FORMAT_FLOAT3, tile, tile)
    _lib.oidnSetFilter1b(ctypes.c_void_p(f), b"hdr", True)
    oidn.CommitFilter(f)
    oidn.ExecuteFilter(f)
    err = oidn.GetDeviceError(dev)
    code = err[0] if isinstance(err, (tuple, list)) else err
    if code != oidn.ERROR_NONE:
        raise SystemExit(f"oidn error {err}")
    oidn.ReleaseFilter(f)

    den = np.transpose(out, (2, 0, 1))
    den = np.clip(den, 0.0, None)
    den = den / (1.0 + den)          # back to the stored range

    sum_raw += float(np.sum((colour - target) ** 2 / (target ** 2 + EPS)))
    sum_oidn += float(np.sum((den - target) ** 2 / (target ** 2 + EPS)))
    n += target.size

oidn.ReleaseDevice(dev)
print(f"unfiltered  {np.sqrt(sum_raw / n):.5f}")
print(f"OIDN        {np.sqrt(sum_oidn / n):.5f}   "
      f"{np.sqrt(sum_raw / n) / np.sqrt(sum_oidn / n):.2f}x")
