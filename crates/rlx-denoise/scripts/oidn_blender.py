"""Score OpenImageDenoise on a dataset using Blender's own bundled library.

    python3 scripts/oidn_blender.py real9_testset.bin [predictions.f32]

Pass a second path to write OIDN's predictions as raw planar f32, laid out the
same way `rlx-denoise eval --dump` writes ours — so the two can be differenced
tile for tile to see *where* one is beating the other, which the single number
cannot say.

The sibling `oidn_reference.py` drives the `oidn` PyPI package, which ships a
Linux `.so`. This one talks to `libOpenImageDenoise.dylib` inside Blender.app
through ctypes, which is worth doing for its own sake: it is the exact binary
Cycles calls, at the same quality setting Cycles defaults to, so the number it
prints is what Blender would produce and not an approximation of it.

Settings are taken from `intern/cycles/integrator/denoiser_oidn_base.cpp`:
filter "RT", hdr on, srgb off, quality HIGH.

The dataset stores colour compressed by `x / (1 + x)`; OIDN wants linear HDR.
Each tile is expanded back to radiance, denoised, and recompressed before the
error is taken, so the result lands in the same space as every other row of the
benchmark. The `unfiltered` line printed here has to match the one
`rlx-denoise train` reports for the same file — if it does not, the two are not
measuring the same thing.
"""

import ctypes
import sys

import numpy as np

LIB = "/Applications/Blender.app/Contents/Resources/lib/libOpenImageDenoise.dylib"
MAGIC = b"RLXDN002"
MAGIC_V1 = b"RLXDN001"
LOSS_EPSILON = 0.01

FORMAT_FLOAT3 = 3
# The default device here is Metal, which only reads buffers it allocated.
# The CPU device takes host pointers and runs the same weights.
DEVICE_TYPE_CPU = 1
QUALITY_HIGH = 6  # OIDN_QUALITY_HIGH
ERROR_NONE = 0


def load():
    lib = ctypes.CDLL(LIB)
    lib.oidnNewDevice.restype = ctypes.c_void_p
    lib.oidnNewDevice.argtypes = [ctypes.c_int]
    lib.oidnCommitDevice.argtypes = [ctypes.c_void_p]
    lib.oidnNewFilter.restype = ctypes.c_void_p
    lib.oidnNewFilter.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.oidnSetSharedFilterImage.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.oidnSetFilterBool.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
    lib.oidnSetFilterInt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.oidnCommitFilter.argtypes = [ctypes.c_void_p]
    lib.oidnExecuteFilter.argtypes = [ctypes.c_void_p]
    lib.oidnReleaseFilter.argtypes = [ctypes.c_void_p]
    lib.oidnGetDeviceError.restype = ctypes.c_int
    lib.oidnGetDeviceError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]
    return lib


def read_header(raw):
    if raw[:8] == MAGIC:
        return (
            int.from_bytes(raw[8:12], "little"),
            int.from_bytes(raw[12:16], "little"),
            int.from_bytes(raw[16:20], "little"),
            int.from_bytes(raw[20:24], "little"),
            24,
        )
    if raw[:8] == MAGIC_V1:
        return (
            int.from_bytes(raw[8:12], "little"),
            int.from_bytes(raw[12:16], "little"),
            9,
            3,
            16,
        )
    raise SystemExit("not an RLXDN file")


def relative_error(y, t):
    return float(np.mean((y - t) ** 2 / (t * t + LOSS_EPSILON)))


def main():
    if not 2 <= len(sys.argv) <= 3:
        raise SystemExit(__doc__)
    dump = sys.argv[2] if len(sys.argv) == 3 else None
    raw = open(sys.argv[1], "rb").read()
    tile, count, ins, outs, header = read_header(raw)
    pixels = tile * tile
    per_tile = (ins + outs) * pixels
    body = np.frombuffer(raw, dtype=np.float32, offset=header)
    if body.size != count * per_tile:
        raise SystemExit(f"{body.size} floats for {count} tiles of {tile}")

    lib = load()
    dev = ctypes.c_void_p(lib.oidnNewDevice(DEVICE_TYPE_CPU))
    lib.oidnCommitDevice(dev)

    unfiltered = 0.0
    denoised = 0.0
    predictions = []
    for i in range(count):
        t = body[i * per_tile : (i + 1) * per_tile]
        # Planar [C, H, W] in the file; OIDN wants interleaved [H, W, 3].
        planes = t[: ins * pixels].reshape(ins, tile, tile)
        target = t[ins * pixels :].reshape(outs, tile, tile)

        colour = np.ascontiguousarray(planes[0:3].transpose(1, 2, 0))
        albedo = np.ascontiguousarray(planes[3:6].transpose(1, 2, 0))
        normal = np.ascontiguousarray(planes[6:9].transpose(1, 2, 0))
        tgt = np.ascontiguousarray(target.transpose(1, 2, 0))

        # Undo x/(1+x) to get radiance back. The compression is total, so a
        # stored 1.0 came from an infinity and has no finite inverse; clamp
        # just below it rather than dividing by zero.
        c = np.clip(colour, 0.0, 1.0 - 1e-6)
        linear = np.ascontiguousarray(c / (1.0 - c), dtype=np.float32)
        # Normals are stored mapped to [0,1]; OIDN wants them signed.
        nrm = np.ascontiguousarray(normal * 2.0 - 1.0, dtype=np.float32)
        alb = np.ascontiguousarray(albedo, dtype=np.float32)
        out = np.zeros_like(linear)

        f = ctypes.c_void_p(lib.oidnNewFilter(dev, b"RT"))
        for name, buf in ((b"color", linear), (b"albedo", alb), (b"normal", nrm), (b"output", out)):
            lib.oidnSetSharedFilterImage(
                f, name, buf.ctypes.data, FORMAT_FLOAT3, tile, tile, 0, 0, 0
            )
        # Exactly what Cycles sets.
        lib.oidnSetFilterBool(f, b"hdr", True)
        lib.oidnSetFilterBool(f, b"srgb", False)
        lib.oidnSetFilterInt(f, b"quality", QUALITY_HIGH)
        lib.oidnCommitFilter(f)
        lib.oidnExecuteFilter(f)
        msg = ctypes.c_char_p()
        code = lib.oidnGetDeviceError(dev, ctypes.byref(msg))
        if code != ERROR_NONE:
            raise SystemExit(f"OIDN error {code}: {msg.value}")
        lib.oidnReleaseFilter(f)

        # Back into the compressed range the metric is defined in.
        recompressed = out / (1.0 + out)
        unfiltered += relative_error(colour, tgt)
        denoised += relative_error(recompressed, tgt)
        if dump:
            # Planar, to match what `rlx-denoise eval --dump` writes, so the
            # two can be differenced tile for tile without a transpose.
            predictions.append(np.ascontiguousarray(recompressed.transpose(2, 0, 1)))

    if dump:
        with open(dump, "wb") as f:
            for p in predictions:
                f.write(p.astype(np.float32).tobytes())

    unfiltered = (unfiltered / count) ** 0.5
    denoised = (denoised / count) ** 0.5
    print(f"tiles       {count} of {tile}x{tile}")
    print(f"unfiltered  {unfiltered:.5f}")
    print(f"OIDN        {denoised:.5f}   {unfiltered / denoised:.2f}x")


if __name__ == "__main__":
    main()
