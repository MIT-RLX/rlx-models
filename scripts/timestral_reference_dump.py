#!/usr/bin/env python3
"""Dump Timestral reference forecasts for the rlx-timestral parity tests.

Runs the upstream `timestral` package (https://github.com/mistralai/timestral) on a
fixed set of cases and writes inputs, intermediates and forecasts to one safetensors
file. NaN in an input marks a missing value, exactly as the upstream API takes it.

    git clone https://github.com/mistralai/timestral /tmp/timestral/src
    PYTHONPATH=/tmp/timestral/src/src python3 scripts/timestral_reference_dump.py \
        --weights /tmp/timestral/w/5M --out /tmp/timestral/ref-5M.safetensors

Each case `<name>` stores `<name>.context` (B, V, T) plus optional
`<name>.past_cov` / `<name>.known_cov` / `<name>.cat_known`, the horizon in
`<name>.horizon`, and two forecasts: `<name>.raw` (no symmetric averaging, no
clipping) and `<name>.full` (the upstream defaults). The first transformer call of
the `raw` pass is captured too: `<name>.embed` (patch embedding plus forecast
queries), `<name>.hidden` (transformer output after `out_norm`) and `<name>.head`
(normalized monotone quantiles).

`--dtype float64` runs every step in f64 instead (for `--number-format f64`).
Upstream pins float32 in its forecaster (`torch.as_tensor(..., float32)`,
float32 output buffers, `.float()` in RMSNorm), so the model is converted with
`.double()` and the `timestral` modules see float64 wherever they name float32;
`finfo(float32)` thresholds keep their f32 values. Forecasts are then also
stored as `<name>.raw.lo` / `<name>.full.lo`: the f32 tensor plus its `lo`
is the f64 forecast, which keeps the file readable as plain f32.
"""

import argparse
import sys
import types

import numpy as np
import torch
from safetensors.numpy import save_file
from timestral import TimestralForecaster


def cases(rng: np.random.Generator) -> dict[str, dict]:
    out: dict[str, dict] = {}

    out["uni"] = {"context": rng.standard_normal(1000).cumsum()[None, None], "horizon": 96}

    t = np.arange(777, dtype=np.float64)
    out["long"] = {
        "context": (10 * np.sin(2 * np.pi * t / 24) + 0.05 * t + rng.standard_normal(777))[None, None],
        "horizon": 1500,
    }

    multi = rng.standard_normal((2, 3, 300)).cumsum(-1)
    multi[0, 1, :70] = np.nan  # late start: leading patches are left padding
    multi[1, 2, 100:140] = np.nan  # gap inside the context
    multi[0, 0, rng.integers(0, 300, 25)] = np.nan
    out["multi"] = {"context": multi, "horizon": 64}

    t = np.arange(400 + 80, dtype=np.float64)
    known = np.stack([np.cos(2 * np.pi * t / 7), (t % 7 >= 5).astype(np.float64)])[None]
    past = rng.standard_normal((1, 1, 400))
    cat = (t // 3 % 4).astype(np.float64)[None, None]
    cat[0, 0, 10:13] = np.nan
    y = 3 * known[0, 0, :400] + 2 * known[0, 1, :400] + rng.standard_normal(400) * 0.3
    out["cov"] = {
        "context": y[None, None],
        "past_cov": past,
        "known_cov": known,
        "cat_known": cat,
        "horizon": 80,
    }

    # Large level, small variation: where low-precision running sums would show.
    out["level"] = {"context": (1e6 + rng.standard_normal(4000).cumsum())[None, None], "horizon": 200}

    out["nonneg"] = {"context": np.abs(rng.standard_normal((1, 1, 200))) * 5.0, "horizon": 40}
    return out


def force_float64() -> None:
    """Make the `timestral` modules compute in float64 (see the module docs)."""
    real_finfo = torch.finfo

    def finfo(dtype=None):
        # Thresholds written against float32 keep their float32 values.
        return real_finfo(torch.float32 if dtype is torch.float64 else dtype)

    torch64 = types.SimpleNamespace(**{k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
    torch64.float32 = torch.float64
    torch64.float = torch.float64
    torch64.finfo = finfo
    np64 = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
    np64.float32 = np.float64
    for name, mod in list(sys.modules.items()):
        if name.startswith("timestral"):
            if getattr(mod, "torch", None) is torch:
                mod.torch = torch64
            if getattr(mod, "np", None) is np:
                mod.np = np64
    torch.Tensor.float = lambda self, *a, **k: self.to(torch.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = ap.parse_args()
    f64 = args.dtype == "float64"

    torch.manual_seed(0)
    fc = TimestralForecaster.from_pretrained(args.weights)
    if f64:
        force_float64()
        fc.model.double()
    model = fc.model
    tensors: dict[str, np.ndarray] = {}

    captured: dict[str, torch.Tensor] = {}

    def grab(name: str):
        def hook(_m, inputs, output):
            if name not in captured:
                captured[name] = (inputs[0] if name == "embed" else output).detach().clone()
        return hook

    hooks = [
        model.transformer.register_forward_hook(grab("embed")),
        model.transformer.register_forward_hook(grab("hidden")),
        model.head.register_forward_hook(grab("head")),
    ]

    for name, case in cases(np.random.default_rng(7)).items():
        # The file stores inputs as f32; feed exactly those values to the f64 run too.
        for key in ("context", "past_cov", "known_cov", "cat_known"):
            if key in case:
                case[key] = np.asarray(case[key], dtype=np.float32).astype(np.float64)
        kwargs = {}
        if "past_cov" in case:
            kwargs["past_covariates"] = case["past_cov"]
        if "known_cov" in case:
            kwargs["known_covariates"] = case["known_cov"]
        if "cat_known" in case:
            kwargs["categorical_known_covariates"] = case["cat_known"]
        h = case["horizon"]

        captured.clear()
        raw = fc.forecast(case["context"], h, symmetric_averaging=False, clip_nonnegative=False, **kwargs)
        for key in ("embed", "hidden", "head"):
            tensors[f"{name}.{key}"] = captured[key].detach().cpu().numpy().astype(np.float32)
        full = fc.forecast(case["context"], h, **kwargs)

        for key, fcst in (("raw", raw), ("full", full)):
            q = np.asarray(fcst.quantiles.detach().cpu().numpy(), dtype=np.float64)
            tensors[f"{name}.{key}"] = q.astype(np.float32)
            if f64:
                tensors[f"{name}.{key}.lo"] = (q - q.astype(np.float32)).astype(np.float32)
        tensors[f"{name}.horizon"] = np.array([h], dtype=np.float32)
        for key in ("context", "past_cov", "known_cov", "cat_known"):
            if key in case:
                tensors[f"{name}.{key}"] = np.asarray(case[key], dtype=np.float32)
        print(f"{name}: raw {tuple(raw.quantiles.shape)} median[:4] {raw.median.flatten()[:4].tolist()}")

    for hk in hooks:
        hk.remove()
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, args.out)
    print(f"wrote {len(tensors)} tensors to {args.out}")


if __name__ == "__main__":
    main()
