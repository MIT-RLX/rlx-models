#!/usr/bin/env python3
"""Activation comparison framework: compare tsac-ng decoder layer outputs
against libnc GDB-captured references. Pinpoints first divergence layer."""

import numpy as np
import json, os, sys, glob, struct
from pathlib import Path

def load_activation(path, n=None, dtype=np.float32):
    """Load n floats from a binary .bin file. Returns None if not found."""
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=dtype)
    if n is not None and len(data) != n:
        print(f"  [WARN] {path}: expected {n} elements, got {len(data)}")
    return data

def compute_metrics(ours, reference):
    """Compute correlation, MAE, RMSE, max_abs_diff, SNR_dB."""
    ours = np.nan_to_num(ours).flatten()
    reference = np.nan_to_num(reference).flatten()
    n = min(len(ours), len(reference))
    ours, reference = ours[:n], reference[:n]
    corr = np.corrcoef(ours, reference)[0, 1] if n > 1 else 0.0
    mae = np.mean(np.abs(ours - reference))
    rmse = np.sqrt(np.mean((ours - reference) ** 2))
    max_diff = np.max(np.abs(ours - reference))
    ref_rms = np.sqrt(np.mean(reference ** 2))
    snr_db = 20 * np.log10(ref_rms / (rmse + 1e-30))
    return {"correlation": float(corr), "mae": float(mae), "rmse": float(rmse),
            "max_abs_diff": float(max_diff), "snr_db": float(snr_db)}

def load_libnc_inproj(codebook_idx):
    """Load libnc in_proj capture for given codebook."""
    path = f"docs/evidence/libnc_inproj_cb{codebook_idx}_f32.bin"
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.float32)
    if len(data) == 8192:
        return data.reshape(1024, 8)
    return data

def load_gdb_model0_input():
    """Load GDB-captured model.0 input (RVQ output)."""
    path = "docs/evidence/gdb_model0_input_9216f32.bin"
    data = load_activation(path)
    if data is not None and len(data) == 9216:
        return data.reshape(1024, 9)
    return data

def load_our_activation(layer_name):
    """Load our decoder's DUMP_ACT output from /tmp/act_{name}.bin."""
    return load_activation(f"/tmp/act_{layer_name}.bin")

def generate_heatmap(metrics_dict, output_path="experimental/correlation_heatmap.png"):
    """Generate heatmap of metrics across layers."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed — heatmap skipped")
        return
    layer_names = list(metrics_dict.keys())
    short_names = {"rvq_out": "RVQ", "m0_conv1d": "M0", "block1_convt": "B1C",
                   "block2_convt": "B2C", "block3_convt": "B3C",
                   "block4_convt": "B4C", "m6_pre_tanh": "M6"}
    labels = [short_names.get(n, n) for n in layer_names]
    metrics_keys = ["correlation", "mae", "rmse", "snr_db"]
    data = np.zeros((len(metrics_keys), len(layer_names)))
    for j, name in enumerate(layer_names):
        for i, key in enumerate(metrics_keys):
            data[i, j] = metrics_dict[name].get(key, 0.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap='RdYlGn' if len(layer_names) > 0 else 'viridis', aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(metrics_keys)))
    ax.set_yticklabels([k.capitalize() for k in metrics_keys])
    for i in range(len(metrics_keys)):
        for j in range(len(layer_names)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Heatmap saved to {output_path}")

def generate_scatter(ours, reference, layer_name, max_points=10000, output_dir="experimental"):
    """Generate scatter plot of ours vs reference."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ours_f = ours.flatten()
    ref_f = reference.flatten()
    n = min(len(ours_f), len(ref_f))
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        ours_f, ref_f = ours_f[idx], ref_f[idx]
    corr = np.corrcoef(ours_f, ref_f)[0, 1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ref_f, ours_f, s=1, alpha=0.5)
    lims = [min(ref_f.min(), ours_f.min()), max(ref_f.max(), ours_f.max())]
    ax.plot(lims, lims, 'r--', alpha=0.8)
    ax.set_xlabel("Reference")
    ax.set_ylabel("Ours")
    ax.set_title(f"{layer_name}: corr={corr:.4f}")
    plt.tight_layout()
    path = f"{output_dir}/scatter_{layer_name}.png"
    plt.savefig(path, dpi=150)
    print(f"  Scatter saved to {path}")

def main():
    LAYERS = [
        ("rvq_out", (1024, 16), "gdb_model0_input_9216f32.bin"),
        ("m0_conv1d", (1536, 16), None),
        ("block1_convt", (768, 128), None),
        ("block2_convt", (384, 512), None),
        ("block3_convt", (192, 1024), None),
        ("block4_convt", (96, 2048), None),
        ("m6_pre_tanh", (2, 4096), None),
    ]
    os.makedirs("experimental", exist_ok=True)
    metrics_dict = {}
    first_divergence = None
    print("=" * 60)
    print("Activation Comparison Report")
    print("=" * 60)
    for name, shape, gdb_src in LAYERS:
        our = load_our_activation(name)
        ref = None
        if gdb_src and our is not None:
            ref = load_activation(f"docs/evidence/{gdb_src}")
        if our is None:
            print(f"  [SKIP] {name}: no our activation dump (run decoder with DEBUG_DECODER=1)")
            continue
        if ref is None and gdb_src:
            print(f"  [SKIP] {name}: no GDB reference at docs/evidence/{gdb_src}")
            continue
        if ref is None and gdb_src is None:
            print(f"  [INFO] {name}: no GDB reference yet — computing stats only")
            metrics_dict[name] = {"correlation": None, "mae": None, "rmse": None,
                                  "max_abs_diff": None, "snr_db": None,
                                  "shape": list(our.shape),
                                  "rms": float(np.sqrt(np.mean(our ** 2))),
                                  "max_abs": float(np.max(np.abs(our)))}
            print(f"    ours: shape={our.shape} rms={metrics_dict[name]['rms']:.4f}")
            continue
        metrics = compute_metrics(our, ref)
        metrics["shape"] = list(our.shape)
        metrics_dict[name] = metrics
        corr_str = f"{metrics['correlation']:.4f}" if metrics['correlation'] is not None else "N/A"
        print(f"  {name:20s} corr={corr_str} RMSE={metrics['rmse']:.6f} SNR={metrics['snr_db']:.1f}dB")
        if metrics['correlation'] is not None and metrics['correlation'] < 0.95 and first_divergence is None:
            first_divergence = name
        # Generate scatter if we have reference
        if ref is not None:
            generate_scatter(our, ref, name)
    # JSON report
    report = {
        "date": "2026-05-29",
        "n_layers": len(metrics_dict),
        "layers": metrics_dict,
        "first_divergence_layer": first_divergence,
        "first_divergence_at_correlation_lt": 0.95,
        "recommendation": f"First divergence at {first_divergence}" if first_divergence else "No divergence detected."
    }
    with open("experimental/activation_comparison.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to experimental/activation_comparison.json")
    if first_divergence:
        print(f"⚠ First divergence at: {first_divergence}")
    generate_heatmap(metrics_dict)
    print("=" * 60)

if __name__ == "__main__":
    main()
