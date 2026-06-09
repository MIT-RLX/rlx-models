#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#
"""Full SAM3 inference backend used by the Rust `Sam3` wrapper.

This script intentionally delegates the full detector/tracker stack to the
official `facebookresearch/sam3` package. It is the complete execution path
for SAM3 image and base video while native RLX kernels are brought up.
"""

import json
import os
import sys
import tempfile

import numpy as np


def env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        print(f"missing env var: {name}", file=sys.stderr)
        sys.exit(2)
    return value


def load_rgb() -> "np.ndarray":
    h = int(env("RLX_SAM3_INPUT_H"))
    w = int(env("RLX_SAM3_INPUT_W"))
    data = np.fromfile(env("RLX_SAM3_IMAGE_U8_BIN"), dtype=np.uint8)
    expected = h * w * 3
    if data.size != expected:
        raise ValueError(f"image expected {expected} bytes, got {data.size}")
    return data.reshape(h, w, 3)


def tensor_to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def write_array(out_dir: str, name: str, value) -> list[int]:
    arr = tensor_to_numpy(value)
    if arr is None:
        arr = np.zeros((0,), dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    arr.tofile(os.path.join(out_dir, f"{name}.f32"))
    return list(arr.shape)


def run_image(weights: str, device: str, rgb: "np.ndarray", out_dir: str) -> dict:
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    prompt = os.environ.get("RLX_SAM3_TEXT_PROMPT") or "object"
    with torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
        model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            checkpoint_path=weights,
            load_from_HF=False,
            enable_inst_interactivity=False,
            compile=False,
        )
        processor = Sam3Processor(model)
        state = processor.set_image(Image.fromarray(rgb))
        result = processor.set_text_prompt(state=state, prompt=prompt)

    masks_shape = write_array(out_dir, "masks", result.get("masks"))
    boxes_shape = write_array(out_dir, "boxes", result.get("boxes"))
    scores_shape = write_array(out_dir, "scores", result.get("scores"))
    return {
        "mode": "image",
        "masks_shape": masks_shape,
        "boxes_shape": boxes_shape,
        "scores_shape": scores_shape,
    }


def run_video(weights: str, device: str, rgb: "np.ndarray", out_dir: str) -> dict:
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_video_predictor

    prompt = os.environ.get("RLX_SAM3_TEXT_PROMPT") or "object"
    with tempfile.TemporaryDirectory(prefix="rlx_sam3_video_") as frames_dir:
        # The public video predictor accepts a folder of frames.
        frame_path = os.path.join(frames_dir, "00000.jpg")
        Image.fromarray(rgb).save(frame_path)
        with torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
            predictor = build_sam3_video_predictor(
                checkpoint_path=weights,
                load_from_HF=False,
                device=device,
                compile=False,
            )
            response = predictor.handle_request(
                request={"type": "start_session", "resource_path": frames_dir}
            )
            session_id = response.get("session_id")
            response = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                }
            )

    outputs = response.get("outputs", response)
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
    elif isinstance(outputs, dict):
        first = outputs
    else:
        first = {}
    masks_shape = write_array(out_dir, "masks", first.get("masks"))
    boxes_shape = write_array(out_dir, "boxes", first.get("boxes"))
    scores_shape = write_array(out_dir, "scores", first.get("scores"))
    return {
        "mode": "video",
        "masks_shape": masks_shape,
        "boxes_shape": boxes_shape,
        "scores_shape": scores_shape,
    }


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"torch must be installed for SAM3 full backend: {exc}", file=sys.stderr)
        return 3

    weights = env("RLX_SAM3_WEIGHTS")
    out_dir = env("RLX_SAM3_OUT_DIR")
    mode = os.environ.get("RLX_SAM3_MODE", "image")
    device = os.environ.get("RLX_SAM3_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("RLX_SAM3_DEVICE=cuda but CUDA is unavailable; falling back to CPU", file=sys.stderr)
        device = "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.makedirs(out_dir, exist_ok=True)
    rgb = load_rgb()
    if mode == "video":
        meta = run_video(weights, device, rgb, out_dir)
    elif mode == "image":
        meta = run_image(weights, device, rgb, out_dir)
    else:
        raise ValueError(f"unknown RLX_SAM3_MODE={mode!r}")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
