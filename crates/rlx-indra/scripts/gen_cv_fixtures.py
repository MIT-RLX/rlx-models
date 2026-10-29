"""Write OpenCV reference outputs for tests/imgproc_cv_reference.rs.

    python crates/rlx-indra/scripts/gen_cv_fixtures.py <page.png> <out-dir>

Each case is <name>.in.u8 / <name>.out.u8 (raw grayscale) plus a line in
cases.json with the call and the shapes.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main(page: str, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(page, cv2.IMREAD_GRAYSCALE)
    src = np.ascontiguousarray(gray[: min(gray.shape[0], 700), : min(gray.shape[1], 900)])
    small = np.ascontiguousarray(src[100:160, 50:300])
    cases = []

    def add(name, img, res, call):
        (out_dir / f"{name}.in.u8").write_bytes(img.tobytes())
        (out_dir / f"{name}.out.u8").write_bytes(np.ascontiguousarray(res).tobytes())
        cases.append({"name": name, "in": [img.shape[1], img.shape[0]],
                      "out": [res.shape[1], res.shape[0]], **call})

    for i, (w, h) in enumerate([(611, 433), (450, 350), (300, 233), (1201, 977), (37, 23), (900, 700)]):
        for interp, name in ((cv2.INTER_LINEAR, "linear"), (cv2.INTER_AREA, "area")):
            add(f"resize_{name}_{i}", src, cv2.resize(src, (w, h), interpolation=interp),
                {"op": "resize", "interp": name, "size": [w, h]})
    for i, (w, h) in enumerate([(125, 30), (83, 32), (400, 96), (250, 32), (17, 9)]):
        for interp, name in ((cv2.INTER_LINEAR, "linear"), (cv2.INTER_AREA, "area")):
            add(f"resize_small_{name}_{i}", small, cv2.resize(small, (w, h), interpolation=interp),
                {"op": "resize", "interp": name, "size": [w, h]})
    for i, a in enumerate([0.1, -0.3, 1.7, -12.5, 33.0, 45.0, 0.5]):
        hh, ww = src.shape
        M = cv2.getRotationMatrix2D((ww / 2, hh / 2), a, 1.0)
        add(f"rotate_rep_{i}", src,
            cv2.warpAffine(src, M, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
            {"op": "rotate", "degrees": a, "border": "replicate"})
        ink = 255 - src
        add(f"rotate_const_{i}", ink, cv2.warpAffine(ink, M, (ww, hh)),
            {"op": "rotate", "degrees": a, "border": "constant"})
    (out_dir / "cases.json").write_text(json.dumps(cases, indent=1))
    print(len(cases), "cases ->", out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:3])
