#!/usr/bin/env python3
"""Reference cases for rlx-flux3's port of BFL `flux3/bbox.py`.

Runs the reference module (from a checkout of
github.com/black-forest-labs/flux-3-image-klein, --bfl-repo) on the box
tables and captions in its `examples/`, plus perturbed upsampler answers
(renamed ids, shifted boxes, malformed rows), and writes inputs and outputs
as JSON for `crates/rlx-flux3/tests/bbox_cases.rs`.
"""

import argparse
import copy
import glob
import json
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfl-repo", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, os.path.join(a.bfl_repo, "src"))
    from flux3 import bbox

    records = []
    for f in sorted(glob.glob(os.path.join(a.bfl_repo, "examples", "**", "*.json"), recursive=True)):
        r = json.load(open(f))
        if isinstance(r, dict) and "prompt" in r:
            records.append((os.path.relpath(f, a.bfl_repo), r))

    cases = {"to_canvas": [], "detect": [], "plan": [], "reconcile": []}
    box_sets = [r["boxes"] for _, r in records if r.get("boxes")]
    box_sets.append([{"bbox": [0.0005, 0.2, 1.3, 0.5], "desc": "edge"}, {"id": "b", "bbox": [0.1, 0.1, 0.2, 0.2]}])
    box_sets.append([{"id": "m", "from": "ref_image_0", "src_bbox": [0.1, 0.1, 0.5, 0.5], "tgt_bbox": None, "desc": "gone"}])
    for boxes in box_sets:
        cases["to_canvas"].append({"boxes": boxes, "rows": json.dumps(bbox.to_canvas(boxes), ensure_ascii=False)})

    prompts = [r["prompt"] for _, r in records] + [r["input_prompt"] for _, r in records if "input_prompt" in r]
    prompts += ["A sign that says [SALE] in red.", "No boxes at all.",
                'ORIGINAL PROMPT:\nhello\n\nLOCKED BOUNDING-BOX ANNOTATIONS (x):\n[{"id": "a", "bbox": [10, 10, 500, 500], "desc": "d"}]']
    for text in prompts:
        found = bbox.detect(text)
        cases["detect"].append({
            "prompt": text,
            "detected": None if found is None else [found[0], json.dumps(found[1], ensure_ascii=False)],
            "finished": bbox.is_finished(text),
        })

    for _, r in records:
        for n in (0, 1):
            boxes = r.get("boxes") or None
            for text in (r.get("input_prompt", r["prompt"]), r["prompt"]):
                task, user, locked = bbox.plan(text, boxes, n)
                cases["plan"].append({
                    "prompt": text, "boxes": boxes, "n_images": n, "task": task, "text": user,
                    "locked": None if locked is None else json.dumps(locked, ensure_ascii=False),
                })

    for _, r in records:
        if not r.get("boxes"):
            continue
        locked = bbox.to_canvas(r["boxes"])
        caption, rows = bbox.split(r["prompt"])
        answers = [r["prompt"]]
        if rows:
            # The upsampler re-planned a locked box under its own id, nudged it,
            # added a row of its own, and a malformed one.
            moved = copy.deepcopy(rows)
            text = caption
            for row in moved:
                for k in ("bbox", "tgt_bbox"):
                    if isinstance(row.get(k), list):
                        row[k] = [v + 3 for v in row[k]]
                        old, row["id"] = row["id"], row["id"] + "_x"
                        text = text.replace(f"<{old}>", f"<{row['id']}>")
                        break
            moved.append({"id": "extra_1", "bbox": [5, 5, 100, 100], "desc": "  added  "})
            moved.append({"id": "bad", "bbox": [0.1, 0.1, 0.2, 0.2], "desc": "0-1 coords"})
            answers.append(f"{text} <extra_1> <bad> {json.dumps(moved, ensure_ascii=False)}")
        answers.append("A caption with no table.")
        for ans in answers:
            cases["reconcile"].append({"text": ans, "locked": json.dumps(locked, ensure_ascii=False),
                                       "out": bbox.reconcile(ans, copy.deepcopy(locked))})

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(cases, open(a.out, "w"), ensure_ascii=False, indent=1)
    print({k: len(v) for k, v in cases.items()})


if __name__ == "__main__":
    main()
