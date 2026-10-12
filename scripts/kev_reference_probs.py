"""Reference probabilities from kev's OWN code (kev/model.py + kev/checkpoint.py)."""
import json, sys, torch
sys.path.insert(0, ".")
from kev.checkpoint import Checkpoint, LoadOptions
from kev.api import to_record

run_dir, base_dir, req_path, out_path = sys.argv[1:5]

# kev resolves `base` from head.pt as a Hub id; point it at the local snapshot.
import kev.checkpoint as ck
_orig = ck.read_meta
def patched(run):
    m = _orig(run)
    m.base, m.base_revision = base_dir, None
    return m
ck.read_meta = patched

c = Checkpoint(run_dir)
tok, model = c.load("cpu", LoadOptions(dtype=torch.float32, merge=True))
print("base:", c.meta.base, "temperature:", model.head.temperature, "hybrid:", model.hybrid, flush=True)

req = json.load(open(req_path))
rec, meta = to_record(type("R", (), {
    "state": req["state"], "model": req.get("model", "kev-latest"),
    "questions": type("Q", (), {"items": lambda self: req["questions"].items()})()
})()) if False else (None, None)

# to_record wants the pydantic model; build the record directly instead.
from kev.api import render, option_text
qs, meta = [], []
for qid, q in req["questions"].items():
    keys = (list(q["criteria"]) if q["type"] == "choice"
            else ["false", "true"] if q["type"] == "noul"
            else [str(i) for i in range(len(q["criteria"]))])
    if q["type"] == "noul":
        cr = q.get("criteria") or {}
        opts = [option_text("no", cr.get("false")), option_text("yes", cr.get("true"))]
    elif q["type"] == "choice":
        opts = [option_text(k, v) for k, v in q["criteria"].items()]
    else:
        opts = [render(x) for x in q["criteria"]]
    qs.append({"instr": render(q["instructions"]), "options": opts, "label": 0})
    meta.append({"id": qid, "type": q["type"], "keys": keys})
rec = {"state": render(req["state"]), "questions": qs}

enc = model.encode(tok, rec, max_state=8192, max_branch=8192)
print("ids:", len(enc["ids"]), "questions:", len(enc["decide_idx"]), flush=True)
probs = [p.tolist() for p in model.probs(enc)]
json.dump({"ids": enc["ids"], "probs": probs,
           "meta": [{"id": m["id"], "keys": m["keys"]} for m in meta],
           "temperature": float(model.head.temperature)},
          open(out_path, "w"), indent=1)
for m, p in zip(meta, probs):
    print(f"  {m['id']:12} " + "  ".join(f"{k}={v:.4f}" for k, v in zip(m["keys"], p)))
