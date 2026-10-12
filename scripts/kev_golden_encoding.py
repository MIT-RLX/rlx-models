"""Regenerate crates/rlx-kev/tests/fixtures/golden_encoding.json.

The encode/render logic below is lifted VERBATIM from the reference
implementation (jaredpalmer/kev: kev/model.py::encode, kev/api.py::to_record
and render) so the fixture is a statement about the reference, not about our
port. Driven by a real Qwen3.5 tokenizer.json.

    pip install tokenizers==0.22
    python3 scripts/kev_golden_encoding.py <run_dir> [out.json]

<run_dir> is a kev checkpoint directory containing tokenizer.json (e.g. a
snapshot of jaredpalmer/kev-4b). The Rust side checks itself against the
fixture in crates/rlx-kev/tests/real_checkpoint.rs.
"""
import json, re, sys
from tokenizers import Tokenizer

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "kev-4b"
tok = Tokenizer.from_file(f"{RUN_DIR}/tokenizer.json")

# ---- verbatim from kev/model.py -------------------------------------------
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
MAX_STATE, MAX_BRANCH = 384, 1024
OPT_NONE, OPT_DECIDE = -1, -2
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

def user_tokens(text):
    return tok.encode(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).ids

def tid(t):
    i = tok.token_to_id(t)
    assert i is not None, t
    return i

def encode(rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, option_isolation=False):
    state_tokens = user_tokens(rec["state"])
    S = [tid(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids, seg, pos, opt = list(S), [0]*len(S), list(range(len(S))), [OPT_NONE]*len(S)
    q_id, o_id, c_id, d_id = (tid(t) for t in SPECIAL[1:])
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(q["instr"])
        spans = [[o_id] + user_tokens(o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        assert len(br) <= max_branch - len(S), "branch too long"
        base = len(ids); p0 = len(S)
        br_opt = [OPT_NONE]*len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        if option_isolation:
            longest = max(len(sp) for sp in spans)
            br_pos = list(range(p0, p0+len(instr))) + [p0+len(instr)+i for sp in spans for i in range(len(sp))] + [p0+len(instr)+longest]
        else:
            br_pos = list(range(p0, p0+len(br)))
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp); ends.append(cursor-1)
        ids += br; seg += [k]*len(br); pos += br_pos; opt += br_opt
        decide_idx.append(base+len(br)-1); opt_idx.append([base+e for e in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt,
            "decide_idx": decide_idx, "opt_idx": opt_idx,
            "state_truncated": len(state_tokens)+1 > max_state}

# ---- verbatim from kev/api.py ---------------------------------------------
def render(v, indent=0):
    pad = "  " * indent
    if v is None: return ""
    if isinstance(v, (str, int, float, bool)): return str(v)
    if isinstance(v, list): return "\n".join(f"{pad}- {render(x, indent+1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent+1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}" for k, x in v.items())

def option_text(name, desc):
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"

def to_record(req):
    qs = []
    for qid, q in req["questions"].items():
        if q["type"] == "noul":
            c = q.get("criteria") or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q["type"] == "choice":
            opts = [option_text(k, v) for k, v in q["criteria"].items()]
        else:
            opts = [render(x) for x in q["criteria"]]
        qs.append({"instr": render(q["instructions"]), "options": opts})
    return {"state": render(req["state"]), "questions": qs}

OUT = sys.argv[2] if len(sys.argv) > 2 else "crates/rlx-kev/tests/fixtures/golden_encoding.json"
fixture = json.load(open(OUT))
out = encode(to_record(fixture["request"]), max_state=8192, max_branch=8192)
fixture["expected"] = out
json.dump(fixture, open(OUT, "w"), indent=1)
print("wrote", OUT, "ids:", len(out["ids"]))
