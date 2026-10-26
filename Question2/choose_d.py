"""Pick model D's design from the VALIDATION-only screening results (select_d.py) and write results/d_choice.json.
Score = E5@1x + E2@2x + E2@3x on the validation split (lower is better). Usage: python choose_d.py"""
import json, os
from select_d_cfg import CANDS

HERE = os.path.dirname(os.path.abspath(__file__))
res = {}
for k in CANDS:
    p = os.path.join(HERE, "results", f"select_d_{k}.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}: run  python select_d.py {k} STEPS  first")
    res[k] = json.load(open(p))
best = min(res, key=lambda k: res[k]["score"])
out = {**CANDS[best], "chosen": best, "scores": {k: res[k]["score"] for k in res}, "screening_steps": res[best]["steps"]}
json.dump(out, open(os.path.join(HERE, "results", "d_choice.json"), "w"), indent=1)
print("model D design (validation only):", best, out["scores"])
