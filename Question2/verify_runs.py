"""Check that every training run is a complete 30-epoch run on the GPU with the expected parameter counts.
Usage: python verify_runs.py [runs_dir]   (exit code 1 if anything is wrong)"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
runs = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs")
EXPECT = {"A": 480784, "B": 480784, "C": 480784}  # D's count depends on the chosen design (must be <= 721176)
bad = 0
for m in ("A", "B", "C", "D"):
    for s in (0, 1):
        p = os.path.join(runs, f"{m}_s{s}.json")
        if not os.path.exists(p):
            print("MISSING", p); bad += 1; continue
        d = json.load(open(p))
        ok = (d.get("epochs") == 30 and len(d.get("epoch_seconds", [])) == 30 and "val200_E5_1x" in d and not d.get("smoke")
              and d.get("device") == "cuda" and d.get("batch_size") == 32 and d.get("lr") == 1e-3
              and (d["params"] == EXPECT[m] if m in EXPECT else d["params"] <= 721176))
        mean = sum(d["epoch_seconds"]) / max(1, len(d["epoch_seconds"]))
        print(("OK     " if ok else "PROBLEM"), f"{m} seed {s}: epochs {d.get('epochs')}/{len(d.get('epoch_seconds', []))} params {d['params']} "
              f"device {d.get('device')} ({d.get('hardware')}) mean epoch {mean:.1f}s final loss {d['train_loss'][-1]:.5f}")
        bad += not ok
print("ALL RUNS OK" if not bad else f"{bad} PROBLEM(S)")
sys.exit(1 if bad else 0)
