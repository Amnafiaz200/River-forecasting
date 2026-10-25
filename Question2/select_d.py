"""Choose model D's design on the VALIDATION split only (never the test split).
Each candidate is trained with the speed-augmented protocol (seed 0, same optimiser) for a short budget and scored on validation:
score = E(5)@1x + E(2)@2x + E(2)@3x  (lower is better).  Usage: python select_d.py CANDIDATE STEPS"""
import json, os, sys, time
import torch
import mm

HERE = os.path.dirname(os.path.abspath(__file__))
from select_d_cfg import CANDS
name, steps = sys.argv[1], int(sys.argv[2])
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
allv = mm.load_videos(os.path.join(HERE, "data"))
tr, va = allv[mm.TRAIN], mm.to_float(allv[mm.VAL])
torch.manual_seed(0)
model = mm.ConvLSTMForecaster(**CANDS[name]).to(mm.DEVICE, memory_format=torch.channels_last)
opt = torch.optim.Adam(model.parameters(), 1e-3)
gen = torch.Generator().manual_seed(1000)
import train as T
t0, step = time.perf_counter(), 0
while step < steps:
    perm = torch.randperm(len(tr), generator=gen)
    for i in range(0, len(perm), 32):
        if step >= steps: break
        X, Y, M = T.sample_batch(tr, perm[i:i+32], "D", gen)
        X, Y, M = X.to(mm.DEVICE), Y.to(mm.DEVICE), M.to(mm.DEVICE)
        opt.zero_grad(); mm.masked_loss(model(X), Y, M).backward(); opt.step(); step += 1
model.eval()
out = {"candidate": name, "params": mm.n_params(model), "steps": steps, "seconds": time.perf_counter() - t0}
with torch.no_grad():
    for k in (1, 2, 3):
        X, Y, M = mm.make_eval(va, k, 1)
        mse = mm.per_horizon_mse(mm.predict(model, X), Y)
        out[f"E5_{k}x"] = mse.mean().item(); out[f"E2_{k}x"] = mse[:, :2].mean().item()
out["score"] = out["E5_1x"] + out["E2_2x"] + out["E2_3x"]
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", f"select_d_{name}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
