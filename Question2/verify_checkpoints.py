"""Check the saved final checkpoints against their training logs and against the reported metrics. NO training.

For every model A, B, C, D and seed 0, 1 it checks that
  * runs/<M>_s<seed>.json is a complete 30-epoch, non-smoke, cuda run (batch 32, lr 1e-3),
  * runs/<M>_s<seed>.pt loads into the model built by mm.build_model, with the parameter count recorded in the log
    (A, B, C: 480,784; D: <= 721,176 and the d_config of the log equals results/d_choice.json),
  * re-evaluating the checkpoint on the 1,000 test videos (1x, t = 1) gives the E(5) stored in results/metrics_<M>_<seed>.json
    (to a relative difference of 1e-3; CPU and GPU arithmetic differ in the last digits).
With --sym the symmetrised B and D are also checked on the first 100 test videos (C_M and C_R must be ~1e-16).
With --manifest, results/checkpoint_manifest.json (SHA-256, size, epochs, parameters, hardware of every checkpoint) is written.

  python verify_checkpoints.py [--sym] [--manifest]        (a GPU is not needed: ALLOW_CPU=1 is set automatically; inference only)
Exit code 1 if anything does not match.
"""
import hashlib, json, os, sys
os.environ.setdefault("ALLOW_CPU", "1")
import torch
import mm

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, RES = os.path.join(HERE, "runs"), os.path.join(HERE, "results")
problems = []


def bad(msg):
    problems.append(msg); print("PROBLEM ", msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    do_sym, do_manifest = "--sym" in sys.argv, "--manifest" in sys.argv
    V = mm.to_float(mm.load_videos(os.path.join(HERE, "data"))[mm.TEST])
    assert len(V) == 1000
    X, Y, _ = mm.make_eval(V, 1, 1)
    dchoice = json.load(open(os.path.join(RES, "d_choice.json")))
    manifest = {}
    for name in "ABCD":
        for seed in (0, 1):
            tag = f"{name}_s{seed}"
            log = json.load(open(os.path.join(RUNS, tag + ".json")))
            pt = os.path.join(RUNS, tag + ".pt")
            ok = (log["epochs"] == 30 and len(log["epoch_seconds"]) == 30 and not log.get("smoke") and log["device"] == "cuda"
                  and log["batch_size"] == 32 and log["lr"] == 1e-3 and log["model"] == name and log["seed"] == seed)
            if not ok:
                bad(f"{tag}: log is not a complete 30-epoch cuda run with batch 32 / lr 1e-3")
            model = mm.build_model(name)
            model.load_state_dict(torch.load(pt, map_location="cpu"))
            model.eval()
            n = mm.n_params(model)
            if n != log["params"] or (name != "D" and n != 480784) or (name == "D" and n > 721176):
                bad(f"{tag}: parameter count {n} (log {log['params']})")
            if name == "D":
                cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in log["d_config"].items()}
                want = {k: dchoice[k] for k in ("patch", "hidden", "kernels", "dilations")}
                if cfg != want:
                    bad(f"{tag}: d_config {cfg} differs from results/d_choice.json {want}")
            P = mm.predict(model, X)
            E5 = mm.per_horizon_mse(P, Y).mean().item()
            ref = json.load(open(os.path.join(RES, f"metrics_{name}_{seed}.json")))["E5_1x"]
            flag = "OK " if abs(E5 - ref) <= 1e-3 * ref else "BAD"
            if flag == "BAD":
                bad(f"{tag}: recomputed E5 {E5:.6f} vs reported {ref:.6f}")
            line = f"{flag} {tag}: {log['epochs']} epochs, {n:,} params, {os.path.getsize(pt)/1e6:.2f} MB, recomputed E5(1x) {E5:.4f} vs reported {ref:.4f}"
            if do_sym and name in "BD":  # symmetrised check on the first 100 test videos only (8 forward passes per transformation)
                f = mm.symmetrised(model); Xs, Ys = X[:100], Y[:100]
                Ps = mm.predict(f, Xs)
                CM = mm.per_horizon_mse(mm.predict(f, mm.mirror(Xs)), mm.mirror(Ps)).mean().item()
                CR = mm.per_horizon_mse(mm.predict(f, mm.rot(Xs, 1)), mm.rot(Ps, 1)).mean().item()
                if CM > 1e-12 or CR > 1e-12:
                    bad(f"{tag} sym: C_M {CM:.2e}, C_R {CR:.2e} are not ~0")
                line += f" | symmetrised (first 100 test videos): C_M {CM:.1e}, C_R {CR:.1e}"
            print(line, flush=True)
            manifest[tag] = {"file": f"runs/{tag}.pt", "sha256": sha256(pt), "bytes": os.path.getsize(pt), "epochs": log["epochs"],
                             "params": n, "seed": seed, "hardware": log["hardware"], "reported_E5_1x": ref}
    if do_manifest:
        json.dump(manifest, open(os.path.join(RES, "checkpoint_manifest.json"), "w"), indent=1)
        print("wrote results/checkpoint_manifest.json")
    print("ALL CHECKPOINTS MATCH THEIR LOGS AND REPORTED METRICS" if not problems else f"{len(problems)} PROBLEM(S)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
