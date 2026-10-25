"""Runs model-D screening (validation only), then trains B, C, D (seeds 0, 1) with at most 4 concurrent CPU jobs.
Usage: python schedule.py EPOCHS [SCREEN_STEPS]"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = sys.argv[1] if len(sys.argv) > 1 else "6"
STEPS = sys.argv[2] if len(sys.argv) > 2 else "250"
MAXP, THREADS = 4, "3"
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)


def job(cmd, log):
    return {"cmd": cmd, "log": os.path.join(HERE, "runs", log)}


screens = ["p8_k3", "p4_k5_h32"]
queue = [job(["select_d.py", s, STEPS, THREADS], f"select_{s}.log") for s in screens]
queue += [job(["train.py", "--model", "B", "--seed", str(s), "--epochs", EPOCHS, "--threads", THREADS], f"B_s{s}.log") for s in (0, 1)]
queue += [job(["train.py", "--model", "C", "--seed", str(s), "--epochs", EPOCHS, "--threads", THREADS], f"C_s{s}.log") for s in (0, 1)]
running, decided = [], False
while queue or running:
    running = [(p, j) for p, j in running if p.poll() is None]
    if not decided and all(os.path.exists(os.path.join(HERE, "results", f"select_d_{s}.json")) for s in screens):
        res = {s: json.load(open(os.path.join(HERE, "results", f"select_d_{s}.json"))) for s in screens}
        best = min(res, key=lambda s: res[s]["score"])
        sys.path.insert(0, HERE)
        import select_d_cfg  # noqa  (tiny module with the candidate table)
        cfg = select_d_cfg.CANDS[best]
        json.dump({**cfg, "chosen": best, "scores": {s: res[s]["score"] for s in screens}}, open(os.path.join(HERE, "results", "d_choice.json"), "w"), indent=1)
        print("model D design chosen on validation:", best, {s: res[s]["score"] for s in screens}, flush=True)
        queue = [job(["train.py", "--model", "D", "--seed", str(s), "--epochs", EPOCHS, "--threads", THREADS], f"D_s{s}.log") for s in (0, 1)] + queue
        decided = True
    while queue and len(running) < MAXP:
        j = queue.pop(0)
        print(time.strftime("%H:%M:%S"), "start", " ".join(j["cmd"]), flush=True)
        running.append((subprocess.Popen([sys.executable] + j["cmd"], cwd=HERE, stdout=open(j["log"], "w"), stderr=subprocess.STDOUT), j))
    time.sleep(10)
print("all jobs finished", flush=True)
