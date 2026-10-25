"""Train one model. Protocol (handout, 'Training and evaluation'):
Adam lr 1e-3 (no schedule), batch 32, one random window per training video per epoch (shuffled),
loss = squared error averaged over pixels + available target frames per video, then over the batch.
No teacher forcing. Final-epoch weights are kept.

  python train.py --model A --seed 0 --epochs 30
Models: A (1x, no sym), B (1x, symmetry aug), C (speed aug), D (speed aug, larger reach), DB (D + symmetry aug).
"""
import argparse, json, os, platform, time
import torch
import mm

HERE = os.path.dirname(os.path.abspath(__file__))

SPEED_AUG = {"A": False, "B": False, "C": True, "D": True, "DB": True}
SYM_AUG = {"A": False, "B": True, "C": False, "D": False, "DB": True}


def sample_batch(videos, idx, model_name, gen):
    B = len(idx)
    ks, ts, gs = [], [], []
    for _ in range(B):
        if SPEED_AUG[model_name]:
            k = int(torch.randint(1, 4, (1,), generator=gen))
        else:
            k = 1
        starts = mm.allowed_starts(k, SPEED_AUG[model_name])
        ts.append(starts[int(torch.randint(len(starts), (1,), generator=gen))])
        ks.append(k)
        gs.append(int(torch.randint(8, (1,), generator=gen)) if SYM_AUG[model_name] else 0)
    X, Y, M = mm.batch_windows(videos[idx], ts, ks)
    if SYM_AUG[model_name]:  # same g for input and target
        for b in range(B):
            if gs[b] != 0:
                X[b] = mm.g_apply(X[b], gs[b])
                Y[b] = mm.g_apply(Y[b], gs[b])
    return X, Y, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPEED_AUG))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)

    videos = mm.load_videos(os.path.join(HERE, "data"))[mm.TRAIN]  # uint8
    val = mm.load_videos(os.path.join(HERE, "data"))[mm.VAL]

    torch.manual_seed(args.seed)  # initialisation
    model = mm.build_model(args.model).to(mm.DEVICE, memory_format=torch.channels_last)  # same maths, ~12% faster on CPU
    gen = torch.Generator().manual_seed(1000 + args.seed)  # shuffling, windows, augmentation
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tag = f"{args.model}_s{args.seed}"
    log = {"model": args.model, "seed": args.seed, "epochs": args.epochs, "params": mm.n_params(model),
           "hardware": f"{platform.processor() if mm.DEVICE.type == 'cpu' else torch.cuda.get_device_name(0)} | torch {torch.__version__} | {args.threads} CPU threads",
           "epoch_seconds": [], "train_loss": []}
    print(tag, "params", log["params"], flush=True)

    for ep in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(videos), generator=gen)
        tot, n = 0.0, 0
        for i in range(0, len(perm), 32):
            idx = perm[i : i + 32]
            X, Y, M = [t.to(mm.DEVICE) for t in sample_batch(videos, idx, args.model, gen)]
            opt.zero_grad(set_to_none=True)
            loss = mm.masked_loss(model(X), Y, M)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx); n += len(idx)
        if mm.DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        log["epoch_seconds"].append(dt); log["train_loss"].append(tot / n)
        print(f"{tag} epoch {ep+1}/{args.epochs} loss {tot/n:.5f} time {dt:.0f}s", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out, tag + ".pt"))  # final-epoch weights
        json.dump(log, open(os.path.join(args.out, tag + ".json"), "w"), indent=1)

    # implementation check on validation (never used on test)
    model.eval()
    X, Y, M = mm.make_eval(mm.to_float(val[:200]))
    with torch.no_grad():
        log["val200_E5_1x"] = mm.per_horizon_mse(model(X.to(mm.DEVICE)).cpu(), Y).mean().item()
    json.dump(log, open(os.path.join(args.out, tag + ".json"), "w"), indent=1)
    print(tag, "done; val200 E5 (1x)", log["val200_E5_1x"], flush=True)


if __name__ == "__main__":
    main()
