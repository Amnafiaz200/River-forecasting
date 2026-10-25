"""Evaluation, tables and figures for Question 2 (test split = videos 9000-9999).

Stages (run in this order; see README):
  python analysis.py table2        # Table 2 + Figure 2            (model A, Blank, Copy-last)
  python analysis.py baseline      # Tables 3, 4 + Figures 3, 4, 5 (model A, Blank, Copy-last)
  python analysis.py models        # Table 5 / Table 6 metrics for all trained models (cached per model)
  python analysis.py final NAME    # Figure 6 with the chosen final model, e.g. D (symmetrised)
"""
import json, os, sys, time
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import mm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
REP = os.path.join(HERE, "..", "report"); os.makedirs(REP, exist_ok=True)
GIF = os.path.join(HERE, "gifs"); os.makedirs(GIF, exist_ok=True)
RUNS = os.path.join(HERE, "runs")
torch.set_num_threads(6)

R90 = lambda x: mm.rot(x, 1)
TRANS = {"M": mm.mirror, "R": R90}


# ----------------------------------------------------------------------------- loading / predictors
def test_videos():
    return mm.to_float(mm.load_videos(os.path.join(HERE, "data"))[mm.TEST])


def load_model(name, seed):
    m = mm.build_model(name)
    m.load_state_dict(torch.load(os.path.join(RUNS, f"{name}_s{seed}.pt"), map_location="cpu"))
    return m.eval()


def make_predictor(spec):
    """spec: 'blank' | 'copy' | (name, seed) | (name, seed, 'sym')"""
    if spec == "blank":
        return lambda X: torch.zeros_like(X)
    if spec == "copy":
        return lambda X: X[:, -1:].expand(-1, 5, -1, -1, -1).contiguous()
    f = load_model(spec[0], spec[1])
    if len(spec) == 3:
        return mm.symmetrised(f)
    return f


def key(spec):
    return spec if isinstance(spec, str) else "_".join(map(str, spec))


def fmt(x):
    """3 significant figures; scientific notation below 1e-3."""
    if x == 0:
        return "0"
    return f"{x:.2e}" if abs(x) < 1e-3 else f"{x:.3g}"


# ----------------------------------------------------------------------------- metrics
@torch.no_grad()
def full_metrics(f, V, light=False):
    X1, Y1, _ = mm.make_eval(V, 1, 1)
    P = mm.predict(f, X1)
    mse = mm.per_horizon_mse(P, Y1)  # (N, 5)
    out = {"mse_h": mse.mean(0).tolist(), "E5_1x": mse.mean().item(), "E2_1x": mse[:, :2].mean().item()}
    per_video = {}
    if light:
        return out, per_video
    for name, T in TRANS.items():
        PT = mm.predict(f, T(X1))  # f(T(X))
        C = mm.per_horizon_mse(PT, T(P)).mean(1)  # consistency gap per video  MSE(f(TX), T f(X))
        E = mm.per_horizon_mse(PT, T(Y1)).mean(1)  # transformed error        MSE(f(TX), T(Y))
        out[f"C_{name}"], out[f"E_{name}"] = C.mean().item(), E.mean().item()
        per_video[f"C_{name}"] = C
    for k in (2, 3):
        Xk, Yk, Mk = mm.make_eval(V, k, 1)
        mk = mm.per_horizon_mse(mm.predict(f, Xk), Yk)
        out[f"E2_{k}x"] = mk[:, :2].mean().item()  # only the first two horizons are compared across speeds
        if k == 2:
            out["E5_2x"] = mk.mean().item()
    return out, per_video


@torch.no_grad()
def inference_time(f, V):
    X1, _, _ = mm.make_eval(V, 1, 1)
    f(X1[:32])  # warm-up (not timed)
    t = 0.0
    for i in range(0, len(X1), 32):
        xb = X1[i : i + 32]  # already on the model's device
        t0 = time.perf_counter(); f(xb); t += time.perf_counter() - t0
    return t


def train_log(name, seed):
    return json.load(open(os.path.join(RUNS, f"{name}_s{seed}.json")))


def cached(spec, V, light=False, timing=True):
    path = os.path.join(RES, f"metrics_{key(spec)}.json")
    if os.path.exists(path):
        d = json.load(open(path))
        if light or "C_M" in d:
            return d
    f = make_predictor(spec)
    d, _ = full_metrics(f, V, light=light)
    if not isinstance(spec, str):
        d["params"] = mm.n_params(load_model(spec[0], spec[1]))
        log = train_log(spec[0], spec[1])
        d["train_s_per_epoch"] = float(np.mean(log["epoch_seconds"]))
        d["epochs"] = log["epochs"]
        if timing:
            d["inference_s"] = inference_time(f, V)
    else:
        d["params"] = 0
    json.dump(d, open(path, "w"), indent=1)
    return d


# ----------------------------------------------------------------------------- plotting helpers
def n_comp(frame):
    return cv2.connectedComponents((frame[0].numpy() > 0.2).astype(np.uint8))[0] - 1


def tile_grid(rows, row_labels, col_labels, title=None, fig=None, size=0.78):
    """rows: list (per row) of lists (per column) of 64x64 numpy arrays or None."""
    nr, nc = len(rows), len(rows[0])
    own = fig is None
    if own:
        fig = plt.figure(figsize=(nc * size + 0.9, nr * size + 0.7))
    axs = fig.subplots(nr, nc, gridspec_kw=dict(wspace=0.04, hspace=0.04, left=0.12, right=0.995, top=0.88, bottom=0.01), squeeze=False)
    for r in range(nr):
        for c in range(nc):
            ax = axs[r][c]; ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if rows[r][c] is None:
                ax.axis("off"); continue
            ax.imshow(rows[r][c], cmap="gray", vmin=0, vmax=1)
            if r == 0 and col_labels:
                ax.set_title(col_labels[c], fontsize=5.5, pad=1.5)
        axs[r][0].set_ylabel(row_labels[r], fontsize=5.5, rotation=0, ha="right", va="center", labelpad=2)
    if title:
        fig.suptitle(title, fontsize=7, y=0.995)
    if own:
        return fig
    return axs


def np_frames(x):  # (5,1,64,64) tensor -> list of arrays
    return [x[i, 0].numpy() for i in range(x.shape[0])]


def symmetry_panel_rows(f, V, idx, T):
    X, Y, _ = mm.make_eval(V[idx : idx + 1], 1, 1)
    with torch.no_grad():
        fX = f(X); fTX = f(T(X)); TfX = T(fX)
    rows = [np_frames(X[0]) + np_frames(fX[0]),
            np_frames(T(X)[0]) + np_frames(T(Y)[0]),
            np_frames(T(X)[0]) + np_frames(fTX[0]),
            np_frames(T(X)[0]) + np_frames(TfX[0])]
    labels = ["(1) X | f(X)", "(2) T(X) | T(Y)", "(3) T(X) | f(T(X))", "(4) T(X) | T(f(X))"]
    return rows, labels


def save_panel_gif(rows, labels, path, title):
    frames = []
    for h in range(5):
        r2 = [r[:5] + [r[5 + h]] for r in rows]
        fig = tile_grid(r2, labels, [f"in {i+1}" for i in range(5)] + [f"forecast h={h+1}"], title=title, size=0.95)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=900, loop=0)


def fig_symmetry(f, V, items, out_png, gif_tag, who):
    """items: list of (T name, video idx). 2x2 grid of four-row panels (Figure 3 / Figure 6a)."""
    fig = plt.figure(figsize=(14.2, 7.6))
    sub = fig.subfigures(2, 2, wspace=0.02, hspace=0.04)
    for s, (tn, idx) in zip(sub.flatten(), items):
        rows, labels = symmetry_panel_rows(f, V, idx, TRANS[tn])
        name = "mirror M" if tn == "M" else "rotation R"
        title = f"{name}, test video {idx} (dataset index {9000 + idx})"
        tile_grid(rows, labels, [f"in {i+1}" for i in range(5)] + [f"h={i+1}" for i in range(5)], title=title, fig=s)
        save_panel_gif(rows, labels, os.path.join(GIF, f"{gif_tag}_{tn}_test{idx}.gif"), f"{who}: {name}, test video {idx}")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------- video selection (rules use ground truth only)
def pick_fig2(V):
    comps = lambda i, frames: [n_comp(V[i, f - 1]) for f in frames]
    sep, ov = [], None
    for i in range(len(V)):
        c = comps(i, range(1, 11))
        if all(x == 2 for x in c) and len(sep) < 2:
            sep.append(i)
        # digits come together in the forecast window: >=2 of the target frames show a single blob; inputs are clean
        if ov is None and all(x == 2 for x in c[:5]) and sum(x == 1 for x in c[5:]) >= 2:
            ov = i
        if len(sep) == 2 and ov is not None:
            break
    return sep + [ov]


def pick_fig5(V):
    need = sorted(set(list(range(1, 8)) + [1, 3, 5, 7, 9, 11, 13] + [1, 4, 7, 10, 13, 16, 19]))
    for i in range(len(V)):
        if all(n_comp(V[i, f - 1]) == 2 for f in need):
            return i
    raise RuntimeError("no suitable video")


# ----------------------------------------------------------------------------- stages
def stage_table2():
    V = test_videos()
    out = {}
    for spec in ["blank", "copy", ("A", 0), ("A", 1)]:
        out[key(spec)] = cached(spec, V, light=True, timing=False)
        if not isinstance(spec, str):
            out[key(spec)]["params"] = mm.n_params(load_model(*spec))
    json.dump(out, open(os.path.join(RES, "table2.json"), "w"), indent=1)
    # Figure 2
    picks = pick_fig2(V)
    f = make_predictor(("A", 0))
    rows, rl = [], []
    for i in picks:
        X, Y, _ = mm.make_eval(V[i : i + 1], 1, 1)
        with torch.no_grad():
            P = f(X)
        rows += [np_frames(X[0]) + np_frames(Y[0]), [None] * 5 + np_frames(P[0])]
        rl += [f"video {i}: input | truth", f"video {i}: model A (seed 0)"]
    fig = tile_grid(rows, rl, [f"in {i+1}" for i in range(5)] + [f"h={i+1}" for i in range(5)], size=0.95)
    fig.savefig(os.path.join(REP, "fig2_baseline.png"), dpi=170); plt.close(fig)
    json.dump({"fig2_videos": picks}, open(os.path.join(RES, "fig2_videos.json"), "w"))
    print(json.dumps(out, indent=1), picks)


def stage_baseline():
    V = test_videos()
    t3 = {}
    perv = None
    for spec in ["blank", "copy", ("A", 0), ("A", 1)]:
        f = make_predictor(spec)
        d, pv = full_metrics(f, V)
        if spec == ("A", 0):
            perv = pv
        t3[key(spec)] = d
        print(key(spec), {k: fmt(v) for k, v in d.items() if k != "mse_h"}, flush=True)
    json.dump(t3, open(os.path.join(RES, "table3_4.json"), "w"), indent=1)
    f = make_predictor(("A", 0))
    # Figure 3: two largest per-video C_M and two largest per-video C_R
    items = []
    for tn in ("M", "R"):
        top = torch.topk(perv[f"C_{tn}"], 2).indices.tolist()
        items += [(tn, i) for i in top]
    fig_symmetry(f, V, items, os.path.join(REP, "fig3_symmetry_A.png"), "fig3_A", "model A seed 0")
    # Figure 4
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, tn, nm in zip(axs, ("M", "R"), ("mirror", "rotation")):
        c = perv[f"C_{tn}"].numpy()
        ax.hist(c, bins=50, color="#4c78a8")
        for j, (t_, i) in enumerate([it for it in items if it[0] == tn]):
            ax.axvline(c[i], color=["#d62728", "#2ca02c"][j], lw=1.6, label=f"Fig. 3: video {i}")
        ax.set_xlabel(f"per-video consistency gap C_{tn} ({nm})"); ax.set_ylabel("number of test videos")
        ax.set_yscale("log"); ax.legend(fontsize=7); ax.set_title(f"C_{tn}, model A seed 0 (mean {fmt(c.mean())})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(REP, "fig4_hist_A.png"), dpi=160); plt.close(fig)
    json.dump({"fig3_items": items, "C_M_max": float(perv["C_M"].max()), "C_R_max": float(perv["C_R"].max()),
               "C_M_median": float(perv["C_M"].median()), "C_R_median": float(perv["C_R"].median())},
              open(os.path.join(RES, "fig3_items.json"), "w"))
    # Figure 5
    vi = pick_fig5(V)
    fig5(f, V, vi, os.path.join(REP, "fig5_speed_A.png"), "model A (seed 0)")
    json.dump({"fig5_video": vi}, open(os.path.join(RES, "fig5_video.json"), "w"))
    print("fig3 items", items, "fig5 video", vi)


def fig5(f, V, vi, path, who):
    rows, rl = [], []
    for k in (1, 2, 3):
        X, Y, M = mm.make_eval(V[vi : vi + 1], k, 1)
        with torch.no_grad():
            P = f(X)
        rows += [np_frames(X[0]) + np_frames(Y[0][:2]), [None] * 5 + np_frames(P[0][:2])]
        rl += [f"{k}x: input | truth", f"{k}x: {who}"]
    fig = tile_grid(rows, rl, [f"in {i+1}" for i in range(5)] + ["h=1", "h=2"], size=0.95)
    fig.savefig(path, dpi=170); plt.close(fig)


def stage_models(names):
    V = test_videos()
    for spec in names:
        d = cached(spec, V)
        print(key(spec), {k: fmt(v) for k, v in d.items() if k != "mse_h"}, flush=True)


def stage_final(name):
    V = test_videos()
    items = [tuple(i) for i in json.load(open(os.path.join(RES, "fig3_items.json")))["fig3_items"]]
    vi = json.load(open(os.path.join(RES, "fig5_video.json")))["fig5_video"]
    f = make_predictor((name, 0, "sym"))
    fig_symmetry(f, V, items, os.path.join(REP, "fig6a_symmetry_final.png"), f"fig6a_{name}sym", f"final model ({name} symmetrised, seed 0)")
    fig5(f, V, vi, os.path.join(REP, "fig6b_speed_final.png"), f"final ({name}+sym)")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "table2":
        stage_table2()
    elif st == "baseline":
        stage_baseline()
    elif st == "models":
        names = []
        for a in sys.argv[2:]:  # e.g. B:0 D:1 D:0:sym
            p = a.split(":"); names.append((p[0], int(p[1])) + (("sym",) if len(p) == 3 else ()))
        stage_models(names)
    elif st == "final":
        stage_final(sys.argv[2])
