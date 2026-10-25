"""Figure 1: dense optical-flow statistics of the first 50 training videos (950 consecutive frame pairs)."""
import json, os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mm

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(HERE, "..", "report"); os.makedirs(REP, exist_ok=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

vid = mm.load_videos(os.path.join(HERE, "data"))[:50, :, 0].numpy()  # (50, 20, 64, 64) uint8 (== float*255, exactly)


def collect(skip_merged=False, vids=None):
    th, sp = [], []
    n_pairs = 0
    for v in (vid if vids is None else vids):
        for s in range(19):
            a, b = v[s], v[s + 1]
            if skip_merged:  # robustness check: drop pairs in which the two digits merge into one blob
                if min(cv2.connectedComponents((a > 51).astype(np.uint8))[0], cv2.connectedComponents((b > 51).astype(np.uint8))[0]) < 3:
                    continue
            n_pairs += 1
            flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[..., 0], flow[..., 1]
            mag = np.sqrt(dx ** 2 + dy ** 2)
            keep = (a > 51) & (b > 51) & (mag > 0.5)
            th.append(np.degrees(np.arctan2(-dy[keep], dx[keep])) % 360.0)
            sp.append(mag[keep])
    return np.concatenate(th), np.concatenate(sp), n_pairs


theta, speed, n_pairs = collect()
bins = np.arange(0, 361, 20)
hist = np.histogram(theta, bins)[0] / len(theta)
tv = lambda p, q: 0.5 * np.abs(p - q).sum()
# Mirror: theta -> 180 - theta.  Rotation by 90 deg anticlockwise: theta -> theta + 90.
# Compared on a 10-degree histogram (90 is not a multiple of the 20-degree bin width).
fine = np.histogram(theta, np.arange(0, 361, 10))[0] / len(theta)  # 36 bins of 10 deg
def transformed(th_arr, f):
    return np.histogram(f(th_arr) % 360.0, np.arange(0, 361, 10))[0] / len(th_arr)
h_mirror = transformed(theta, lambda t: 180.0 - t)
h_rot = transformed(theta, lambda t: t + 90.0)
tv_mirror, tv_rot = tv(fine, h_mirror), tv(fine, h_rot)
# compare with the uniform distribution and with the "no-motion-prior" level
tv_uniform = tv(hist, np.full(18, 1 / 18))

theta2, speed2, n_pairs2 = collect(skip_merged=True)
fine2 = np.histogram(theta2, np.arange(0, 361, 10))[0] / len(theta2)
tv_mirror2 = tv(fine2, transformed(theta2, lambda t: 180.0 - t))
tv_rot2 = tv(fine2, transformed(theta2, lambda t: t + 90.0))

# axis-aligned share: directions within 20 deg of the 4 axes
def share(t, centre, half=20):
    d = np.abs((t - centre + 180) % 360 - 180)
    return (d <= half).mean()


# ---- supplementary (NOT part of Figure 1): how much of the lumpiness is sampling noise?
half = lambda vs: collect(vids=vs)[0]
th_a, th_b = half(vid[:25]), half(vid[25:])
fa = np.histogram(th_a, np.arange(0, 361, 20))[0] / len(th_a); fb = np.histogram(th_b, np.arange(0, 361, 20))[0] / len(th_b)
tv_split_half = tv(fa, fb)
full = mm.load_videos(os.path.join(HERE, "data"))
big = full[:1000, :, 0].numpy()
th_big, sp_big, _ = collect(vids=big)
hist_big = np.histogram(th_big, bins)[0] / len(th_big)
fine_big = np.histogram(th_big, np.arange(0, 361, 10))[0] / len(th_big)
supp_res = {"supp_videos": 1000, "supp_pixels": int(len(th_big)), "supp_TV_to_uniform": float(tv(hist_big, np.full(18, 1 / 18))),
            "supp_TV_hist_vs_mirrored": float(tv(fine_big, transformed(th_big, lambda t: 180.0 - t))),
            "supp_TV_hist_vs_rot90": float(tv(fine_big, transformed(th_big, lambda t: t + 90.0))),
            "supp_max_bin": float(hist_big.max()), "supp_min_bin": float(hist_big.min()), "supp_median_speed": float(np.median(sp_big)),
            "supp_p99_speed": float(np.percentile(sp_big, 99)), "TV_split_half_25_vs_25": float(tv_split_half)}

res = {
    "pairs": 950, "pixels_kept": int(len(theta)), "median_speed": float(np.median(speed)),
    "mean_speed": float(speed.mean()), "p90_speed": float(np.percentile(speed, 90)),
    "hist_18": hist.tolist(), "max_bin": float(hist.max()), "min_bin": float(hist.min()),
    "TV_to_uniform": float(tv_uniform), "TV_hist_vs_mirrored": float(tv_mirror), "TV_hist_vs_rot90": float(tv_rot),
    "no_merge_pairs": int(n_pairs2), "no_merge_pixels": int(len(theta2)),
    "no_merge_TV_hist_vs_mirrored": float(tv_mirror2), "no_merge_TV_hist_vs_rot90": float(tv_rot2),
    "no_merge_median_speed": float(np.median(speed2)),
    **supp_res,
    "share_right": share(theta, 0), "share_up": share(theta, 90), "share_left": share(theta, 180), "share_down": share(theta, 270),
}
# a fair reference for TV: the sampling noise of a perfectly symmetric histogram with the same number of pixels is tiny
json.dump(res, open(os.path.join(HERE, "results", "fig1_flow.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "hist_18"}, indent=1))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.bar(bins[:-1], hist, width=20, align="edge", color="#4c78a8", edgecolor="white")
a1.axhline(1 / 18, color="k", ls="--", lw=1, label="uniform = 1/18")
a1.set_xlabel("flow direction θ = atan2(−dy, dx)  [deg]  (0 = right, 90 = up on screen)")
a1.set_ylabel("fraction of kept pixels"); a1.set_xticks(range(0, 361, 45)); a1.legend(); a1.set_title("(a) flow direction, 18 bins of 20°")
sb = np.arange(0.5, 11.0, 0.15)
a2.hist(speed, bins=sb, weights=np.full(len(speed), 1 / len(speed)), color="#f58518", edgecolor="white")
a2.axvline(np.median(speed), color="k", lw=1.5, label=f"median = {np.median(speed):.2f} px/frame")
med = np.median(speed)
a2.axvline(2 * med, color="gray", ls="--", lw=1, label=f"2x median = {2*med:.1f}"); a2.axvline(3 * med, color="gray", ls=":", lw=1.3, label=f"3x median = {3*med:.1f}")
a2.axvline(8, color="#d62728", ls="-.", lw=1, label="model A reach = 8 px (Q2.3)")
a2.set_xlabel("flow speed  √(dx²+dy²)  [pixels per frame]"); a2.set_ylabel("fraction of kept pixels"); a2.legend(); a2.set_title("(b) flow speed")
fig.tight_layout()
fig.savefig(os.path.join(REP, "fig1_flow.png"), dpi=160)
