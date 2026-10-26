"""Extra numbers that back the written answers (Q2.7-Q2.9, Q2.13): not new experiments, only measurements on the test split.
  python extra_analysis.py NAME:SEED [NAME:SEED ...]  -> results/extra_<name>_<seed>.json"""
import json, os, sys
import cv2
import numpy as np
import torch
import analysis as an
import mm

HERE = os.path.dirname(os.path.abspath(__file__))


def blur(x, s=3.0):  # x: (N, 5, 1, 64, 64) tensor -> same shape
    a = x.numpy()[:, :, 0]
    out = np.stack([[cv2.GaussianBlur(fr, (0, 0), s) for fr in v] for v in a])
    return out


def corr(a, b):
    a = a.reshape(len(a), -1); b = b.reshape(len(b), -1)
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def mass_centroid(fr):
    m = fr.sum()
    ys, xs = np.mgrid[0:64, 0:64]
    return float(m), (float((fr * ys).sum() / (m + 1e-9)), float((fr * xs).sum() / (m + 1e-9)))


def main(spec):
    V = an.test_videos()
    f = an.make_predictor(spec)
    out = {}
    # (1) does the prediction stall? correlation of h=1,2 prediction with (blurred) last input vs (blurred) true target
    for k in (1, 2, 3):
        X, Y, M = mm.make_eval(V, k, 1)
        P = mm.predict(f, X)
        last = blur(X[:, 4:5].expand(-1, 5, -1, -1, -1).contiguous())
        tr, pr = blur(Y), blur(P)
        d = {}
        for h in (0, 1):
            d[f"h{h+1}_corr_pred_vs_last_input"] = float(corr(pr[:, h], last[:, h]).mean())
            d[f"h{h+1}_corr_pred_vs_truth"] = float(corr(pr[:, h], tr[:, h]).mean())
            d[f"h{h+1}_corr_last_input_vs_truth"] = float(corr(last[:, h], tr[:, h]).mean())
            d[f"h{h+1}_pred_mass_over_truth_mass"] = float(P[:, h].sum((1, 2, 3)).mean() / Y[:, h].sum((1, 2, 3)).mean())
        out[f"{k}x"] = d
    # (2) paired per-video comparison of C_R and C_M
    X1, Y1, _ = mm.make_eval(V, 1, 1)
    P1 = mm.predict(f, X1)
    C = {}
    for n, T in an.TRANS.items():
        C[n] = mm.per_horizon_mse(mm.predict(f, T(X1)), T(P1)).mean(1).numpy()
    out["paired_C_R_gt_C_M_fraction"] = float((C["R"] > C["M"]).mean())
    out["median_C_R_over_C_M"] = float(np.median(C["R"] / C["M"]))
    out["E_mean_over_C_mean_M"] = float(mm.per_horizon_mse(P1, Y1).mean() / C["M"].mean())
    # (3) mass and centroid per horizon for the Figure 3 videos: rows (3) f(T X), (4) T f(X), truth T(Y)
    items = json.load(open(os.path.join(HERE, "results", "fig3_items.json")))["fig3_items"]
    rows = {}
    for tn, i in items:
        T = an.TRANS[tn]
        X, Y, _ = mm.make_eval(V[i:i + 1], 1, 1)
        fX = an.call(f, X); fTX = an.call(f, T(X)); TfX = T(fX); TY = T(Y)
        r = {}
        for nm, arr in (("row3_f_of_TX", fTX), ("row4_T_of_fX", TfX), ("truth_TY", TY)):
            ms = [mass_centroid(arr[0, h, 0].numpy()) for h in range(5)]
            r[nm] = {"mass": [round(m, 1) for m, _ in ms], "centroid_rc": [[round(c[0], 1), round(c[1], 1)] for _, c in ms]}
        r["mse_row3_vs_row4"] = float(((fTX - TfX) ** 2).mean())
        r["mse_row3_vs_truth"] = float(((fTX - TY) ** 2).mean())
        r["mse_row4_vs_truth"] = float(((TfX - TY) ** 2).mean())
        rows[f"{tn}_{i}"] = r
    out["fig3_videos"] = rows
    json.dump(out, open(os.path.join(HERE, "results", f"extra_{an.key(spec)}.json"), "w"), indent=1)
    print(an.key(spec), json.dumps({k: v for k, v in out.items() if k != "fig3_videos"}, indent=1))


if __name__ == "__main__":
    mm.require_gpu()
    for a in sys.argv[1:]:
        p = a.split(":")
        main((p[0], int(p[1])) + (("sym",) if len(p) == 3 else ()))
