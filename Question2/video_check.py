"""Per-video check used in Q2.7 / Q2.16 of the report: for the four Figure 3 videos, the centre of mass and total brightness of the
forecast (h = 1 and h = 5) and the per-video errors of model A (seed 0) and of the final model (symmetrised D, seed 0).
Inference only (no training). Writes results/fig6_video_check.json.

  python video_check.py            (GPU, or ALLOW_CPU=1 on a CPU: inference of 4 videos only)
"""
import json, os
import numpy as np
import analysis as an
import mm

HERE = os.path.dirname(os.path.abspath(__file__))


def cen(fr):
    a = fr[0].numpy(); ys, xs = np.mgrid[:64, :64]; m = float(a.sum())
    return [float((a * ys).sum() / m), float((a * xs).sum() / m), m]  # row, column, total brightness


def main():
    V = an.test_videos()
    items = [tuple(i) for i in json.load(open(os.path.join(an.RES, "fig3_items.json")))["fig3_items"]]
    preds = {"A": an.make_predictor(("A", 0)), "Dsym": an.make_predictor(("D", 0, "sym"))}
    out = {}
    for tn, i in items:
        T = an.TRANS[tn]
        X, Y, _ = mm.make_eval(V[i : i + 1], 1, 1)
        rec = {"truth_TY": [cen(T(Y)[0][h]) for h in (0, 4)]}
        for nm, f in preds.items():
            P = an.call(f, X); PT = an.call(f, T(X))  # f(X), f(T(X))
            rec[nm] = {"C_T": float(((PT - T(P)) ** 2).mean()), "E_T": float(((PT - T(Y)) ** 2).mean()),
                       "row3_f_of_TX": [cen(PT[0][h]) for h in (0, 4)], "row4_T_of_fX": [cen(T(P)[0][h]) for h in (0, 4)]}
        out[f"{tn}_{i}"] = rec
    json.dump(out, open(os.path.join(an.RES, "fig6_video_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
