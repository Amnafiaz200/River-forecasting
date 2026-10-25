"""Table builders for Question 2: every number is read from Question2/results/*.json (nothing typed by hand)."""
import json, os
import numpy as np
from rl_helpers import *

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "Question2", "results")
RUNS = os.path.join(HERE, "..", "Question2", "runs")


def J(name):
    return json.load(open(os.path.join(RES, name), encoding="utf8"))


def fmt(x):
    """3 significant figures; scientific notation below 1e-3."""
    if x == 0:
        return "0"
    return f"{x:.2e}" if abs(x) < 1e-3 else f"{x:.3g}"


def ms(vals, f=fmt):
    """'mean (seed 0, seed 1)'"""
    if len(vals) == 1:
        return f(vals[0])
    return f"{f(float(np.mean(vals)))} ({', '.join(f(v) for v in vals)})"


def seeds(spec_prefix, field, sym=False, fn=None):
    out = []
    for s in (0, 1):
        k = f"{spec_prefix}_{s}" + ("_sym" if sym else "")
        d = J(f"metrics_{k}.json")
        out.append(d[field])
    return out


def table1():
    rows = [["Check", "Expected", "Yours"]]
    for r in J("table1.json"):
        rows.append([r["check"], r["expected"], r["yours"]])
    return table(rows, [0.42, 0.29, 0.29])


def table2():
    t = J("table2.json")
    hdr = ["Predictor"] + [f"MSE<sub>{h}</sub>" for h in range(1, 6)] + ["E<super>(5)</super>", "Trainable params"]
    rows = [hdr]
    for name, k in [("Blank", "blank"), ("Copy-last", "copy")]:
        d = t[k]
        rows.append([name] + [fmt(v) for v in d["mse_h"]] + [fmt(d["E5_1x"]), "0"])
    a = [t["A_0"], t["A_1"]]
    rows.append(["Model A (mean (seed 0, seed 1))"] + [ms([x["mse_h"][h] for x in a]) for h in range(5)] +
                [ms([x["E5_1x"] for x in a]), f"{a[0]['params']:,}"])
    return table(rows, [0.17, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115])


def table3():
    t = J("table3_4.json")
    rows = [["Predictor", "E<super>(5)</super>", "C<sub>M</sub>", "E<sub>M</sub>", "C<sub>R</sub>", "E<sub>R</sub>"]]
    rows.append(["Model A (mean (s0, s1))"] + [ms([t["A_0"][f], t["A_1"][f]]) for f in ("E5_1x", "C_M", "E_M", "C_R", "E_R")])
    for name, k in [("Blank", "blank"), ("Copy-last", "copy")]:
        rows.append([name] + [fmt(t[k][f]) for f in ("E5_1x", "C_M", "E_M", "C_R", "E_R")])
    return table(rows, [0.25, 0.15, 0.15, 0.15, 0.15, 0.15])


def table4():
    t = J("table3_4.json")
    rows = [["Predictor", "E<super>(2)</super> 1×", "E<super>(2)</super> 2×", "E<super>(2)</super> 3×", "E<super>(5)</super> 1×", "E<super>(5)</super> 2×"]]
    rows.append(["Model A (mean (s0, s1))"] + [ms([t["A_0"][f], t["A_1"][f]]) for f in ("E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x")])
    rows.append(["Copy-last"] + [fmt(t["copy"][f]) for f in ("E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x")])
    rows.append(["Blank (reference)"] + [fmt(t["blank"][f]) for f in ("E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x")])
    return table(rows, [0.25, 0.15, 0.15, 0.15, 0.15, 0.15])


def _solo(timing, name, seed=None, sym=False):
    k = f"{name}_{seed}" + ("_sym" if sym else "")
    return timing.get(k, {})


def table56(rows_spec, timing_name="timing.json"):
    """rows_spec: list of (label, model name, symmetrised?, training-model name for time)"""
    timing = J(timing_name)
    hdr = ["Model", "E<super>(5)</super> 1×", "C<sub>M</sub>", "E<sub>M</sub>", "C<sub>R</sub>", "E<sub>R</sub>",
           "E<super>(2)</super> 1×", "E<super>(2)</super> 2×", "E<super>(2)</super> 3×", "Params", "Train s / epoch", "Inference s"]
    rows = [hdr]
    for label, name, sym, tname in rows_spec:
        g = lambda f: seeds(name, f, sym)
        tr = [timing[f"{tname}_{s}"]["train_s_per_epoch_solo"] for s in (0, 1)]
        inf = [timing[f"{name}_{s}" + ("_sym" if sym else "")]["inference_s"] for s in (0, 1)]
        rows.append([label, ms(g("E5_1x")), ms(g("C_M")), ms(g("E_M")), ms(g("C_R")), ms(g("E_R")),
                     ms(g("E2_1x")), ms(g("E2_2x")), ms(g("E2_3x")), f"{g('params')[0]:,}",
                     ms(tr, lambda v: f"{v:.0f}"), ms(inf, lambda v: f"{v:.1f}")])
    return table(rows, [0.095, 0.095, 0.08, 0.085, 0.08, 0.085, 0.08, 0.08, 0.08, 0.06, 0.09, 0.09], font=ParagraphStyle("c6", parent=CELL, fontSize=6.2, leading=7.8))


def logged_epoch_times():
    out = {}
    for n in ("A", "B", "C", "D"):
        v = []
        for s in (0, 1):
            f = os.path.join(RUNS, f"{n}_s{s}.json")
            if os.path.exists(f):
                v.append(float(np.mean(json.load(open(f))["epoch_seconds"])))
        out[n] = v
    return out
