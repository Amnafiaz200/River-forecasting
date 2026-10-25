"""Draws the Q1 architecture diagram -> ../report/fig_q1_architecture.png and fig_q1_graph.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report")
os.makedirs(OUT, exist_ok=True)

C_SRC, C_PRE, C_CORE, C_OUT, C_STAT = "#dbe9f6", "#fdebc8", "#d9efd8", "#f6d5d5", "#e6e0f3"


def box(ax, x, y, w, h, text, fc, fs=8, bold=False, ec="#333333", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, p, q, label=None, fs=7, color="#222222", off=(0, 0.12), ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, fontsize=fs, ha="center", va="bottom",
                color="#7a1f1f", style="italic")


# ------------------------------------------------------------------ Figure 1: full architecture
fig, ax = plt.subplots(figsize=(17.6, 10.2))
ax.set_xlim(-0.4, 17.6); ax.set_ylim(0, 10.2); ax.axis("off")

ax.text(8.5, 9.95, "River forecasting system: data sources → alignment → graph space-time network → S1–S4 forecasts (issued every 10 min)",
        ha="center", fontsize=11.5, fontweight="bold")
for x, t in [(1.5, "1. Data sources (as of issuance time τ)"), (5.05, "2. Alignment / projection"),
             (8.4, "3. Node-time tensor"), (11.45, "4. Four ST blocks (T then S)"), (15.05, "5. Forecast mapping")]:
    ax.text(x, 9.55, t, ha="center", fontsize=9.5, fontweight="bold", color="#444")

# sources
srcs = [
    ("Water levels S1–S4, T1\nW  [B×72×5]  (m)\n10-min, ts ∈ (τ−720, τ] min", 8.1),
    ("Rainfall R1, R2\nP  [B×72×2]  (mm/10 min)\n+ gauge coordinates ρ [2×2]", 6.55),
    ("Reservoir ACTUAL release\nq^act [B×72×1]  (m³/s)\npast only", 5.0),
    ("Reservoir ANNOUNCED schedule\nq^sch [B×78×1] + flag f^sch [B×78×1]\nannounced ≤ τ, effective τ−720…τ+60", 3.45),
    ("Static river graph\nstation chainage d_i (km), connectivity A [7×7],\nedge lengths Δ_ji [E=6]", 1.9),
]
for txt, y in srcs:
    box(ax, 0.1, y, 2.9, 1.2, txt, C_SRC if "Static" not in txt else C_STAT, fs=7.6)

# alignment boxes
pre = [
    ("QC + 10-min grid snap\nmask M [B×72×5], age Δ [B×72×5]\nGRU-D imputation (Eq. 1)\nreport-time ≤ τ only", 8.1),
    ("Rain → node projection ω (Eq. 2)\n[B×72×2] → [B×7×72]\n(distance-softmax, learned σ)", 6.55),
    ("Concatenate past actual\nand future scheduled in time\n→ q [B×78×2] (+flag, deviation e)", 4.0),
    ("Future steps t∈(τ,τ+60]:\nlevels/rain = masked placeholders\n(mask=0), release = schedule", 2.65),
]
box(ax, 3.7, 8.1, 2.7, 1.2, pre[0][0], C_PRE, fs=7.4)
box(ax, 3.7, 6.55, 2.7, 1.2, pre[1][0], C_PRE, fs=7.4)
box(ax, 3.7, 4.55, 2.7, 1.35, pre[2][0], C_PRE, fs=7.4)
box(ax, 3.7, 2.85, 2.7, 1.3, pre[3][0], C_PRE, fs=7.4)
box(ax, 3.7, 1.05, 2.7, 1.3, "Graph builder: nodes\nRes, S1, T1, S2, U, S3, S4 (N=7)\nU = latent unmonitored tributary;\nedge distances → φ(Δ) [E×16], E=6", C_STAT, fs=7.4)

for _, y in srcs[:2]:
    arrow(ax, (3.0, y + 0.6), (3.7, y + 0.6))
arrow(ax, (3.0, 5.6), (3.7, 5.3)); arrow(ax, (3.0, 4.05), (3.7, 4.95))
arrow(ax, (3.0, 4.0), (3.7, 3.5), off=(0, 0))
arrow(ax, (3.0, 2.5), (3.7, 1.7))

# tensor assembly
box(ax, 7.0, 3.6, 2.8, 4.9,
    "Node-time tensor\nX  ∈ R^{B×N×T×F}\nB = batch (1 at runtime)\nN = 7 nodes\nT = L+H = 72+6 = 78 steps\n"
    "F = 10 features per node:\n [ŵ, m, δ, r_i, q^act, q^sch,\n  f^sch, sin/cos day, season]\n\n"
    "+ node-type embedding e_type\n   (res / river stn / trib stn / latent)\n"
    "Linear F→d (d = 64)\n\nH⁰ ∈ R^{B×7×78×64}", "#fff6d6", fs=8.2, bold=False)
for y in (8.7, 7.15):
    arrow(ax, (6.4, y - 0.1), (7.0, y - 0.6 if y > 8 else 6.9))
arrow(ax, (6.4, 5.2), (7.0, 5.9)); arrow(ax, (6.4, 3.5), (7.0, 4.6)); arrow(ax, (6.4, 1.7), (7.0, 3.9))

# ST blocks
ax.add_patch(FancyBboxPatch((10.1, 2.2), 3.0, 6.3, boxstyle="round,pad=0.01,rounding_size=0.08", fc=C_CORE, ec="#2c5d2a", lw=1.4))
ax.text(11.6, 8.2, "ST block ×4  (l = 1…4)", ha="center", fontsize=8.8, fontweight="bold")
box(ax, 10.25, 5.35, 2.7, 2.55,
    "T: causal temporal conv\nper node, kernel 3, dilation\nD_l ∈ {1, 3, 9, 27}, GLU + residual\n(Eq. 4)\n[B×7×78×64] → same\nreceptive field 81 steps = 13.5 h", "#c9e6c7", fs=7.4)
box(ax, 10.25, 2.35, 2.7, 2.75,
    "S: lag-attention graph conv\nover directed edges j→i:\nsoftmax over lags λ=0…18\n(travel time) with prior\nλ·10 min ≈ Δ_ji / c₀ (Eq. 5)\n+ gate g_ji; sum over parents\n+ residual + LayerNorm", "#c9e6c7", fs=7.4)
arrow(ax, (11.6, 5.35), (11.6, 5.1))
ax.text(11.6, 2.2-0.22, "4 stacked blocks ⇒ 4 hops (Res→S4)", fontsize=7, color="#2c5d2a", style="italic", ha="center")
arrow(ax, (9.8, 6.0), (10.1, 6.0))
ax.text(9.95, 6.1, "H⁰", fontsize=7, ha="center", color="#7a1f1f")

# forecast mapping
box(ax, 13.75, 6.3, 3.6, 1.9, "Slice last 6 time steps t=τ+10…τ+60 and\nnodes S1,S2,S3,S4  (drop Res,T1,U)\nH⁴ → [B×4×6×64]\nconcat h(τ) skip → [B×4×6×128]", C_OUT, fs=7.6)
box(ax, 13.75, 4.35, 3.6, 1.6, "Forecast mapping (Eq. 6)\nŷ_i^q(h) = w̃_i(τ) + s_i·MLP_q(·)\nquantiles q ∈ {0.1, 0.5, 0.9}\nmonotone via cumulative softplus", C_OUT, fs=7.6)
box(ax, 13.4, 2.45, 3.4, 1.5, "Output Ŷ ∈ R^{B×6×4×3}\n(h = 10…60 min; stations S1–S4; 3 quantiles)\nMAIN FORECAST: h = 60 min → [B×4×3]\nmedian = point forecast, q0.9 = flood alert", "#f3b9b9", fs=7.6, bold=True)
arrow(ax, (13.1, 5.6), (13.75, 7.0)); arrow(ax, (15.55, 6.3), (15.55, 5.95)); arrow(ax, (15.55, 4.35), (15.55, 3.95))
ax.text(15.55, 1.9, "Training: pinball loss (Eq. 7) on Y [B×6×4];\nY is NOT available at issuance", ha="center", fontsize=7.4, style="italic")

# persistence skip: routed around the outside
xs=[-0.2,-0.2,17.5,17.5]; ys=[8.7,0.75,0.75,5.15]
ax.plot(xs, ys, ls="--", lw=0.9, color="#8a4b00")
ax.plot([0.1,-0.2],[8.7,8.7], ls="--", lw=0.9, color="#8a4b00")
ax.add_patch(FancyArrowPatch((17.5,5.15),(17.35,5.15), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#8a4b00"))
ax.text(3.0, 0.82, "persistence skip: last valid level w̃_i(τ), i ∈ {S1…S4}, added back in Eq. 6", fontsize=7.4, color="#8a4b00", ha="left")

# legend
for i, (c, t) in enumerate([(C_SRC, "data source"), (C_PRE, "alignment / projection"), (C_STAT, "static / graph"),
                            (C_CORE, "learned ST network"), (C_OUT, "forecast")]):
    ax.add_patch(FancyBboxPatch((6.6 + 2.15 * i, 0.15), 0.3, 0.3, boxstyle="round,pad=0.005", fc=c, ec="#333"))
    ax.text(6.98 + 2.15 * i, 0.3, t, fontsize=7.5, va="center")
fig.savefig(os.path.join(OUT, "fig_q1_architecture.png"), dpi=170, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Figure 2: graph + information paths
fig, ax = plt.subplots(figsize=(11, 3.7))
ax.set_xlim(0, 11); ax.set_ylim(0, 3.7); ax.axis("off")
nodes = {"Res": (0.8, 1.8), "S1": (2.8, 1.8), "S2": (5.0, 1.8), "S3": (7.4, 1.8), "S4": (9.8, 1.8), "T1": (3.9, 3.1), "U": (6.2, 3.1)}
col = {"Res": "#bcd7f0", "S1": "#d9efd8", "S2": "#d9efd8", "S3": "#d9efd8", "S4": "#d9efd8", "T1": "#fdebc8", "U": "#eeeeee"}
for n, (x, y) in nodes.items():
    ax.add_patch(Circle((x, y), 0.42, fc=col[n], ec="#333", lw=1.2, ls="--" if n == "U" else "-"))
    ax.text(x, y, n, ha="center", va="center", fontweight="bold", fontsize=10)
edges = [("Res", "S1"), ("S1", "S2"), ("S2", "S3"), ("S3", "S4"), ("T1", "S2"), ("U", "S3")]
for a, b in edges:
    (x1, y1), (x2, y2) = nodes[a], nodes[b]
    d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    ax.add_patch(FancyArrowPatch((x1 + 0.45 * ux, y1 + 0.45 * uy), (x2 - 0.45 * ux, y2 - 0.45 * uy), arrowstyle="-|>", mutation_scale=13, lw=1.4))
ax.text(1.8, 2.05, "Δ_{Res,1}", fontsize=8, ha="center"); ax.text(3.9, 2.05, "Δ_{12}", fontsize=8, ha="center")
ax.text(6.2, 2.05, "Δ_{23}", fontsize=8, ha="center"); ax.text(8.6, 2.05, "Δ_{34}", fontsize=8, ha="center")
ax.text(1.9, 3.2, "rain R1,R2 → node inputs r_i via ω", fontsize=8, color="#1f4e8c", ha="center")
ax.text(8.0, 3.2, "U: latent inflow (rain-driven + learned)", fontsize=7, ha="center", color="#555", va="center")
ax.text(0.8, 0.95, "q^act (past) + q^sch (future)", fontsize=8, ha="center", color="#1f4e8c")
ax.text(5.4, 0.55, "Edges carry travel-time lag attention α_ji(t,λ); hops: Res→S4 = 4, T1→S4 = 3, U→S4 = 2, S1→S4 = 3.  Four ST blocks ⇒ every upstream node is within reach.",
        fontsize=8, ha="center")
ax.text(5.4, 0.15, "Dashed node U has no sensor: its state is inferred from rain and the S2→S3 mismatch.", fontsize=8, ha="center", style="italic")
fig.savefig(os.path.join(OUT, "fig_q1_graph.png"), dpi=170, bbox_inches="tight")
plt.close(fig)
print("saved")
