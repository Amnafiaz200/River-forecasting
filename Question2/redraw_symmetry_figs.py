"""Re-draws Figure 3 and Figure 6(a) of the report in a portrait layout (four panels stacked, larger labels) from the saved checkpoints.
Same models, same four videos (results/fig3_items.json), same frames as analysis.py; only the layout differs. Inference only, no GIFs written.
  python redraw_symmetry_figs.py         (ALLOW_CPU=1 is set automatically; ~1 minute on a CPU)
"""
import json, os
os.environ.setdefault("ALLOW_CPU", "1")
import matplotlib.pyplot as plt
import analysis as an


def draw(f, V, items, out_png):
    fig = plt.figure(figsize=(6.3, 9.0))
    sub = fig.subfigures(4, 1, hspace=0.02)
    for s, (tn, idx) in zip(sub, items):
        rows, labels = an.symmetry_panel_rows(f, V, idx, an.TRANS[tn])
        name = "mirror M" if tn == "M" else "rotation R"
        an.tile_grid(rows, labels, [f"in {i+1}" for i in range(5)] + [f"h={i+1}" for i in range(5)],
                     title=f"{name}, test video {idx} (dataset index {9000 + idx})", fig=s, fs=6.5, left=0.215)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    V = an.test_videos()
    items = [tuple(i) for i in json.load(open(os.path.join(an.RES, "fig3_items.json")))["fig3_items"]]
    draw(an.make_predictor(("A", 0)), V, items, os.path.join(an.REP, "fig3_symmetry_A.png"))
    draw(an.make_predictor(("D", 0, "sym")), V, items, os.path.join(an.REP, "fig6a_symmetry_final.png"))
    print("wrote fig3_symmetry_A.png and fig6a_symmetry_final.png")


if __name__ == "__main__":
    main()
