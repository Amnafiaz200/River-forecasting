"""Question 2 section of the report, part 1 (Setup, Baseline, Testing the Baseline). Part 2 is q2_text2.py."""
import os
from rl_helpers import *
import q2_tables as T
from q2_tables import J, fmt, ms

HERE = os.path.dirname(os.path.abspath(__file__))
fig = lambda n: os.path.join(HERE, n)


def pred_text(tag):
    """Return the paragraph(s) of report/q2_predictions.md that follow the heading containing `tag` (verbatim, as committed)."""
    txt = open(os.path.join(HERE, "q2_predictions.md"), encoding="utf8").read()
    parts = txt.split("\n## ")
    for p in parts:
        if tag in p.split("\n")[0]:
            return " ".join(l.strip() for l in p.split("\n")[1:] if l.strip())
    raise KeyError(tag)


def build():
    f1, ex = J("fig1_flow.json"), {s: J(f"extra_A_{s}.json") for s in (0, 1)}
    t34 = J("table3_4.json")
    a0, a1 = t34["A_0"], t34["A_1"]
    s = []
    s.append(P("Question 2 — Motion or Memory?", H1))

    # ------------------------------------------------------------------ important notes
    s.append(P("Important notes on how the experiments were run (read first)", H3))
    s += bullets([
        "<b>Reduced training budget.</b> The handout asks for 30 epochs per model. All numbers below come from <b>6 epochs per model</b> (1 500 Adam steps, "
        "the same for A, B, C and D and for both seeds), because the only machine available was a CPU laptop (AMD Ryzen 5 5500U, 6 cores, no GPU, PyTorch 2.10 CPU). One epoch of model A took "
        "about 11 minutes there; the full protocol (8 runs × 30 epochs) would need roughly 20–25 hours. Everything else follows the handout "
        "(Adam 1e-3, batch 32, one random window per video per epoch, per-video masked loss, no teacher forcing, final-epoch weights, seeds 0 and 1). "
        "<b>As a result the models are under-trained</b> (they predict soft blobs, not sharp digits, Figure 2), so absolute errors are larger than a 30-epoch model would give, "
        "and some conclusions below may change at 30 epochs. The same code runs the full protocol with <font name='DVM'>EPOCHS=30 bash run_all.sh</font> (it also uses a GPU if one is present).",
        "<b>Timing.</b> Runs were executed 2–4 at a time, which distorts wall-clock time, so Tables 5 and 6 report training time per epoch and inference time measured <i>afterwards, alone on the machine</i> "
        "(training: mean of 12 real training steps × 250 steps per epoch; inference: all 1 000 test videos, batch 32, <font name='DVM'>eval()</font> and <font name='DVM'>no_grad()</font>). "
        "The mean epoch times logged during the concurrent runs are given under Table 5.",
        "<b>Optional model D+B</b> was not trained (compute). The final model is therefore the symmetrised model D (Section 5).",
        "<b>Model D's design</b> was screened on the validation split only, with 250 training steps per candidate; both candidates were still at the 'static blob' plateau at that point, so the screening is almost a tie (Q2.12).",
        "<b>Q2.5 and Q2.11</b> are quoted exactly as committed to the repository before the corresponding experiments (commit times 12:17 and 12:22 on 25 October 2026, "
        "file <font name='DVM'>report/q2_predictions.md</font>). Where they turned out wrong the answers that follow say so.",
        "All performance numbers use the 1 000 test videos (dataset indices 9000–9999); the validation split was used only for implementation checks and for model D's screening. "
        "Table entries are mean (seed 0, seed 1); figures and GIFs use seed 0.",
    ])

    # ------------------------------------------------------------------ 1. setup
    s.append(P("1. Setup", H2))
    s.append(P("<b>Table 1.</b> Checks of the transformation and window functions (the same functions used in all experiments; synthetic tensors).", CAP))
    s.append(T.table1())
    s.append(P("All rows match. (The extra row also confirms that the eight elements of G are distinct and that g<super>−1</super>(g(x)) = x for each of them.)", CAP))
    s.append(figure(fig("fig1_flow.png"), 0.98))
    s.append(P(f"<b>Figure 1.</b> Optical flow (OpenCV Farneback, the exact settings of the handout) on the first 50 training videos, 950 frame pairs. "
               f"Pooled kept pixels (intensity &gt; 0.2 in both frames, flow magnitude &gt; 0.5 px): <b>{f1['pixels_kept']:,}</b>; "
               f"median speed <b>{f1['median_speed']:.2f} pixels per frame</b>. (a) direction θ with the uniform level 1/18; (b) speed, with the median, 2× and 3× the median, and model A's reach (Q2.3) marked.", CAP))
    s.append(P("Q2.1", Q))
    s.append(P(f"<b>Directions.</b> Mirroring maps θ to 180° − θ and a 90° anticlockwise rotation maps θ to θ + 90°; a histogram that is flat is unchanged by both. "
               f"Figure 1(a) is roughly flat but lumpy: bins run from {f1['min_bin']:.3f} to {f1['max_bin']:.3f} against 1/18 = 0.056. With only 50 videos (about 100 digit trajectories) this is mostly sampling noise: "
               f"the total-variation distance between the histogram and its mirrored version is {f1['TV_hist_vs_mirrored']:.2f} and for the rotated version {f1['TV_hist_vs_rot90']:.2f} (10° bins), while two independent halves of the same data "
               f"(25 versus 25 videos) are already {f1['TV_split_half_25_vs_25']:.2f} apart. As a check beyond Figure 1, on 1 000 videos the histogram is much flatter "
               f"(bins {f1['supp_min_bin']:.3f}–{f1['supp_max_bin']:.3f}, distance to uniform {f1['supp_TV_to_uniform']:.2f}). So <b>mirrored and rotated trajectories are statistically familiar</b> to a model trained on this data; "
               f"the training data is (nearly) symmetric in <i>motion</i>."))
    s.append(P(f"<b>Speeds.</b> The typical speed is the median {f1['median_speed']:.2f} px/frame (90th percentile {f1['p90_speed']:.1f}; on 1 000 videos the 99th percentile is {f1['supp_p99_speed']:.1f}). "
               f"Speed-ups are therefore 2× ≈ {2*f1['median_speed']:.1f} and 3× ≈ {3*f1['median_speed']:.1f} px per frame, well outside anything in the training videos, so fast motion is <b>unfamiliar</b>, unlike mirror and rotation. "
               f"<b>Overlaps.</b> Farneback is unreliable where the two digits overlap, but those pixels are a small share of the data and are partly removed by the intensity and magnitude filters. "
               f"Dropping every frame pair in which the thresholded digits merge into one blob (leaving {f1['no_merge_pairs']} of 950 pairs) changes the median speed only from {f1['median_speed']:.2f} to {f1['no_merge_median_speed']:.2f} "
               f"and the mirror and rotation distances from {f1['TV_hist_vs_mirrored']:.2f}/{f1['TV_hist_vs_rot90']:.2f} to {f1['no_merge_TV_hist_vs_mirrored']:.2f}/{f1['no_merge_TV_hist_vs_rot90']:.2f}, "
               f"within the noise level above. The overlap bias is not large enough to change the answer."))
    s.append(P("Q2.2", Q))
    s.append(P("<b>Mirror</b>: changes the <i>appearance</i> (a mirrored 2 is not a 2) but, given Q2.1, not the motion statistics. <b>Rotation by 90°</b>: also changes only appearance, and more strongly (strokes that were vertical become horizontal; "
               "the patch layout of the network no longer matches), while motion statistics stay the same. <b>Speed-up</b>: changes the <i>motion</i> statistics (speeds the model never saw) and leaves digit appearance unchanged. "
               "Hence the three transformations form a controlled experiment: mirror and rotation vary appearance with motion held (statistically) fixed, speed varies motion with appearance held fixed, so a failure can be assigned to one cause."))

    # ------------------------------------------------------------------ 2. baseline
    s.append(P("2. Baseline: ConvLSTM", H2))
    s.append(P("Model A was implemented exactly as specified (patch folding with <font name='DVM'>pixel_unshuffle(x, 4)</font>, two ConvLSTM layers with 64 channels and 3×3 kernels, one convolution producing [i, f, o, g], "
               "a 1×1 output convolution, <font name='DVM'>pixel_shuffle</font>, sigmoid; 9-step rollout feeding back its own predictions, steps 5–9 are the forecast). It has <b>480 784</b> trainable parameters, as required."))
    s.append(P("Q2.3", Q))
    s.append(P("One cell of the folded 16×16 grid is 4 original pixels, so each 3×3 convolution moves information by at most one cell = <b>4 px</b> (centre to centre). Within one time step τ there are three kinds of path to the prediction x̂<sub>τ+1</sub>: "
               "(a) new input → layer-1 convolution (4 px) → h<super>(1)</super><sub>τ</sub> → layer-2 convolution on its input part (4 px) → 1×1 output (0 px): <b>8 px</b>; "
               "(b) previous layer-1 state h<super>(1)</super><sub>τ−1</sub> → layer-1 convolution on its recurrent part (4 px) → layer-2 input convolution (4 px): <b>8 px</b>; "
               "(c) previous layer-2 state h<super>(2)</super><sub>τ−1</sub> → layer-2 recurrent convolution (4 px) → output: 4 px. The longest path is two convolutions, so "
               "<b>the reach of model A is 2 × 4 = 8 original pixels per time step</b> (the 1×1 convolution, pixel_unshuffle and pixel_shuffle add nothing). "
               "Compared with Figure 1(b): normal motion (median 3.3 px) is well inside 8 px; 2× (6.6 px, fastest digits ≈ 9 px) is near the limit; 3× (10 px) is beyond it."))
    s.append(P("<b>Table 2.</b> Per-horizon test MSE (h = forecast horizon), E<super>(5)</super> and trainable parameters at 1× with t = 1. Model A: mean (seed 0, seed 1).", CAP))
    s.append(T.table2())
    s.append(Spacer(1, 4))
    s.append(figure(fig("fig2_baseline.png"), 0.95))
    s.append(P("<b>Figure 2.</b> Model A (seed 0): inputs and true future frames (first row of each video) and the prediction (second row), for three test videos picked by a rule that uses only the ground truth: "
               "videos 7 and 12 are the first with two separate digits in all ten frames; video 1 is the first whose two digits are clean in the inputs and merge into one blob in at least two target frames (overlap).", CAP))
    t2 = J("table2.json")
    e5 = lambda k: t2[k]["E5_1x"]
    s.append(P("Q2.4", Q))
    s.append(P(f"Model A is better than both reference predictors in MSE: E<super>(5)</super> = {fmt(e5('A_0'))} (seed 0) and {fmt(e5('A_1'))} (seed 1), against {fmt(e5('blank'))} for Blank and {fmt(e5('copy'))} for Copy-last. "
               f"Copy-last is worse than Blank because a copied digit in the wrong place is penalised twice (where it was and where it should be). "
               f"<b>Error growth:</b> Blank is flat. Copy-last grows by {100*(t2['copy']['mse_h'][4]/t2['copy']['mse_h'][0]-1):.0f}% from h = 1 to h = 5 ({fmt(t2['copy']['mse_h'][0])} to {fmt(t2['copy']['mse_h'][4])}), because digits keep moving. "
               f"Model A seed 0 grows by {100*(t2['A_0']['mse_h'][4]/t2['A_0']['mse_h'][0]-1):.0f}% ({fmt(t2['A_0']['mse_h'][0])} to {fmt(t2['A_0']['mse_h'][4])}), a slower but steady rise. "
               f"Seed 1 is almost flat ({fmt(t2['A_1']['mse_h'][0])} to {fmt(t2['A_1']['mse_h'][4])}): it has found only a static blur and uses almost no motion information (large seed difference at this training length). "
               f"<b>Figure 2</b> shows what 'better than Blank' means: A predicts soft grey blobs at about the right places that drift in the right direction and fade with the horizon, not sharp digits; blur lowers MSE but the digit identity is lost. "
               f"<b>Overlap</b> (video 1): as the two digits approach each other the two blobs merge into one elongated blob and A has no way to say which digit continues where; the true digits pass through each other and re-appear."))
    s.append(P("Q2.5 (quoted exactly as committed before the experiments of Section 3)", Q))
    s.append(P(pred_text("Q2.5")))
    s.append(P("<b>Outcome:</b> mirror and rotation passed as predicted (small gaps, Table 3). The speed prediction was only partly right: A does lose accuracy with speed, but at 3× it is not 'as bad as Copy-last' "
               "because blurred predictions are rewarded by MSE (Table 4); it is almost as bad as Blank. See Q2.9.", CAP))

    # ------------------------------------------------------------------ 3. testing the baseline
    s.append(P("3. Testing the Baseline", H2))
    s.append(P("<b>Table 3.</b> E<super>(5)</super> and the consistency gap C and transformed error E under mirror (M) and rotation (R), 1× and t = 1 (mean (seed 0, seed 1)). Blank and Copy-last are deterministic.", CAP))
    s.append(T.table3())
    s.append(Spacer(1, 4))
    items = J("fig3_items.json")
    gif = lambda tn, i: f"Question2/gifs/fig3_A_{tn}_test{i}.gif"
    s.append(figure(fig("fig3_symmetry_A.png"), 1.0))
    s.append(P("<b>Figure 3.</b> Model A (seed 0) on the two test videos with the largest per-video C<sub>M</sub> (top row: videos "
               f"{items['fig3_items'][0][1]} and {items['fig3_items'][1][1]}) and the two with the largest C<sub>R</sub> (bottom row: videos {items['fig3_items'][2][1]} and {items['fig3_items'][3][1]}); "
               "video 929 is among the top two for both, so it appears twice. Rows: (1) X | f(X); (2) T(X) | T(Y); (3) T(X) | f(T(X)); (4) T(X) | T(f(X)); columns 1–5 inputs, columns 6–10 forecasts h = 1…5. "
               "Animated GIFs (the same four rows stepping through h = 1…5): " +
               ", ".join(f"<font name='DVM'>{gif(tn, i)}</font>" for tn, i in items["fig3_items"]) + ".", CAP))
    s.append(figure(fig("fig4_hist_A.png"), 0.85))
    s.append(P("<b>Figure 4.</b> Per-video C<sub>M</sub> and C<sub>R</sub> of model A (seed 0) over the 1 000 test videos (log counts); the videos of Figure 3 are marked.", CAP))
    s.append(P("Q2.6", Q))
    s.append(P(f"Blank and Copy-last have C = 0 <i>exactly</i> (a zero map and a copy map commute with any pixel permutation) but E<sub>M</sub> = E<sub>R</sub> = E<super>(5)</super> = {fmt(t34['blank']['E5_1x'])} and {fmt(t34['copy']['E5_1x'])}: perfectly consistent and useless. "
               f"So C measures only whether a model treats the transformed scene like the original, not whether it is right; a constant or blurry output can be made consistent at no cost. "
               f"E<sub>T</sub> must therefore always be reported next to C<sub>T</sub>. Model A is the same case in milder form: C<sub>M</sub> and C<sub>R</sub> ({fmt(a0['C_M'])} and {fmt(a0['C_R'])} for seed 0) are about 100× smaller than E ({fmt(a0['E_M'])}, {fmt(a0['E_R'])}) "
               f"and E<sub>M</sub>, E<sub>R</sub> are equal to E<super>(5)</super> = {fmt(a0['E5_1x'])} to three digits: the model is not hurt by mirroring or rotating, but it is also not accurate enough on any of them to expose a failure."))
    v7 = ex[0]["fig3_videos"]["M_929"]
    s.append(P("Q2.7", Q))
    s.append(P(f"<b>Video 929, mirror (Figure 3, top-left).</b> Rows 3 and 4 differ (MSE between them {fmt(v7['mse_row3_vs_row4'])}, about 5% of the error against the truth, {fmt(v7['mse_row3_vs_truth'])}), but in a clear way. "
               f"The truth (row 2) is two digits sliding down, with the same ink mass in all frames. In row 4, T(f(X)), the model has the digits in separate blobs: the upper blob fades and the lower one moves down; its total brightness falls from {v7['row4_T_of_fX']['mass'][0]:.0f} to {v7['row4_T_of_fX']['mass'][4]:.0f} "
               f"(truth {v7['truth_TY']['mass'][0]:.0f}) and its centre of mass moves {v7['row4_T_of_fX']['centroid_rc'][4][0]-v7['row4_T_of_fX']['centroid_rc'][0][0]:.1f} px down (truth {v7['truth_TY']['centroid_rc'][4][0]-v7['truth_TY']['centroid_rc'][0][0]:.1f}). "
               f"In row 3, f(T(X)), the two digits blur into <b>one tall blob that barely moves and grows</b>: brightness rises from {v7['row3_f_of_TX']['mass'][0]:.0f} to {v7['row3_f_of_TX']['mass'][4]:.0f} and the centre of mass moves only "
               f"{v7['row3_f_of_TX']['centroid_rc'][4][0]-v7['row3_f_of_TX']['centroid_rc'][0][0]:.1f} px. So when it sees the mirrored scene the model's account changes from 'two digits, one leaving' to 'one smeared object that stalls', "
               f"even though the true motion is the mirror image of the first scene. (Both rows are wrong against the truth, {fmt(v7['mse_row3_vs_truth'])} and {fmt(v7['mse_row4_vs_truth'])}.)"))
    s.append(P("Q2.8", Q))
    s.append(P(f"<b>Slightly, for seed 0; not for seed 1.</b> Seed 0: C<sub>R</sub> = {fmt(a0['C_R'])} is {a0['C_R']/a0['C_M']:.2f}× C<sub>M</sub> = {fmt(a0['C_M'])}, and C<sub>R</sub> is larger than C<sub>M</sub> on {100*ex[0]['paired_C_R_gt_C_M_fraction']:.0f}% of the test videos (median ratio {ex[0]['median_C_R_over_C_M']:.2f}); "
               f"E<sub>R</sub> = {fmt(a0['E_R'])} is marginally above E<sub>M</sub> = {fmt(a0['E_M'])}. Seed 1 shows no difference (C<sub>R</sub> = {fmt(a1['C_R'])}, C<sub>M</sub> = {fmt(a1['C_M'])}), as expected for a nearly static predictor. "
               f"By Q2.1 the <i>motion</i> statistics are the same for both transformations (flat direction histogram), so the motion cannot explain the difference. By Q2.2, rotation changes the <i>appearance</i> more than mirroring "
               f"(stroke orientation, the patch layout), so the extra gap for seed 0 is attributed to appearance. The size of the effect is small relative to the error, and it rests on one informative seed of an under-trained model."))
    s.append(P("<b>Table 4.</b> Faster motion, t = 1: E<super>(2)</super> (first two horizons only, for comparability across speeds) at 1×, 2×, 3× and E<super>(5)</super> at 1× and 2×.", CAP))
    s.append(T.table4())
    s.append(Spacer(1, 4))
    vi = J("fig5_video.json")["fig5_video"]
    s.append(figure(fig("fig5_speed_A.png"), 0.82))
    s.append(P(f"<b>Figure 5.</b> Test video {vi} (digits do not overlap in any displayed true frame; the selection used only the ground truth). Model A (seed 0): five inputs, true h = 1, 2 (first row of each speed) and predicted h = 1, 2 (second row) at 1×, 2× and 3×.", CAP))
    r3 = {k: ex[0][k] for k in ("1x", "2x", "3x")}
    s.append(P("Q2.9", Q))
    e2 = lambda f_: (a0[f_], a1[f_])
    s.append(P(f"<b>Where it falls apart.</b> E<super>(2)</super> of model A (seed 0) rises from {fmt(a0['E2_1x'])} at 1× to {fmt(a0['E2_2x'])} at 2× (+{100*(a0['E2_2x']/a0['E2_1x']-1):.0f}%) and {fmt(a0['E2_3x'])} at 3× (+{100*(a0['E2_3x']/a0['E2_1x']-1):.0f}%), "
               f"which is {100*a0['E2_3x']/t34['blank']['E2_3x']:.0f}% of the Blank error {fmt(t34['blank']['E2_3x'])}: at 3× the forecast has almost no skill left, at 2× it is clearly degraded already. "
               f"(Seed 1, the near-static predictor, is already at {fmt(a1['E2_1x'])} at 1× and stays there.) "
               f"<b>Against the reach estimate (8 px, Q2.3):</b> reach predicted trouble only beyond about 2.4× the typical speed, i.e. at 3×. The data say trouble starts already at 2×. "
               f"What my estimate missed is that reach limits how far a single update can connect old and new positions, but the model also has to <i>measure</i> speed from the inputs and use it. "
               f"The measurement shows it does not: the correlation of its h = 1 prediction with the last input frame stays at {r3['1x']['h1_corr_pred_vs_last_input']:.2f}, {r3['2x']['h1_corr_pred_vs_last_input']:.2f} and {r3['3x']['h1_corr_pred_vs_last_input']:.2f} at 1×, 2×, 3×, "
               f"while its correlation with the true frame falls from {r3['1x']['h1_corr_pred_vs_truth']:.2f} to {r3['2x']['h1_corr_pred_vs_truth']:.2f} to {r3['3x']['h1_corr_pred_vs_truth']:.2f} "
               f"(copying the last input would give {r3['1x']['h1_corr_last_input_vs_truth']:.2f}, {r3['2x']['h1_corr_last_input_vs_truth']:.2f}, {r3['3x']['h1_corr_last_input_vs_truth']:.2f}). "
               f"(The digits are about 20 px wide, so a digit that jumps 7–10 px still overlaps its old position and a convolutional state can partly follow it, which is another reason why reach is a soft limit.) "
               f"<b>What the failure looks like (Figure 5):</b> grey blur and a <b>stall</b>. At 1× the two blobs sit close to the true digits. At 2× and 3× they are still centred near where the digits were in the last input and move only a few pixels, "
               f"while the true digits have jumped ~7–10 px; there is no duplicated digit and no vanishing digit in this video, only blobs that lag behind. The blobs' total brightness is {r3['3x']['h1_pred_mass_over_truth_mass']:.1f}× the true ink mass at 3× "
               f"(it is {r3['1x']['h1_pred_mass_over_truth_mass']:.1f}× at 1×): the model spreads its uncertainty. A stall suggests the model is applying the typical 1× displacement it learned (≈ 3 px per frame) instead of the observed one: it behaves like a <i>memorised speed prior</i> plus blur, not a measured velocity."))
    s.append(P("Q2.10", Q))
    s.append(P("<b>No.</b> The two explanations make the same prediction for A: unfamiliar speeds and a limited reach both make fast clips fail, and A was trained only at 1× with an 8 px reach, so both are present at once. "
               "An experiment that separates them changes only one of them: (i) train the <i>same</i> architecture (same 8 px reach) on clips at 1×, 2× and 3× (this is model C): if it then handles 3× well, the failure was unfamiliar motion; if 3× stays bad, reach (or capacity) is the limit. "
               "(ii) Train a model with a larger reach on 1× clips only and test it at 2× and 3×: if it improves although it never saw fast motion, reach was the bottleneck. Comparing A, C and the larger-reach model D gives the answer (Section 4)."))
    return s
