"""Question 2 section of the report, part 2 (What can data fix?, The Final Model)."""
import os
from rl_helpers import *
import q2_tables as T
from q2_tables import J, fmt, ms
from q2_text import pred_text

HERE = os.path.dirname(os.path.abspath(__file__))
fig = lambda n: os.path.join(HERE, n)


def M(name, seed, sym=False):
    return J(f"metrics_{name}_{seed}" + ("_sym" if sym else "") + ".json")


def mean2(name, f, sym=False):
    return (M(name, 0, sym)[f] + M(name, 1, sym)[f]) / 2


def build():
    s = []
    sel = {k: J(f"select_d_{k}.json") for k in ("p8_k3", "p4_k5_h32")}
    ch = J("d_choice.json")
    ex = {k: J(f"extra_{k}.json") for k in ("A_0", "C_0", "C_1", "D_0", "D_1")}
    tm = J("timing.json")
    ep = T.logged_epoch_times()

    # ------------------------------------------------------------------ 4
    s.append(P("4. What Can Data Fix?", H2))
    s.append(P("Q2.11 (quoted exactly as committed before B, C and D were trained)", Q))
    s.append(P(pred_text("Q2.11")))
    s.append(P("<b>Table 5.</b> Models A–D (6 epochs each; mean (seed 0, seed 1)). E<super>(5)</super>, C and E at 1×, t = 1; E<super>(2)</super> at 1×, 2×, 3×. "
               "Training time per epoch (250 steps) and inference time (1 000 test videos, batch 32) were measured alone on the CPU after training (AMD Ryzen 5 5500U, 6 threads).", CAP))
    s.append(T.table56([("A", "A", False, "A"), ("B", "B", False, "B"), ("C", "C", False, "C"), ("D", "D", False, "D")]))
    s.append(P(f"Times logged <i>during</i> training are not comparable: A ran with one other job ({ep['A'][0]:.0f} s and {ep['A'][1]:.0f} s per epoch); B, C, D ran with up to three other jobs, and the laptop went to sleep "
               f"once, which inflated some epochs to hours. Model D (patch 8) is about {tm['A_0']['train_s_per_epoch_solo']/tm['D_0']['train_s_per_epoch_solo']:.1f}× cheaper per epoch than A.", CAP))
    s.append(P("Q2.12", Q))
    s.append(P(f"<b>Model D</b> = model A with patch size 8: <font name='DVM'>pixel_unshuffle(x, 8)</font> gives 64 × 8 × 8; two ConvLSTM layers with 64 hidden channels and 3×3 kernels; a 1×1 convolution 64→64; <font name='DVM'>pixel_shuffle(·, 8)</font> and sigmoid; "
               f"same 9-step rollout without teacher forcing, same training. It has <b>594 496</b> parameters (≤ 721 176; 1.24× model A). No attention. "
               f"<b>Reach</b> (same conventions as Q2.3): one cell is now 8 px. (a) new input → layer-1 conv (8 px) → layer-2 conv (8 px) → 1×1 output: 16 px; (b) previous layer-1 state → layer-1 conv (8) → layer-2 conv (8): 16 px; (c) previous layer-2 state → layer-2 conv (8): 8 px. "
               f"Longest path = 2 × 8 = <b>16 px per step, strictly larger than A's 8 px</b>. It covers even the fastest 3× digits (99th percentile of training speed 4.6 px/frame × 3 ≈ 14 px). "
               f"<b>Design choice on validation only:</b> two candidates with reach 16 px were trained for 250 steps (speed-augmented, seed 0) and scored on the validation split by E<super>(5)</super>@1× + E<super>(2)</super>@2× + E<super>(2)</super>@3×: "
               f"patch 8 with 3×3 kernels (594 496 params) {sel['p8_k3']['score']:.4f}; patch 4 with 5×5 kernels and 32 hidden channels (359 184 params) {sel['p4_k5_h32']['score']:.4f}. "
               f"Both were still at the static-blob plateau (E ≈ 0.039–0.040 at every speed), so this is almost a tie; I took the lower score, which is also the cheaper model per step. "
               f"The consequence is visible below: model D is coarse (8×8 grid) and learned more slowly in 6 epochs."))
    s.append(P("Q2.13", Q))
    a0, c0, c1, d0, d1 = (M("A", 0), M("C", 0), M("C", 1), M("D", 0), M("D", 1))
    s.append(P(f"Caveat first: after 6 epochs seed 1 of A and B and seed 0 of D are close to a static blur (E(2) ≈ {fmt(M('A',1)['E2_1x'])}–{fmt(d0['E2_1x'])} at every speed), so only some seeds carry information. Labels:"))
    s += bullets([
        f"<b>Mirror and rotation failure → none to label, and in particular not 'appearance'.</b> For A, C<sub>M</sub> = {fmt(a0['C_M'])}, C<sub>R</sub> = {fmt(a0['C_R'])} are about 1% of E; model B (symmetry augmentation) does not reduce them "
        f"(seed 0: {fmt(M('B',0)['C_M'])} and {fmt(M('B',0)['C_R'])}), and E<sub>M</sub>, E<sub>R</sub> equal E<super>(5)</super> for A, B, C and D. At this training length there is no appearance-driven failure to remove (Table 5, columns C<sub>M</sub> … E<sub>R</sub>).",
        f"<b>Loss of accuracy with speed → partly unfamiliar motion.</b> Speed augmentation (A→C) lowers E<super>(2)</super> at 3× from {fmt(a0['E2_3x'])} to {fmt(c0['E2_3x'])} (seed 0), and at 2× from {fmt(a0['E2_2x'])} to {fmt(c0['E2_2x'])}; "
        f"the gap between 1× and 3× shrinks from {fmt(a0['E2_3x']-a0['E2_1x'])} (A) to {fmt(c0['E2_3x']-c0['E2_1x'])} (C seed 0) and {fmt(c1['E2_3x']-c1['E2_1x'])} (C seed 1). "
        f"The correlation of the h = 1 prediction with the true frame at 3× rises from {ex['A_0']['3x']['h1_corr_pred_vs_truth']:.2f} (A) to {ex['C_0']['3x']['h1_corr_pred_vs_truth']:.2f} (C seed 0) and {ex['C_1']['3x']['h1_corr_pred_vs_truth']:.2f} (C seed 1). The speed prior learned from 1× data was a large part of the failure.",
        f"<b>Remaining 3× gap → architectural reach cannot be confirmed.</b> If C were still bad at 3× while D were good, the failure would be reach. Here D is <i>not</i> clearly better than C: at 3× D seed 1 gives {fmt(d1['E2_3x'])} against {fmt(c0['E2_3x'])} and {fmt(c1['E2_3x'])} for C, "
        f"and D seed 0 never left the static blur ({fmt(d0['E2_3x'])}). With 6 epochs I can neither show nor rule out a reach effect; D's coarser grid and slower learning confound the comparison.",
    ])
    s.append(P("<b>Why more data cannot carry information further than the architecture allows.</b> Training data only changes the weights; the set of pixels that can influence an output in one update is fixed by kernel size, patch size and depth. "
               "If a digit moves farther than the reach, the old and new positions never meet inside one convolution, whatever the weights are, so no amount of fast training clips can create a path between them. "
               "Data can teach the network to <i>use</i> the available reach (what C does for speeds within it), but only a larger reach (D) can extend it."))
    s.append(P("Q2.14", Q))
    s.append(P(f"<b>Model C vs A at 1×:</b> mean E<super>(5)</super> is {fmt(mean2('C','E5_1x'))} against {fmt(mean2('A','E5_1x'))}; seed 0 is worse ({fmt(c0['E5_1x'])} vs {fmt(a0['E5_1x'])}, +{100*(c0['E5_1x']/a0['E5_1x']-1):.0f}%) and seed 1 better ({fmt(c1['E5_1x'])} vs {fmt(M('A',1)['E5_1x'])}, because A seed 1 was stuck). "
               f"So there is a small cost for the informative seed, in return for a visibly better 2× and 3× (Q2.13): I judge it worth it. <b>Model D vs A:</b> mean E<super>(5)</super> {fmt(mean2('D','E5_1x'))} (seed 0 {fmt(d0['E5_1x'])}, seed 1 {fmt(d1['E5_1x'])}), about {100*(mean2('D','E5_1x')/mean2('A','E5_1x')-1):.0f}% worse than A, "
               f"and one seed collapsed; the larger reach did not pay off in 6 epochs, so for this budget D was not worth it (it is, however, 3× cheaper to train). <b>B vs A:</b> no change ({fmt(mean2('B','E5_1x'))} vs {fmt(mean2('A','E5_1x'))})."))

    # ------------------------------------------------------------------ 5
    s.append(P("5. The Final Model", H2))
    s.append(P("<b>Why f̄ has C<sub>M</sub> = C<sub>R</sub> = 0 for any f.</b> Let h ∈ G. Each g ∈ G is a pixel permutation (linear). "
               "f̄(h(X)) = ⅛ Σ<sub>g∈G</sub> g<super>−1</super>(f(g(h(X)))). Put g′ = g∘h; as g runs over G so does g′ (G is a group), and g<super>−1</super> = h∘g′<super>−1</super>. "
               "Then f̄(h(X)) = ⅛ Σ<sub>g′</sub> h(g′<super>−1</super>(f(g′(X)))) = h(f̄(X)), since h is linear and commutes with the average. Taking h = M and h = R (both in G) gives f̄(T(X)) = T(f̄(X)), i.e. C<sub>M</sub> = C<sub>R</sub> = 0, "
               "whatever f is. In floating point the sum is evaluated in a different order for the two sides, hence values of order 10<super>−17</super> instead of 0."))
    s.append(P("<b>Choice of final model.</b> Symmetrised model D (as the handout's default). D+B was not trained (optional, no compute left), and Table 5 gives no reason for it: "
               "B did not improve consistency or accuracy over A. <b>Caution:</b> D seed 0 never left the static blur, so the seed-0 figures below show an almost constant forecast; the seed-1 model does learn some motion."))
    s.append(P("<b>Table 6.</b> Table 5 columns for symmetrised B and symmetrised D (the final model). The training-time column is the training time of the underlying model (B or D). C<sub>M</sub> and C<sub>R</sub> in scientific notation.", CAP))
    s.append(T.table56([("sym B", "B", True, "B"), ("sym D (final)", "D", True, "D")]))
    items = J("fig3_items.json")["fig3_items"]
    s.append(Spacer(1, 4))
    s.append(figure(fig("fig6a_symmetry_final.png"), 1.0))
    s.append(P("<b>Figure 6(a).</b> Figure 3 repeated with the final model (symmetrised D, seed 0) on the same four panels. Rows 3 and 4 are identical up to 10<super>−17</super>. "
               "GIFs: " + ", ".join(f"<font name='DVM'>Question2/gifs/fig6a_Dsym_{tn}_test{i}.gif</font>" for tn, i in items) + ".", CAP))
    vi = J("fig5_video.json")["fig5_video"]
    s.append(figure(fig("fig6b_speed_final.png"), 0.8))
    s.append(P(f"<b>Figure 6(b).</b> Figure 5 repeated with the final model (symmetrised D, seed 0) on test video {vi}: forecasts at h = 1, 2 for 1×, 2×, 3×.", CAP))
    s.append(P("Q2.15", Q))
    bs0, bs1 = M("B", 0), M("B", 1)
    ss0, ss1 = M("B", 0, True), M("B", 1, True)
    s.append(P(f"<b>Consistency.</b> Learned symmetry (B) is approximate: C<sub>M</sub> = {fmt(bs0['C_M'])}, {fmt(bs1['C_M'])} and C<sub>R</sub> = {fmt(bs0['C_R'])}, {fmt(bs1['C_R'])} for seeds 0 and 1, no smaller than the un-augmented A ({fmt(a0['C_M'])}, {fmt(a0['C_R'])}). "
               f"Built-in symmetry is exact: C<sub>M</sub> = {fmt(ss0['C_M'])} and C<sub>R</sub> = {fmt(ss0['C_R'])} (sym B seed 0); {fmt(M('D',0,True)['C_M'])} and {fmt(M('D',0,True)['C_R'])} (sym D seed 0), zero up to floating point, because the average over a group is invariant under the group by construction (proof above), for any f. "
               f"<b>Accuracy.</b> On original clips symmetrisation changes E<super>(5)</super> very little: B seed 0 {fmt(bs0['E5_1x'])} → {fmt(ss0['E5_1x'])}, seed 1 {fmt(bs1['E5_1x'])} → {fmt(ss1['E5_1x'])}; D seed 1 {fmt(d1['E5_1x'])} → {fmt(M('D',1,True)['E5_1x'])}. "
               f"On transformed clips E<sub>M</sub> and E<sub>R</sub> equal E<super>(5)</super> for every model, so there is nothing to gain or lose there either. "
               f"<b>Cost.</b> Inference needs 8 forward passes: {tm['B_0']['inference_s']:.1f} s → {tm['B_0_sym']['inference_s']:.1f} s for B ({tm['B_0_sym']['inference_s']/tm['B_0']['inference_s']:.1f}×) and {tm['D_0']['inference_s']:.1f} s → {tm['D_0_sym']['inference_s']:.1f} s for D; training cost is unchanged, "
               f"whereas learned symmetry (B) costs nothing at inference but gives no guarantee. <b>Blur.</b> Averaging eight predictions hurts when the base model's predictions are sharp but not equivariant: the average of eight slightly displaced sharp digits is blurred and loses accuracy. "
               f"It did not hurt here, because these under-trained models already predict blur, and the averaging even lowered the error slightly."))
    s.append(P("Q2.16", Q))
    s.append(P(f"<b>No.</b> Video 929 (Q2.7) now gives identical rows 3 and 4 (Figure 6a), so the <i>inconsistency</i> is fixed by symmetrisation. But the final model (D seed 0) outputs an almost constant dim image at every horizon and does not move any digit, "
               f"so the stalled, smeared account is not replaced by a correct one (E<super>(5)</super> = {fmt(M('D',0,True)['E5_1x'])} for the whole test set). Symmetrisation fixed the disagreement between rows 3 and 4; it cannot fix a base model that has not learned the motion, and the larger reach of D and the speed augmentation did not fix that within 6 epochs."))
    s.append(P("Q2.17 Final verdict", Q))
    s.append(P("<b>No: this final model has not learned motion.</b> Its seed-0 forecasts are almost constant, so zero consistency (C = 10<super>−17</super>) is the trivial kind seen with Blank (Q2.6), and E is near the Blank error. "
               "A skeptic would want: sharp, correctly moved digits at 1× that beat Copy-last and Blank by a wide margin on both seeds; the same quality after mirror, rotation and 2×/3× speed-up (E<sub>T</sub> and E<super>(2)</super> close to the 1× values); "
               "and correct behaviour at wall bounces and digit overlaps. The test it has clearly not passed is accuracy on the original clips itself, and with it the speed test, since E<super>(2)</super> at 3× is no better than at 1×. "
               "A 30-epoch run is needed before any claim about motion."))
    return s
