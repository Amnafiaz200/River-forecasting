"""Question 1 section of the report (flowables)."""
import os
from rl_helpers import *

HERE = os.path.dirname(os.path.abspath(__file__))
fig = lambda n: os.path.join(HERE, n)


def build():
    s = []
    s.append(P("Question 1 — River Forecasting Proposal", H1))
    s.append(P("<b>Proposal in one paragraph.</b> Every 10 minutes the system predicts the level at S1–S4 one hour ahead with a "
               "<b>graph space-time network</b>. The river is a directed graph (Reservoir → S1 → S2 → S3 → S4, T1 → S2, an unobserved "
               "node U → S3). Each node has a 13.5-hour history. Blocks alternate a causal temporal convolution (T) with a "
               "<b>travel-time-aware</b> message-passing step (S) along the river. Announced releases are placed on the "
               "future part of the reservoir's time axis. The forecast is the last observed level plus a learned change, "
               "with 10/50/90 % quantiles. Missing data, late data and release changes are handled in the input layer, not by "
               "retraining."))

    # ------------------------------------------------------------------ 1
    s.append(P("1. Architecture diagram", H2))
    s.append(figure(fig("fig_q1_architecture.png"), 1.0))
    s.append(P("<b>Figure Q1.1.</b> Architecture. Sizes: B = batch (1 at run time, many in training); N = 7 graph nodes "
               "(Res, S1, T1, S2, U, S3, S4); L = 72 history steps (12 h) and H = 6 future steps (1 h), so T = 78; F = 10 input features per node; "
               "d = 64 hidden channels; E = 6 directed edges (Res→S1, S1→S2, T1→S2, S2→S3, U→S3, S3→S4); every node also keeps its own state through the W<sub>1</sub>z<sub>i</sub> term of Eq. 5b. "
               "<b>Alignment / reshaping:</b> (i) all series are snapped to one 10-min grid and stacked on a common time axis of length 78; "
               "(ii) the two rain gauges [B×72×2] are projected to node space [B×7×72] by the weights ω of Eq. 2, then right-padded with zeros and mask 0 to length 78; "
               "(iii) station levels [B×72×5] are scattered to the nodes (T1→T1, S1–S4→S1–S4; Res and U get zeros and their own mask); "
               "(iv) reservoir actual [B×72] and announced [B×78] releases are written on the Res node's time axis; "
               "(v) channels are concatenated to [B×7×78×10] and projected to d = 64. The ST stack keeps [B×7×78×64] throughout; the head keeps only nodes S1–S4 and the last six time positions, "
               "giving [B×4×6×64] → forecasts [B×6×4×3] (6 horizons × 4 stations × 3 quantiles).", CAP))
    s.append(figure(fig("fig_q1_graph.png"), 0.86))
    s.append(P("<b>Figure Q1.2.</b> The river as a directed graph and the number of hops from each source to S4. Because T1 joins before S2 and U before S3, "
               "four stacked ST blocks (4 hops) are enough for information from the reservoir to reach S4.", CAP))

    # ------------------------------------------------------------------ 2
    s.append(P("2. Assumptions", H2))
    s.append(P("<b>Supplied facts (not assumptions):</b> F1 levels at S1–S4 and T1 every 10 min; F2 rainfall at R1, R2 with timestamps and gauge locations; "
               "F3 recorded reservoir releases and future release schedules announced before issuance, with announcement and effective times; "
               "F4 station locations, connectivity and distances (stations unevenly spaced); F5 a monitored tributary joins between S1 and S2, "
               "an unmonitored one between S2 and S3; F6 future rain and future levels are unavailable at issuance; F7 travel times and unmonitored inflow are not given; "
               "F8 additional information may be requested but is not guaranteed. <b>Everything below is an assumption.</b>"))
    A = [["ID", "Assumption", "Why it is reasonable / how it affects the design"],
         ["A1", "Record length: ≥ 5 years of 10-min data (~263 000 steps) that include several flood events and wet seasons.",
          "Typical for a national gauge archive. Allows a ~0.25 M-parameter network and a chronological train/val/test split (first 3.5 y / next 0.75 y / last 0.75 y). If the record is under 2 y, shrink d and lean on the physical priors (A6–A8)."],
         ["A2", "All series share one clock (UTC) with error < 1 min; a level is an instantaneous reading at its timestamp; rainfall is an accumulation over the previous 10 min.",
          "Standard telemetry practice. Series are snapped to a common 10-min grid with ±2 min tolerance; no sub-grid interpolation model is needed."],
         ["A3", "Reporting delay is usually ≤ 5 min but sometimes 30 min to several hours; each record also has an arrival (ingest) time.",
          "Needed to know what was available at issuance. Drives the as-of snapshot, the ‘age’ feature δ and delay simulation in training (§5)."],
         ["A4", "1–5 % of readings are missing at random, outages of up to ~2 h occur, multi-day outages are rare; spikes and flat-lines are flagged by QC.",
          "Sets the random-masking rates used in training and the choice of masks + decay imputation (Eq. 1)."],
         ["A5", "Level noise σ ≈ 1–2 cm, 1 cm resolution, no datum shifts within the record.",
          "Below the forecast skill we target, so no explicit sensor-noise model; the pinball loss is robust to outliers."],
         ["A6", "Travel times: reservoir → S1 under ~1 h; each later reach 20 min – 3 h; end-to-end reservoir → S4 ≲ 8 h; shorter at high flow (wave celerity rises with stage).",
          "Not supplied (F7), so the model learns them. It fixes L = 12 h (> 1.5× the longest path), the lag window Λ = 18 steps (3 h) per edge, and a state-dependent celerity in Eq. 5. If a reach is slower than 3 h the lag window is widened."],
         ["A7", "Flow is one-dimensional and downstream; backwater from downstream onto upstream stations is negligible (no tide, no downstream dam).",
          "Justifies a <i>directed</i> graph: messages travel only downstream. If backwater existed we would add upstream edges."],
         ["A8", "Mass balance at confluences: Q<sub>down</sub> ≈ Q<sub>up</sub> + Q<sub>trib</sub> + Q<sub>lateral</sub>; at each station level is a monotone, smooth function of discharge (unknown stage–discharge curve).",
          "Messages entering S2 and S3 are <i>summed</i> (Eq. 5) rather than averaged; non-negative gates keep contributions monotone."],
         ["A9", "T1 is gauged close to its confluence (travel time < 30 min), so T1 is effectively the tributary inflow joining between S1 and S2.",
          "Gives a short edge T1 → S2 with a small expected lag."],
         ["A10", "The unmonitored tributary U has a catchment responding to rain like T1 (lag 1–6 h) plus a slowly varying base flow; it enters between S2 and S3, so it affects S3 and S4 only.",
          "Modelled as a latent node whose state comes from rainfall and a learned base level; the only information about it is the mismatch between S2 and S3, so S3/S4 get wider intervals (§6)."],
         ["A11", "R1 and R2 give 10-min accumulations in mm; their coordinates are known but two gauges only roughly represent the catchment. Rain reaches the river with a lag; a small part is direct channel runoff.",
          "Rain is a node feature from distance weights (Eq. 2); because future rain is unknown (F6), only rain that has already fallen can help the 1-hour forecast."],
         ["A12", "Releases are announced typically 1–24 h ahead as planned discharge with an effective time; a newer announcement replaces an older one for the same effective time; actual release is metered every 10 min. Where nothing is announced for a future step, the current release continues.",
          "Defines the schedule channel with an ‘announced’ flag f<sup>sch</sup> (0 = default persistence, 1 = announced), so the model knows how much to trust it (§5)."],
         ["A13", "Actual release follows the schedule within ~10 %, with gate-actuation delays up to 30 min; unannounced emergency changes are rare but possible.",
          "Past actual and future scheduled releases are separate channels and a deviation feature e = q<sup>act</sup> − q<sup>sch</sup> is kept for the recent past (§5, iv)."],
         ["A14", "Channel geometry and gauge datums are stable; seasonal roughness (vegetation) changes slowly.",
          "Season encodings capture the slow change; retrain yearly or after a major flood reshapes the channel."],
         ["A15", "A forecast is issued on every 10-min tick τ after a 2-min collection cut-off; the target is the level on the grid at τ + 60 min.",
          "Fixes the task in §3 and the as-of rule: only records that have arrived by the cut-off are used."],
         ["A16", "A CPU server runs inference; training is offline and may use a GPU.",
          "Model stays small (§5 feasibility)."]]
    s.append(table(A, [0.05, 0.40, 0.55]))

    # ------------------------------------------------------------------ 3
    s.append(P("3. Forecasting task and data specification", H2))
    s.append(P("<b>Issuance time.</b> τ is any 10-minute grid time. The forecast issued at τ uses only records whose arrival time is ≤ τ + 2 min (A15), "
               "and announcements made ≤ τ. <b>History window.</b> L = 72 steps (12 h): grid times τ−710, …, τ min (A6: the longest path is under 8 h). "
               "<b>Horizon.</b> H = 6 steps = 60 min. The network produces h = 1, …, 6 (τ+10, …, τ+60 min) as auxiliary outputs and the "
               "<b>main product is h = 6</b>. <b>Outputs.</b> Water level in metres at S1, S2, S3, S4 at τ + 60 min as the 10 %, 50 % and 90 % "
               "quantiles; the median is the point forecast, the 90 % quantile feeds flood warnings."))
    D = [["Item", "Symbol and shape", "Axes", "Timestamps / coverage (relative to τ)", "At issuance?"],
         ["Water levels S1–S4, T1", "W ∈ R<super>B×72×5</super> (m)", "batch; time (10-min grid); station (S1,S2,S3,S4,T1)", "τ−710 … τ min", "Yes, only records arrived by τ"],
         ["Observation mask, age", "M ∈ {0,1}<super>B×72×5</super>, Δ ∈ R<super>B×72×5</super> (min)", "as W", "τ−710 … τ", "Yes (derived from arrival log)"],
         ["Rainfall R1, R2", "P ∈ R<super>B×72×2</super> (mm per 10 min); gauge coordinates ρ ∈ R<super>2×2</super>", "batch; time; gauge", "(τ−720, τ] accumulations, stamped at interval end", "Past only. Future rain: NOT available (F6)"],
         ["Reservoir release, actual", "q<super>act</super> ∈ R<super>B×72</super> (m³/s)", "batch; time", "τ−710 … τ", "Yes"],
         ["Reservoir release, announced", "q<super>sch</super> ∈ R<super>B×78</super>, flag f<super>sch</super> ∈ {0,1}<super>B×78</super>", "batch; time (72 past + 6 future)", "τ−710 … τ+60; effective times; announced ≤ τ", "Yes (announcement time ≤ τ only)"],
         ["Static graph", "chainage d ∈ R<super>7</super> (km), adjacency A ∈ {0,1}<super>7×7</super>, edge length Δ ∈ R<super>6</super>", "node; edge", "constant", "Yes"],
         ["Calendar", "sin/cos of day and of year ∈ R<super>B×78×4</super>", "batch; time; 4 channels", "τ−710 … τ+60", "Yes (known in advance)"],
         ["Node-time tensor (model input)", "X ∈ R<super>B×7×78×10</super>", "batch; node (Res,S1,T1,S2,U,S3,S4); time; feature", "τ−710 … τ+60 (future slots masked)", "Yes: built from the rows above"],
         ["Forecast (output)", "Ŷ ∈ R<super>B×6×4×3</super>, main: Ŷ[:,5] ∈ R<super>B×4×3</super> (m)", "batch; horizon h=1…6; station S1–S4; quantile", "τ+10 … τ+60 min", "Produced at τ"],
         ["Target (training only)", "Y ∈ R<super>B×6×4</super> with mask", "batch; horizon; station", "τ+10 … τ+60 min", "NOT available at τ"]]
    s.append(table(D, [0.14, 0.27, 0.20, 0.23, 0.16]))
    s.append(P("<b>Alignment.</b> (a) <i>Sampling times</i>: every series is snapped to the same 10-min grid; a level is the reading nearest to the grid time within ±2 min, rain is summed over (t−10, t], "
               "and the release schedule is a step function evaluated by its <i>effective</i> time (zero-order hold). "
               "(b) <i>Locations</i>: stations are not resampled in space. They sit on a graph with their true chainage and edge lengths, so unequal spacing is represented by the edge feature φ(Δ) and by the travel-time prior in Eq. 5. "
               "(c) <i>Gauges to nodes</i>: R1 and R2 are mapped to nodes by distance weights (Eq. 2). "
               "(d) <i>What is known at τ</i>: all past levels, rain, actual releases; the schedule for τ+10 … τ+60 if announced by τ; the graph and calendar. "
               "<i>Not known</i>: any level or rain after τ, and any later announcement. Training windows are built the same way (as-of τ), so nothing from the future leaks into the inputs. "
               "<b>Batch axis</b>: in training each sample is one issuance time τ (stride 1, shuffled); at run time B = 1."))

    # ------------------------------------------------------------------ 4
    s.append(P("4. Model computations and design justification", H2))
    s.append(P("<b>Notation.</b> i, j index nodes; t a grid step (10 min); w<sub>i</sub>(t) a level, m<sub>i</sub>(t) its mask (1 = valid), δ<sub>i</sub>(t) minutes since the last valid value; "
               "P<sub>k</sub>(t) rain at gauge k; q<super>act</super>(t), q<super>sch</super>(t), f<super>sch</super>(t) release channels; h<sub>i</sub><super>l</super>(t) ∈ R<super>d</super> the hidden state after block l; "
               "Δ<sub>ji</sub> the river distance of edge j→i; 𝒫(i) the parents of node i; 10λ minutes is a lag. Only the four river stations carry a level to forecast."))
    s.append(eq(r"\hat{w}_i(t)=m_i(t)\,w_i(t)+\bigl(1-m_i(t)\bigr)\bigl[\gamma_i(t)\,w_i^{\mathrm{last}}(t)+(1-\gamma_i(t))\,\mu_i\bigr],\quad \gamma_i(t)=\exp\!\bigl(-\mathrm{softplus}(a_i\,\delta_i(t)+b_i)\bigr)", "1"))
    s.append(P("Missing levels decay from the last valid value towards the station mean μ<sub>i</sub> (as in GRU-D); the mask and age are also inputs. "
               "<i>Physical assumption:</i> none. <i>Learned:</i> a<sub>i</sub>, b<sub>i</sub> (how fast a gap loses information).", SMALL))
    s.append(eq(r"r_i(t)=\sum_{k=1}^{2}\omega_{ik}\,P_k(t),\qquad \omega_{ik}=\mathrm{softmax}_k\!\left(-\,\|x_i-\rho_k\|/s\right)", "2"))
    s.append(P("Rain reaches a node with weights that decrease with the distance between the node and the gauge (initialised from coordinates, then fine-tuned; free logits for the latent node U). "
               "The node input is x<sub>i</sub>(t) = [ŵ<sub>i</sub>, m<sub>i</sub>, δ<sub>i</sub>/Δ<sub>0</sub>, r<sub>i</sub>, q<super>act</super>, q<super>sch</super>, f<super>sch</super>, 4 calendar terms] "
               "(release channels are non-zero only on Res, level channels only on S1–S4 and T1). Then h<sub>i</sub><super>0</super>(t) = W<sub>in</sub> x<sub>i</sub>(t) + e<sub>type(i)</sub>, with a learned embedding for "
               "the node type (reservoir, river station, tributary station, latent).", SMALL))
    s.append(eq(r"z_i^{l}(t)=h_i^{l-1}(t)+W_o^{l}\bigl[\,u\odot\sigma(v)\bigr],\qquad [u;v]=\sum_{k=0}^{2}W_k^{l}\,h_i^{l-1}(t-kD_l),\quad D_l\in\{1,3,9,27\}", "3"))
    s.append(P("<b>T block:</b> a causal, dilated, gated convolution along time at each node, with a residual connection. Four blocks give a receptive field of 1 + 2(1+3+9+27) = 81 steps (13.5 h) ≥ L. "
               "It lets each node extract trends, rates of rise and rain accumulations from its own history.", SMALL))
    s.append(eq(r"s_{ji}(t,\lambda)=v^{\top}\tanh\!\bigl(A\,z_j(t-\lambda)+B\,z_i(t)+C\,\phi(\Delta_{ji})\bigr)-\frac{(\lambda-\kappa_{ji}(t))^{2}}{2\varsigma^{2}},\qquad \kappa_{ji}(t)=\frac{\Delta_{ji}}{c_{ji}(t)},\quad c_{ji}(t)=\mathrm{softplus}\bigl(c_0+\beta^{\top}z_j(t)\bigr)", "4"))
    s.append(eq(r"\alpha_{ji}(t,\lambda)=\mathrm{softmax}_{\lambda\in\{0..\Lambda\}}\,s_{ji}(t,\lambda),\quad m_{ji}(t)=\sum_{\lambda=0}^{\Lambda}\alpha_{ji}(t,\lambda)\,W_m\,z_j(t-\lambda),\quad g_{ji}(t)=\sigma\bigl(w_g^{\top}[z_j(t);z_i(t)]+b_{ji}\bigr)", "5a"))
    s.append(eq(r"h_i^{l}(t)=\mathrm{LN}\Bigl(z_i^{l}(t)+W_2\,\mathrm{GELU}\bigl(W_1 z_i^{l}(t)+\sum_{j\in\mathcal{P}(i)}g_{ji}(t)\,m_{ji}(t)\bigr)\Bigr)", "5b"))
    s.append(P("<b>S block (travel-time attention):</b> node i looks at each upstream parent j at several earlier times t−λ and learns <i>which lag</i> to read, "
               "α<sub>ji</sub>(t,λ). The Gaussian term is a physical prior centred on the expected travel time κ<sub>ji</sub> = distance / celerity (in 10-min steps); celerity c<sub>ji</sub> grows with the upstream state "
               "(β), so travel time shortens when the river is high (A6). Messages from all parents are <i>summed</i> (mass balance, A8). Lags satisfy λ ≥ 0, so nothing from the future of node j is used. "
               "On the future positions (t > τ) the released-water channel on Res carries the schedule, and the S block moves it downstream along the learned lags — this is how an announced release influences S1–S4 "
               "within the hour. After four blocks, S4 has read Res (4 hops), T1 (3), U (2) and S1–S3.", SMALL))
    s.append(eq(r"\hat{y}_i^{(q)}(\tau{+}h)=\tilde{w}_i(\tau)+\sigma_i\,\mathrm{MLP}_q\bigl([\,h_i^{4}(\tau{+}h);\,h_i^{4}(\tau)\,]\bigr),\quad h=1..6,\ q\in\{0.1,0.5,0.9\}", "6"))
    s.append(P("<b>Forecast mapping:</b> persistence plus a learned change. w̃<sub>i</sub>(τ) is the last valid level in metres (Eq. 1) and σ<sub>i</sub> the standard deviation of one-hour level changes at that station. "
               "Quantiles are kept ordered by predicting the median and two non-negative offsets (softplus).", SMALL))
    s.append(eq(r"\mathcal{L}=\frac{1}{|\mathcal{B}|}\sum_{b\in\mathcal{B}}\sum_{h=1}^{6}\sum_{i=1}^{4}\sum_{q}\omega_h\,\mu^{y}_{bih}\,\frac{\rho_q\bigl(y_{bih}-\hat{y}^{(q)}_{bih}\bigr)}{\sigma_i}+\lambda_{\mathrm{wd}}\|\theta\|^{2},\quad \rho_q(e)=\max\{qe,(q-1)e\},\ \ \omega_h=\frac{h}{21}", "7"))
    s.append(P("<b>Training objective:</b> pinball (quantile) loss, normalised by σ<sub>i</sub> so the four stations count equally, masked by μ<super>y</super> (missing targets), with larger weight on later horizons. "
               "AdamW, early stopping on validation MAE of the median at h = 6, chronological split (A1). Baselines for the committee: persistence and a linear ARX model on the same inputs.", SMALL))

    s.append(P("How each source influences each forecast", H3))
    I = [["Station", "Upstream stations / tributaries", "Rainfall", "Reservoir release", "Why this path"],
         ["S1", "none gauged upstream of the reservoir", "R1 through r<sub>S1</sub> (local runoff)", "Res→S1: <b>actual</b> past release read at learned lag; <b>announced</b> release for τ…τ+60 enters directly if travel time < 1 h", "The release is the main driver; without the schedule S1 is the hardest station because its water has not yet left the dam."],
         ["S2", "S1 (1 hop), T1 (1 hop, added)", "r<sub>S2</sub>; rain also drives T1's future flow", "2 hops (Res→S1→S2)", "Water now at S1 and T1 arrives at S2 within the hour (A6, A9)."],
         ["S3", "S2 (1 hop), U (latent, 1 hop), T1 (2 hops)", "drives U's inflow (A10); r<sub>S3</sub>", "3 hops", "S2's current level carries most of the signal; U adds an unseen inflow."],
         ["S4", "S3 (1 hop), U (2), S1–S2 (via S3), T1 (3)", "indirect, through U, T1 and upstream", "4 hops", "The longest travel time: today's S3 and S2 levels are the best predictor."]]
    s.append(table(I, [0.08, 0.19, 0.17, 0.25, 0.31]))
    s.append(P("Physical assumptions versus learned mappings", H3))
    PH = [["Physical assumption (built into the structure)", "Learned mapping (fitted from data)"],
          ["Directed downstream graph, no backwater (A7)", "Stage–discharge relation at each station (inside the MLP and the T blocks)"],
          ["Travel time = distance / celerity, prior in Eq. 4 and lag window Λ = 3 h (A6)", "The celerity c<sub>0</sub>, β, the spread of α (attenuation, dispersion) and each lag attention"],
          ["Summed contributions at confluences; monotone, non-negative gates (A8)", "Rainfall–runoff for T1, U and local runoff; gauge weights ω fine-tuned from the distance initialisation"],
          ["Forecast = last level + change (persistence skip, Eq. 6)", "The change itself; how reliable the announced schedule is (A12–A13); how much a gap in a series should be trusted (Eq. 1)"],
          ["Future rain and levels are unknown, so they are masked placeholders (F6)", "The unmonitored tributary U (latent, no direct supervision, A10)"]]
    s.append(table(PH, [0.5, 0.5]))
    s.append(P("<b>Why this design.</b> (1) A graph keeps physical locations: stations of unequal spacing and tributary junctions are explicit, instead of being forced into a regular grid or a flat vector. "
               "(2) Alternating T and S blocks (joint space-time paths) lets a change at S1 be read at S2 at the right lag, which neither a temporal-only nor a spatial-only stack can do. "
               "(3) Lag attention is a small, interpretable way to learn changing travel times that were not supplied. "
               "(4) Persistence plus residual starts every forecast from the best cheap baseline. (5) Quantile outputs give a usable uncertainty without a separate probabilistic model."))

    # ------------------------------------------------------------------ 5
    s.append(P("5. Operation and exceptional conditions", H2))
    s.append(P("The service loop at every tick τ: (1) freeze an <i>as-of snapshot</i> of the bitemporal archive (event time + arrival time) at τ + 2 min; "
               "(2) build X (Figure Q1.1, left half); (3) run the network; (4) store forecast, inputs’ ages and flags; (5) publish."))
    O = [["Case", "Handling", "Reason / what the model sees"],
         ["(i) Missing observations", "Value not available at τ: m = 0, δ keeps growing, value filled by Eq. 1 (decay from last value to station mean). Training randomly masks 5–20 % of values, bursts up to 2 h, and sometimes a whole station for up to 24 h. If S<sub>i</sub> is missing at τ the forecast for S<sub>i</sub> is still produced from neighbours and flagged; intervals widen as δ grows.",
          "The network was trained under the same gaps, so it learns to lean on neighbours and on the last valid value. Target gaps are masked in the loss."],
         ["(ii) Delayed observations", "Availability is defined by <i>arrival</i> time: a record that arrives after τ + 2 min is treated as missing at τ (so it appears with age δ > 0). When it arrives later it is written into the archive and used in all following windows. Late-arriving corrections do not overwrite past forecasts; they are logged. Training uses delays sampled from the empirical arrival log (A3).",
          "No leakage: the training windows are rebuilt as-of τ with simulated delays, so test-time staleness matches training."],
         ["(iii) Announced future releases", "For each future step τ+10 … τ+60 take the most recent announcement made ≤ τ whose effective time covers that step; write q<super>sch</super> on the Res time axis (zero-order hold) with f<super>sch</super> = 1. Steps with no announcement get the current actual release and f<super>sch</super> = 0. The model sees the schedule only on positions after τ; the S block carries it downstream with the learned lag.",
          "Uses only information that exists at τ; the flag lets the model separate ‘planned’ from ‘assumed unchanged’. Announcements made after τ are ignored by construction."],
         ["(iv) Scheduled versus actual release", "Past steps use the metered <i>actual</i> release; future steps use the schedule. The recent deviation e(t) = q<super>act</super>(t) − q<super>sch</super>(t) over the last 2 h is a feature on Res. Training pairs actual history with the schedule as announced at the time, so real mismatches (gate delay, emergency spill) are in the data; extra augmentation shifts schedule steps by 0–30 min and scales them by ±10 %. If |e| is large the 90 % quantile is automatically wider.",
          "The model learns how much to trust a schedule from how well past schedules were followed (A13). An unannounced change is a limitation (§6)."]]
    s.append(table(O, [0.14, 0.52, 0.34]))
    s.append(P("<b>Feasibility every 10 minutes.</b> The network has ≈ 0.25 M parameters. One forecast reads N·T = 7·78 = 546 node-time positions: the temporal convolutions cost about 7·78·(3·64·128)·2 ≈ 27 MFLOP per block, "
               "and the lag attention (6 edges × 78 steps × 19 lags × 32 hidden units) is below 1 MFLOP per block once the projections are shared, so the whole forward pass is a few hundred MFLOP (my estimate; no benchmark was run) — "
               "well under a second on one CPU core, against a 600 s budget. Data access (a few kilobytes per tick) and publishing dominate. "
               "Training (≈ 263 000 windows × ~1 GFLOP including backward ≈ 3·10<super>14</super> FLOP per epoch) is offline, once per year or after a major flood. "
               "Fallbacks: if the model or a feed fails, publish the last-level persistence forecast with a ‘degraded’ flag; a health check compares each forecast with the level observed one hour later."))

    # ------------------------------------------------------------------ 6
    s.append(P("6. Uncertainty, limitations and additional information", H2))
    U = [["ID", "Uncertainty / limitation", "Impact on the forecast", "Response, or what remains unresolved"],
         ["U1", "Unmonitored tributary between S2 and S3", "Unseen inflow changes S3 and S4 after rain; largest during storm onset", "Latent node U driven by rain and S2→S3 mismatch; q0.9 widens at S3/S4. <b>Unresolved:</b> its true flow cannot be checked without a gauge."],
         ["U2", "Travel times change with flow, season and gate operations", "A wrong lag shifts the arrival of a wave and gives large errors on rising limbs", "State-dependent lag attention (Eq. 4–5); retraining each year. <b>Remains:</b> extreme floods beyond the training range."],
         ["U3", "No future rainfall (F6)", "Rain falling in the next hour changes local and tributary levels that the model cannot see", "Only past rain is used; the 90 % quantile covers part of it. <b>Unresolved</b> unless a rain nowcast is supplied (R2)."],
         ["U4", "Sparse rainfall sampling (two gauges)", "Storm cells missed by R1/R2 make the runoff estimate wrong", "Distance weights and uncertainty widening when T1/S rise without rain. <b>Unresolved</b>: needs more gauges (R4)."],
         ["U5", "Missing, late or faulty observations", "Skill degrades with gap length δ", "Masks, decay imputation, delay-aware training, flags (§5). Outages longer than ~24 h degrade to near-persistence."],
         ["U6", "Unannounced or changed releases", "S1 error grows when the dam acts differently from the schedule", "Deviation feature and a wider q0.9. <b>Unresolved</b> for emergency spills without notice."],
         ["U7", "Rare, extreme floods and short record", "The network may under-predict peaks outside the training range", "Pinball loss with horizon weighting, evaluation on flood events separately, physical priors, persistence fallback. <b>Remains:</b> true extremes cannot be validated."],
         ["U8", "Non-stationary channel (silting, vegetation, gauge drift)", "Systematic bias and shifting stage–discharge", "Season encoding, yearly retraining, drift monitoring against rolling error. <b>Remains</b> until a rating-curve update is supplied (R1)."],
         ["U9", "Backwater, abstractions, weirs not modelled (A7)", "Local level offsets", "Ignored (assumption A7). Documented; can be added as extra edges or inputs if information appears (R5)."],
         ["U10", "Calibration of quantiles", "Intervals may be too narrow or too wide", "Check coverage on validation per station and per flow class; recalibrate with conformal offsets. <b>Remains:</b> coverage in rare events is untested."]]
    s.append(table(U, [0.05, 0.22, 0.27, 0.46]))
    s.append(P("Additional information requests", H3))
    Rq = [["ID", "Request", "How it would help", "System usable without it?"],
          ["R1", "Stage–discharge (rating) curves and cross-sections for S1–S4, T1", "Converts levels to discharge so mass balance at confluences can be checked and applied; reduces U8", "Yes. The curves are learned from data."],
          ["R2", "Rainfall nowcast or radar for the next 1–3 h", "Gives the model the future rain it lacks (U3); biggest gain at storm onset", "Yes. The mask handles absence and the 90 % quantile covers part of the risk."],
          ["R3", "A gauge or even a temporary stage logger on the unmonitored tributary", "Turns U from latent into observed (U1); direct supervision of its rain response", "Yes. It remains a latent node."],
          ["R4", "More rain gauges in the catchment", "Better areal rainfall and fewer missed cells (U4)", "Yes. The distance weights adapt to what exists."],
          ["R5", "Reservoir gate positions, reservoir level, operating rules, any abstractions or weirs", "Explains release deviations (U6) and local offsets (U9); better release model for the Res node", "Yes. The system relies on actual and scheduled flow."],
          ["R6", "History of past flood events with at least 5–10 years of records, and sensor QC flags and measurement timestamps", "More extreme events for training and honest evaluation (U7); cleaner masks (U5)", "Yes. With fewer years use a smaller model and rely on priors."],
          ["R7", "Reporting-delay log (arrival times) and release announcement log", "Exact as-of reconstruction for training (§5 ii, iii)", "Yes, with simulated delays as a substitute (A3, A12)."],
          ["R8", "Flood-stage thresholds at S1–S4", "Evaluate alert skill (hit rate, false alarms) and tune the quantile that triggers warnings", "Yes. Forecasts remain valid; alert tuning would be generic."]]
    s.append(table(Rq, [0.05, 0.27, 0.40, 0.28]))
    return s
