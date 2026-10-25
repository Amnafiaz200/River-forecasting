#!/usr/bin/env bash
# Reproduce Question 2. Run from this folder (Question2/).  EPOCHS=30 is the handout protocol; the submitted
# results were produced with a reduced EPOCHS on a CPU laptop (see the report) -- set EPOCHS=30 on a GPU machine.
set -e
EPOCHS=${EPOCHS:-30}
python checks.py                      # Table 1
python flow_fig1.py                   # Figure 1
for s in 0 1; do python train.py --model A --seed $s --epochs $EPOCHS & done; wait
python analysis.py table2             # Table 2, Figure 2
# --- write + commit Q2.5 (report/q2_predictions.md) BEFORE the next line ---
python analysis.py baseline           # Tables 3-4, Figures 3-5 (model A)
# --- write + commit Q2.11 (report/q2_predictions.md) BEFORE the next lines ---
python select_d.py p8_k3 250; python select_d.py p4_k5_h32 250    # validation-only screening of model D (edit D_CONFIG in mm.py)
for m in B C D; do for s in 0 1; do python train.py --model $m --seed $s --epochs $EPOCHS & done; done; wait
python analysis.py models A:0 A:1 B:0 B:1 C:0 C:1 D:0 D:1 B:0:sym B:1:sym D:0:sym D:1:sym   # Tables 5-6
python analysis.py final D            # Figure 6
cd ../report && python build_report.py   # PDF
