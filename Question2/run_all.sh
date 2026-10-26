#!/usr/bin/env bash
# Question 2 pipeline (handout protocol: 30 epochs, seeds 0 and 1). Run from the Question2/ folder, one stage at a time:
#   bash run_all.sh check      GPU + dataset + Table 1 checks
#   bash run_all.sh A          train model A (2 seeds) -> Table 2, Figure 2 -> Tables 3-4, Figures 3-5
#   bash run_all.sh selectD    validation-only screening of model D -> results/d_choice.json
#   bash run_all.sh BCD        train B, C, D (2 seeds each)
#   bash run_all.sh eval       Tables 5-6 metrics, extra analysis, inference timing, Figure 6, run verification
# Every stage is safe to repeat after a disconnect: finished runs are skipped (train.py --skip_done); an unfinished run restarts at epoch 1.
# IMPORTANT (handout): the Q2.5 / Q2.11 predictions in report/q2_predictions.md must already be committed before stage "A"'s
# baseline tests / stage "BCD" are run. They are (see git history); do not edit them.
set -eo pipefail
EPOCHS=${EPOCHS:-30}
export EXPECTED_EPOCHS=$EPOCHS
SCREEN_STEPS=${SCREEN_STEPS:-1500}
SPECS="A:0 A:1 B:0 B:1 C:0 C:1 D:0 D:1 B:0:sym B:1:sym D:0:sym D:1:sym"

train () { for s in 0 1; do python train.py --model "$1" --seed $s --epochs $EPOCHS --skip_done 2>&1 | tee -a runs/$1_s$s.log; done; }

case "$1" in
  check)
    python -c "import cv2, matplotlib, PIL, numpy, torchvision; print('cv2',cv2.__version__,'| torchvision',torchvision.__version__,'| numpy',numpy.__version__)"
    python -c "import torch,mm; print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| device',mm.DEVICE); mm.require_gpu(); print(torch.cuda.get_device_name(0))"
    python -c "import mm; v=mm.load_videos('data'); print('dataset', tuple(v.shape), v.dtype)"
    python checks.py ;;
  A)
    mkdir -p runs results gifs ../report
    train A
    python analysis.py table2        # Table 2, Figure 2
    python analysis.py baseline      # Tables 3-4, Figures 3-5 (+ Fig. 3 GIFs)
    ;;
  selectD)
    rm -f results/select_d_*.json results/d_choice.json
    for c in p8_k3 p4_k5_h32; do python select_d.py $c $SCREEN_STEPS; done
    python choose_d.py ;;
  BCD)
    test -f results/d_choice.json || { echo "run: bash run_all.sh selectD first"; exit 1; }
    for m in B C D; do train $m; done ;;
  eval)
    python verify_runs.py
    python analysis.py models $SPECS
    python extra_analysis.py A:0 A:1 C:0 C:1 D:0 D:1
    python analysis.py timing $SPECS     # run alone on the GPU
    python analysis.py final D           # Figure 6 (+ Fig. 6a GIFs)
    ;;
  *) echo "usage: bash run_all.sh {check|A|selectD|BCD|eval}"; exit 1 ;;
esac
