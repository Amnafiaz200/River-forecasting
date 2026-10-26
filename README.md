# AI651 Assignment 2 - Code and Results

This repository contains the source code, final trained checkpoints, experiment results and verification evidence for AI651 Assignment 2.

The final PDF report is submitted separately through LMS. Question 1 is a written proposal and does not require an implementation.

## Contents

- Question2/*.py: training, evaluation and verification code
- Question2/run_all.sh: complete GPU experiment pipeline
- Question2/runs/: 8 final 30-epoch checkpoints, JSON metadata and training logs
- Question2/results/: evaluation results, tables, model D selection and checkpoint manifest
- Question2/gifs/: 8 required Figure 3 and Figure 6(a) GIFs
- Question2/q2_predictions.md: original Q2.5 and Q2.11 prediction text
- run.ipynb: Google Colab training notebook
- requirements.txt: Python dependencies

## Verify trained models without retraining

pip install -r requirements.txt
cd Question2
python verify_checkpoints.py --sym

The Moving MNIST dataset is downloaded automatically when needed.

## Full GPU reproduction

Run from Question2:

bash run_all.sh check
bash run_all.sh A
bash run_all.sh selectD
bash run_all.sh BCD
bash run_all.sh eval

The full protocol uses 30 epochs, seeds 0 and 1, Adam with learning rate 0.001 and batch size 32.

## Original prediction history

The prediction text in Question2/q2_predictions.md is copied unchanged from the original file.

The original prediction commits are preserved:
- Q2.5: b7b98ea
- Q2.11: d919944

Inspect the original committed versions with:

git show b7b98ea:report/q2_predictions.md
git show d919944:report/q2_predictions.md

The full report is available through the LMS submission, not the current repository tree.
