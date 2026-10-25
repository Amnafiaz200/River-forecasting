# AI651 Assignment 2

* `report/Assignment2_Report.pdf` — the single PDF report to upload to LMS (Question 1 and Question 2, Tables 1–6, Figures 1–6, answers Q2.1–Q2.17, AI-use disclosure).
* `Question1/` — the supplied companion material (optional, ungraded) and `make_diagram.py`, which draws the Q1 architecture diagrams.
* `Question2/` — all code for the Moving MNIST question.
* `report/` — PDF builder (`build_report.py`, `q1_text.py`, `q2_text.py`, `ai_disclosure.py`) and the figures.

## Question 2: how to reproduce

Requirements: Python 3.10+, `torch`, `torchvision`, `opencv-python`, `numpy`, `matplotlib`, `pillow`, `reportlab`, `pymupdf` (only to preview the PDF).
Run everything from `Question2/` (the dataset is downloaded to `Question2/data/`, ~819 MB, and is not committed).

```
cd Question2
python checks.py                      # Table 1  (results/table1.json)
python flow_fig1.py                   # Figure 1 (optical flow, 950 frame pairs)
python train.py --model A --seed 0 --epochs 30     # and --seed 1
python analysis.py table2             # Table 2, Figure 2
python analysis.py baseline           # Tables 3-4, Figures 3-5 (model A, Blank, Copy-last)
python select_d.py p8_k3 250          # model D screening on the VALIDATION split only
python select_d.py p4_k5_h32 250
python train.py --model B --seed 0 --epochs 30     # B, C, D, seeds 0 and 1
python analysis.py models A:0 A:1 B:0 B:1 C:0 C:1 D:0 D:1 B:0:sym B:1:sym D:0:sym D:1:sym
python analysis.py final D            # Figure 6
cd ../report && python build_report.py    # writes Assignment2_Report.pdf from results/*.json
```

`Question2/run_all.sh` runs the same sequence (`EPOCHS=30` by default).

**Compute note.** The handout protocol is 30 epochs per model. The results in the submitted report were produced on a CPU laptop
(AMD Ryzen 5 5500U, no GPU) with a reduced number of epochs; the exact number is stated in the report and in
`Question2/runs/*.json`. On a GPU use `EPOCHS=30 bash run_all.sh`.

## Files

| File | Purpose |
|---|---|
| `Question2/mm.py` | data split, mirror/rotation, speed windows, ConvLSTM model, symmetrised wrapper, loss |
| `Question2/checks.py` | Table 1 checks |
| `Question2/flow_fig1.py` | Figure 1 (OpenCV Farneback, exact handout settings) |
| `Question2/train.py` | training protocol for models A, B, C, D, DB |
| `Question2/select_d.py` | validation-only screening of model D candidates |
| `Question2/analysis.py` | metrics, tables, figures and GIFs |
| `Question2/gifs/` | animated GIFs for Figures 3 and 6a |
| `Question2/results/` | all numbers behind the tables (JSON) |
| `report/q2_predictions.md` | the answers to Q2.5 and Q2.11, committed before the experiments they refer to |
