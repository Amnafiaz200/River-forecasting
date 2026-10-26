# Colab cells (T4 GPU) - run in this order

Runtime > Change runtime type > Hardware accelerator: **T4 GPU** > Save.
Drive folder used for everything that must survive a disconnect: `MyDrive/AI651_A2/` (contains `Question2_colab.zip`; the folders
`runs/ results/ gifs/` are created automatically and hold checkpoints, logs and results).

## Cell 1 (Python): GPU + Drive
```python
import torch, subprocess
print(torch.__version__, "| cuda:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 2 (bash): unpack code, link Drive folders, check
```bash
%%bash
set -e
D=/content/drive/MyDrive/AI651_A2
mkdir -p $D/runs $D/results $D/gifs /content/proj
cd /content/proj && unzip -oq $D/Question2_colab.zip
cd Question2
ln -sfn $D/runs runs; ln -sfn $D/results results; ln -sfn $D/gifs gifs
mkdir -p ../report
bash run_all.sh check
```

## Cell 3 (bash): model A (2 seeds), Table 2, Figure 2, Tables 3-4, Figures 3-5   (long)
```bash
%%bash
cd /content/proj/Question2
nohup bash run_all.sh A > runs/stage_A.out 2>&1 &
echo started
```
Watch progress (run as often as you like):
```bash
!tail -n 3 /content/proj/Question2/runs/A_s0.log /content/proj/Question2/runs/A_s1.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```
Finished when `runs/stage_A.out` ends without an error and `results/fig5_video.json` exists.

## Cell 4 (bash): model D design, validation split only
```bash
%%bash
cd /content/proj/Question2 && bash run_all.sh selectD 2>&1 | tail -n 5
```

## Cell 5 (bash): models B, C, D (2 seeds each)   (long)
```bash
%%bash
cd /content/proj/Question2
nohup bash run_all.sh BCD > runs/stage_BCD.out 2>&1 &
echo started
```
```bash
!tail -n 2 /content/proj/Question2/runs/[BCD]_s[01].log
```

## Cell 6 (bash): Tables 5-6, extra analysis, inference timing, Figure 6, verification
```bash
%%bash
cd /content/proj/Question2 && bash run_all.sh eval 2>&1 | tail -n 30
```
Must end with `ALL RUNS OK` near the top of the output of `verify_runs.py`.

## Cell 7: download everything needed to rebuild the report
```bash
%%bash
cd /content/proj && rm -f /content/AI651_results.zip
zip -rq /content/AI651_results.zip Question2/runs Question2/results Question2/gifs report
ls -la /content/AI651_results.zip
cp /content/AI651_results.zip /content/drive/MyDrive/AI651_A2/
```
```python
from google.colab import files
files.download('/content/AI651_results.zip')
```

## After a disconnect
Run Cell 1 and Cell 2 again (the dataset is downloaded again, ~1-2 min), then re-run the stage that was running. Finished runs are skipped;
a run that was interrupted restarts from epoch 1 (required by the protocol).
