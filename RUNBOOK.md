# Runbook: from zero to the submitted report

Operating notes for reproducing every number in the report. Written for Windows
(Anaconda Prompt); the commands also work in bash with the obvious path changes.

---

## 0. Environment

```bat
conda activate myenv
python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"
```

Activate the environment first. Running the interpreter directly out of `envs\myenv\python.exe`
without activating it leaves `Library\bin` off `PATH`, and matplotlib then fails when it writes a
figure.

If you build a fresh environment instead:

```bat
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 1. Data

```bat
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git data/PlantDoc-Dataset

git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git data/PlantVillage-Dataset
cd data\PlantVillage-Dataset
git sparse-checkout set raw/color/Tomato___healthy raw/color/Tomato___Early_blight raw/color/Tomato___Late_blight
cd ..\..
```

Only the three tomato folders of the colour variant are needed; the full repository also ships
grayscale and segmented copies of the same leaves.

Check that both datasets are visible to the code:

```bat
python data_stats.py --root plantvillage=data/PlantVillage-Dataset/raw/color --root plantdoc=data/PlantDoc-Dataset --list-only
```

Expected: three PlantVillage classes with four-digit counts, three matched PlantDoc classes, and a
list of unmatched folders. The unmatched folders are the crops and tomato diseases that this
project does not use; they are reported, not silently dropped.

## 2. Statistics and figures

```bat
python data_stats.py --root plantvillage=data/PlantVillage-Dataset/raw/color --root plantdoc=data/PlantDoc-Dataset --out-csv stats/class_counts.csv --out-md stats/class_counts.md

python make_figures.py --root plantvillage=data/PlantVillage-Dataset/raw/color --root plantdoc=data/PlantDoc-Dataset --out-dir figures --seed 0

python make_fig1.py --out figures/fig1_samples.png --seed 0
```

## 3. Training

One command per configuration. Remove `--limit-per-class` for the real runs.

```bat
set PV=data/PlantVillage-Dataset/raw/color

python train_baseline.py --data-root %PV% --model basiccnn --epochs 20 --batch-size 32 --seed 42 --out-dir runs/E1_basiccnn
python train_baseline.py --data-root %PV% --model resnet18 --epochs 20 --batch-size 32 --seed 42 --out-dir runs/E2_resnet18_scratch
python train_baseline.py --data-root %PV% --model resnet18 --pretrained --epochs 20 --batch-size 32 --seed 42 --out-dir runs/E3_resnet18_pre
python train_baseline.py --data-root %PV% --model resnet18 --pretrained --augment --epochs 20 --batch-size 32 --seed 42 --out-dir runs/E4_resnet18_pre_aug
```

A quick smoke test (a few steps, truncated data) checks that the pipeline runs at all:

```bat
python train_baseline.py --data-root %PV% --model basiccnn --epochs 1 --limit-per-class 50 --max-steps 5 --out-dir runs/smoke_test
```

Numbers from a truncated run mean nothing; the script prints a warning when a run is truncated.

## 4. Cross-domain evaluation

No retraining: load a checkpoint and score PlantDoc.

```bat
python train_baseline.py --model resnet18 --eval-only --ckpt runs/E3_resnet18_pre/best.pt --plantdoc-root data/PlantDoc-Dataset --out-dir runs/E3_resnet18_pre
```

Repeat for E1, E2 and E4 by changing `--ckpt` and `--out-dir`. In-domain evaluation uses the same
flag with `--data-root`; it rebuilds the training split from the stored seed and scores the test
part only.

## 5. Figures

```bat
python plot_results.py --runs runs --out-dir figures --errors-run E3_resnet18_pre --errors-tag plantdoc
```

This writes the training curves, confusion matrices, the comparison chart, the error panel and
`figures/index.md` with captions and the values it read from the metrics files.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | batch too large for the GPU | `--batch-size 16`, or `--image-size 128` |
| `torch.cuda.is_available()` is `False` | CPU-only wheel installed | install the CUDA wheel, or accept CPU training |
| `No folder with images was found` | wrong `--data-root` | point it at the folder that contains the class folders |
| `cannot identify image file` | corrupt image in the dataset | the loader skips it; delete the file if it appears often |
| matplotlib exits with `-1066598273` | environment not activated | `conda activate myenv` first |
| `BrokenPipeError` in the first batch | multiprocessing on Windows | keep `--num-workers 0` (the default) |
## 7. Demo app

```bat
python app.py
```

Opens `http://127.0.0.1:7860`. It loads `runs/E3_resnet18_pre/best.pt` on the GPU when one is
available and returns the three class probabilities for one uploaded photo. The four thumbnails
under the upload box are PlantDoc photos that were never used for training.

Two environment notes, both already pinned in `requirements.txt`:

- `gradio==4.44.1` needs `huggingface_hub<1.0`, because it imports `HfFolder`, which 1.x removed.
- `gradio_client 1.3.0` crashes on JSON schemas produced by `pydantic>=2.11`, hence `pydantic==2.9.2`.

Close the window or press `Ctrl+C` to stop the server.