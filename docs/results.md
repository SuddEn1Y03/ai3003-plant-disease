# Results summary

All numbers below were produced by `train_baseline.py` and `plot_results.py` on the
three-class tomato subset described in the report. Every value can be re-read from
`runs/<run>/metrics_test.json` (PlantVillage test split, n = 675) and
`runs/<run>/metrics_plantdoc.json` (PlantDoc, n = 262; evaluation only, never used for training).

## Main results

| ID | Model | Init. | Aug. | PV Acc | PV macro-F1 | PD Acc | PD macro-F1 | Drop |
|---|---|---|---|---|---|---|---|---|
| E1 | Basic CNN | random | no | 0.9541 | 0.9498 | 0.4580 | 0.3409 | 0.4961 |
| E2 | ResNet18 | random | no | 0.9822 | 0.9805 | 0.4733 | 0.3429 | 0.5089 |
| E3 | ResNet18 | ImageNet | no | 0.9896 | 0.9881 | **0.6183** | **0.5281** | 0.3713 |
| E4 | ResNet18 | ImageNet | yes | **0.9926** | **0.9916** | 0.5840 | 0.4512 | 0.4086 |
| E5 | = E4 checkpoint | ImageNet | yes | (same as E4) | (same as E4) | 0.5840 | 0.4512 | 0.4086 |

95 % Wilson confidence intervals:

| run | PlantVillage test | PlantDoc |
|---|---|---|
| E1 | [0.936, 0.967] | [0.399, 0.519] |
| E2 | [0.969, 0.990] | [0.414, 0.534] |
| E3 | [0.979, 0.995] | [0.558, 0.675] |
| E4 | [0.983, 0.997] | [0.523, 0.642] |

## Per-class recall on PlantDoc

| Class | support | E1 | E2 | E3 | E4 |
|---|---|---|---|---|---|
| healthy | 63 | 0.032 | 0.032 | 0.127 | 0.016 |
| early blight | 88 | 0.330 | 0.284 | 0.773 | 0.750 |
| late blight | 111 | 0.802 | 0.874 | 0.775 | 0.775 |

## Training cost

| run | epochs run | best epoch | best val macro-F1 | total time |
|---|---|---|---|---|
| E1 Basic CNN | 19 (early stop) | 14 | 0.9652 | 228 s |
| E2 ResNet18 scratch | 20 | 19 | 0.9869 | 227 s |
| E3 ResNet18 ImageNet | 11 (early stop) | 6 | 0.9966 | 118 s |
| E4 ResNet18 + aug | 17 (early stop) | 12 | 0.9966 | 324 s |

Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, batch size 32, 224 x 224, Adam, cosine
annealing, early stopping with patience 5.

## Data

| Class | PlantVillage (colour) | PV train / val / test | PlantDoc (mapped) | PD train / test |
|---|---|---|---|---|
| healthy | 1,591 | 1,114 / 239 / 238 | 63 | 55 / 8 |
| early blight | 1,000 | 700 / 150 / 150 | 88 | 79 / 9 |
| late blight | 1,909 | 1,336 / 286 / 287 | 111 | 101 / 10 |
| **Total** | **4,500** | 3,150 / 675 / 675 | **262** | 235 / 27 |

## How to reproduce

```bash
python data_stats.py --root plantvillage=data/PlantVillage-Dataset/raw/color \
                     --root plantdoc=data/PlantDoc-Dataset \
                     --out-csv stats/class_counts.csv --out-md stats/class_counts.md
python make_figures.py --root plantvillage=data/PlantVillage-Dataset/raw/color \
                       --root plantdoc=data/PlantDoc-Dataset --out-dir figures
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color \
                         --model resnet18 --pretrained --epochs 20 --seed 42 \
                         --out-dir runs/E3_resnet18_pre
python train_baseline.py --model resnet18 --pretrained --eval-only \
                         --ckpt runs/E3_resnet18_pre/best.pt \
                         --plantdoc-root data/PlantDoc-Dataset \
                         --out-dir runs/E3_resnet18_pre
python plot_results.py --runs runs --out-dir figures
```

> Note: model weights (`*.pt`) are not committed — they are excluded by `.gitignore`.
> Re-running the training commands above regenerates them.
