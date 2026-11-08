# 生成的图与图注

> 图注可直接粘进报告；`Fig. N` 的编号按你们 Word 里的顺序改。

## `fig3_curves_E1_basiccnn.png`

Fig. 3 Training curves. Left: training and validation cross-entropy loss per epoch. Right: validation accuracy and validation macro-F1 per epoch. The checkpoint with the highest validation macro-F1 is kept for evaluation.

## `fig4_cm_E1_basiccnn_plantdoc.png`

Fig. 4 Confusion matrices on the three-class tomato task; cell values are image counts with the row-normalised share in brackets.

## `fig5_compare.png`

Fig. 5 In-domain (PlantVillage test) versus cross-domain (PlantDoc) accuracy and macro-F1 for every configuration. All values are read from the corresponding metrics_*.json files.

## `fig6_errors_E3_resnet18_pre_plantdoc.png`

Fig. 6 Misclassified test images. Each panel shows the ground-truth label, the predicted label, and the predicted probability, illustrating the failure modes discussed in the error analysis.


---

## 本次读到的数值（核对用）

- **E1_basiccnn**: PV-test acc=0.954074, macro-F1=0.949836; PlantDoc acc=0.458015, macro-F1=0.340919; Δacc=+0.4961
- **E2_resnet18_scratch**: PV-test acc=0.982222, macro-F1=0.98052; PlantDoc acc=0.473282, macro-F1=0.342868; Δacc=+0.5089
- **E3_resnet18_pre**: PV-test acc=0.98963, macro-F1=0.988138; PlantDoc acc=0.618321, macro-F1=0.528146; Δacc=+0.3713
- **E4_resnet18_pre_aug**: PV-test acc=0.992593, macro-F1=0.991554; PlantDoc acc=0.583969, macro-F1=0.451165; Δacc=+0.4086
