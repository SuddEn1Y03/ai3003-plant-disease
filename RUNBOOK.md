# 从零到提交：番茄叶病分类项目完整操作手册

> 这份手册假设你只有：一台电脑、一个 GitHub 账号、课程要求（Abstract / Introduction / Related Works /
> Methodology / Progress / Next Steps / References / Contribution）。
> 每一步都写了**做什么 → 跑什么命令 → 应该看到什么 → 产出进报告的哪里**。
> 数字凡是不知道的，一律 `[待填]`，不要猜。

---

## 0. 总览：九个阶段，产出什么

| 阶段 | 大概耗时 | 产出 | 进报告的哪里 |
|---|---|---|---|
| A 环境 | 30 分钟（看网速） | 能 `import torch` 的虚拟环境 | 3.6.3 环境那一行 |
| B 数据 | 0.5–2 小时（看网速） | `data/` 下两个数据集 + commit 记录 | 3.2、4.1.2 |
| C 统计与基础图 | 20 分钟 | `stats/class_counts.md`、`figures/sample_images.png`、`figures/class_distribution.png` | **Table 1**、**Fig. 1**、**Fig. 2** |
| D 冒烟测试 | 15 分钟 | 能跑通的 1 个 batch + 参数量 | 4.1.4、**Table 4/5** 参数量 |
| E 正式实验 | 数小时（看硬件） | `runs/E1..E4/` 全套文件 | **Table 6**（实参）、**Table 9**（结果）、**Fig. 3**、**Fig. 4** |
| F 跨域评估 | 5 分钟 | `runs/E5_plantdoc_eval/metrics_plantdoc.json` | **Table 9** 的 PlantDoc 两列 |
| G 出图 | 10 分钟 | `fig5_compare.png`、`fig6_errors_*.png` | **Fig. 5**、**Fig. 6**、错例分析段落 |
| H 填报告 | 2–4 小时 | 报告 PDF | 全文 |
| I 打包提交 | 30 分钟 | ZIP | iSpace |

时间不够时的最小路径见第 10 节。

---

## 阶段 A：环境（一次性，做完不用再管）

> **命令在哪跑？** 用 **Anaconda Prompt**（开始菜单搜 "Anaconda Prompt"），先 `conda activate myenv`，
> 再 `cd /d` 到项目目录。本手册的多行命令用的是 PowerShell 的**反引号 ` 续行**，Anaconda Prompt 是 cmd，
> **反引号不能用**：换成 `^`，或者删掉换行写成一整行。第 12 节有一份 cmd 版速查。

### A0. 用 Anaconda 的话先看这里（本机已验证）

本机已有装好 PyTorch 的 conda 环境 **`myenv`**：Python 3.9.23、torch 2.5.1+cu121、torchvision 0.20.1、
`cuda.is_available() == True`、numpy 2.0.2、Pillow 11.3.0、matplotlib 3.9.4 —— **不需要再装任何东西**。

```bat
conda activate myenv
cd /d "C:\Users\75342\Desktop\NNDL project\tomato-leaf-transfer"
python -c "import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.cuda.is_available())"
```

三条注意事项：

1. **不要建 `.venv`**（下面的 A2 跳过），不要在 conda 环境里再嵌一层虚拟环境。
2. **不要 `pip install -r requirements.txt`**：它可能用 PyPI 的 torch 覆盖掉现有的 2.5.1+cu121，
   换成 CPU 版甚至装坏。本机依赖已经齐全，什么都不用装。
3. **必须在激活后的环境里运行**。直接从普通 PowerShell 调用 `...\envs\myenv\python.exe` 时，
   该环境的 `Library\bin` 不在 PATH 上，matplotlib 画图会崩（退出码 `-1066598273`），
   看着像代码 bug，其实是没激活环境。

### A1. 确认 Python

```powershell
python --version      # 需要 3.10 或更高
```

### A2. 建虚拟环境（推荐，避免污染系统 Python）

```powershell
cd "C:\Users\75342\Desktop\NNDL project\tomato-leaf-transfer"
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> PowerShell 报"禁止运行脚本"时执行一次：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### A3. 装依赖（二选一）

```powershell
# 有 NVIDIA 显卡（先 nvidia-smi 看 CUDA 版本，12.1 就装 cu121）
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# 只有 CPU（更稳，速度慢但一定能装上）
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

验证：

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

- 看到版本号 + `True/False` 就成功了。
- `False` 不一定是错（说明用的是 CPU 版），**但报告 3.6.3 的 Hardware 必须如实写成 "CPU only"**。

### A4. 确认三个脚本能启动

```powershell
python data_stats.py --help
python make_figures.py --help
python train_baseline.py --help
```

> 装 PyTorch 卡住时的退路：`data_stats.py`、`make_figures.py`、`train_baseline.py --dry-run`
> 都**不需要** PyTorch，可以先把阶段 B–C 做完再回来装。

---

## 阶段 B：下载数据

### B1. PlantDoc（小，直接克隆）

```powershell
cd "C:\Users\75342\Desktop\NNDL project\tomato-leaf-transfer"
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git data/PlantDoc-Dataset
```

### B2. PlantVillage（大，只取三个番茄目录）

整个仓库带 `color/`、`grayscale/`、`segmented/` 三份拷贝，全量克隆等于下三倍的数据。用 sparse checkout 只拿要用的：

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git data/PlantVillage-Dataset
cd data/PlantVillage-Dataset
git sparse-checkout set raw/color/Tomato___healthy raw/color/Tomato___Early_blight raw/color/Tomato___Late_blight
cd ..\..
```

没有 git 的话：浏览器打开仓库 → Code → Download ZIP → 解压到 `data/`，再把 `raw/color` 下那三个目录留下即可（其余删掉省空间）。

### B3. 记录版本（报告 4.1.2 要写）

```powershell
git -C data/PlantVillage-Dataset rev-parse HEAD
git -C data/PlantDoc-Dataset rev-parse HEAD
```

把两个 40 位的哈希抄到记事本，报告里写 "commit `xxxxxxx`（2026-__-__ 下载）"。

### B4. 核对类名（关键一步，别跳）

```powershell
python data_stats.py --root plantvillage=data/PlantVillage-Dataset/raw/color `
                     --root plantdoc=data/PlantDoc-Dataset --list-only
```

**应该看到什么**（数字以你机器为准）：

```
=== plantvillage : ...\raw\color ===
  [all      ] Tomato___Early_blight        n=____  ->  early_blight
  [all      ] Tomato___Late_blight         n=____  ->  late_blight
  [all      ] Tomato___healthy             n=____  ->  healthy

=== plantdoc : ...\PlantDoc-Dataset ===
  [test     ] Tomato Early blight leaf     n=____  ->  early_blight
  ...
  [train    ] Tomato leaf                  n=____  ->  healthy
  [train    ] Tomato leaf late blight      n=____  ->  late_blight
  [train    ] Tomato leaf mosaic virus     n=____  ->  -- 未匹配 --
  ...
```

**判读规则**：

- 三个目标类都带 `-> healthy / early_blight / late_blight` → 通过，进入阶段 C。
- PlantDoc 另外 6 个番茄类显示 `-- 未匹配 --` → **正常**，它们本来就该被丢掉。
- 出现你没见过的目录名、或目标类显示未匹配 → 打开目录看一眼真实名字，改 `data_stats.py` 里的
  `DEFAULT_MAP`，或者写一个 `--mapping my.json`。

---

## 阶段 C：样本统计 + 两张基础图

### C1. 统计（生成 Table 1 的数字）

```powershell
python data_stats.py --root plantvillage=data/PlantVillage-Dataset/raw/color `
                     --root plantdoc=data/PlantDoc-Dataset `
                     --out-csv stats/class_counts.csv --out-md stats/class_counts.md
```

打开 `stats/class_counts.md`，里面有三张表：plantvillage 明细、plantdoc 明细、统一标签汇总。

### C2. 出图（Fig. 1 / Fig. 2）

```powershell
python make_figures.py --root plantvillage=data/PlantVillage-Dataset/raw/color `
                       --root plantdoc=data/PlantDoc-Dataset `
                       --out-dir figures --seed 0
```

产出：`figures/sample_images.png`（3 类 × 2 张）、`figures/class_distribution.png`（三类柱状图）、
`figures/figure_sources.txt`（抽到了哪几张图，写图注时用）。

### C3. 把数字填进 Table 1

Table 1 在补丁包第 2.5 节。填法：

| 列 | 从哪来 |
|---|---|
| PlantVillage (colour) 三类张数 | `class_counts.md` 的 plantvillage 明细 |
| PlantDoc (mapped) 三类张数 | `class_counts.md` 的 plantdoc 明细（train+test 合并） |
| PV train / val / test 三列 | **先空着**，跑完 E1 后从 `runs/E1_*/log.csv` 首行的 `n_train`、`n_val` 和 `metrics_test.json` 的 `n` 取 |
| PD train / test 两列 | `class_counts.md` 里按 split 分的那两行 |

> 这一步做完，你就已经有一张别人伪造不出来的真实数据表了。**先做这个，再谈装环境。**

---

## 阶段 D：冒烟测试（确认代码在这个数据集上真的能跑）

### D1. 数据管线自检（不需要 PyTorch）

```powershell
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --dry-run
```

看三件事：三类张数、70/15/15 划分后的数量、抽检图片是否 `OK`（有 `BAD` 就先删掉那些文件）。

### D2. 记下参数量（Table 4 / Table 5 要填）

```powershell
python train_baseline.py --model basiccnn  --print-model
python train_baseline.py --model resnet18 --pretrained --print-model
```

把两次输出的 `TOTAL` 抄进 Table 4（BasicCNN）和 Table 5（ResNet18）。第一次跑 ResNet18 会联网下载权重（约几十 MB）。

### D3. 跑 5 个 step

```powershell
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color `
    --model basiccnn --epochs 1 --limit-per-class 50 --batch-size 16 --max-steps 5 `
    --out-dir runs/smoke_test
```

成功标志：终端打印 `[E001] train_loss=...`，`runs/smoke_test/log.csv` 有一行数据。

**顺手记两个数**：这 5 个 step 花了多少秒、每个 epoch 有多少 step。用它们估算正式训练：

```
每组总时长 ≈ 单 step 秒数 × (训练集张数 / batch size) × epoch 数
```

### D4. 常见报错对照

| 报错 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | 显存不够 | `--batch-size 16`（再不行 8），或 `--image-size 128` |
| `torch.cuda.is_available() == False` | 装了 CPU 版 | 换成 cu121 的 wheel，或接受 CPU 训练（报告里写明） |
| `FileNotFoundError: data...` | 路径不对 | 用绝对路径，或先 `dir data` 确认 |
| `cannot identify image file` | 有坏图 | 正常现象，脚本会跳过；把坏文件删掉更干净 |
| `BrokenPipeError` / 卡在第一个 batch | Windows 多进程 | 保持 `--num-workers 0`（默认值） |
| 中文路径报错 | 编码问题 | 把数据集放到纯英文路径，如 `D:\data\` |

---

## 阶段 E：正式实验 E1–E4

每组都用**同一条命令模板**，只改 `--model` / `--pretrained` / `--augment` 和输出目录：

```powershell
# E1  BasicCNN（基线）
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color `
    --model basiccnn --epochs 20 --batch-size 32 --seed 42 --out-dir runs/E1_basiccnn

# E2  ResNet18 随机初始化
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color `
    --model resnet18 --epochs 20 --batch-size 32 --seed 42 --out-dir runs/E2_resnet18_scratch

# E3  ResNet18 + ImageNet 预训练
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color `
    --model resnet18 --pretrained --epochs 20 --batch-size 32 --seed 42 --out-dir runs/E3_resnet18_pre

# E4  E3 + 数据增强
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color `
    --model resnet18 --pretrained --augment --epochs 20 --batch-size 32 --seed 42 --out-dir runs/E4_resnet18_pre_aug
```

想要日志文件（写附录用）：

```powershell
python train_baseline.py ... *>&1 | Tee-Object runs\E1_basiccnn\train.log
```

### E5. 每组跑完，检查这 7 个文件

```
runs/E1_basiccnn/
├── log.csv                     每个 epoch 一行（loss / acc / macro-F1 / 耗时）
├── args.json                   这次到底用了什么参数（复现依据）
├── data_stats.json             这次到底读了哪些图（和 Table 1 对账）
├── metrics_test.json           PV 测试集结果 ← Table 9 的来源
├── confusion_matrix_test.csv   3×3 混淆矩阵 ← Fig. 4 的来源
├── predictions_test.csv        逐图预测 ← Fig. 6 错例的来源
└── best.pt / last.pt           权重（别提交进仓库）
```

缺文件说明那一步没跑完，不要往下走。

### E6. 三 seed（可选，但很加分）

同一组命令换 `--seed 43`、`--seed 44`，输出目录加后缀 `_s43`、`_s44`。报告里写 `mean ± std`。
时间不够就只跑 seed 42，并在 Limitations 里写"单 seed 初步结果"。

---

## 阶段 F：跨域评估 E5（不重新训练）

```powershell
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color `
    --model resnet18 --pretrained --augment --eval-only `
    --ckpt runs/E4_resnet18_pre_aug/best.pt `
    --plantdoc-root data/PlantDoc-Dataset --plantdoc-split test `
    --out-dir runs/E5_plantdoc_eval
```

产出 `runs/E5_plantdoc_eval/metrics_plantdoc.json`。

**手算一个数**：`Δacc = PV test 的 accuracy − PlantDoc 的 accuracy`。这个 Δ 就是你们 RQ1 的答案。

> 注意：`--plantdoc-split test` 用的是 PlantDoc 的 test 目录（样本少）。
> 想用全部 PlantDoc 图片，改成 `--plantdoc-split all`，但报告里要写清楚用的是哪一种，别混着报。

---

## 阶段 G：批量出图（Fig. 3–6）

```powershell
python plot_results.py --runs runs --out-dir figures
```

产出：

| 文件 | 内容 | 进报告 |
|---|---|---|
| `fig3_curves_E1_basiccnn.png` | 训练/验证 loss + 验证 acc、macro-F1 曲线 | 3.6 或结果节 |
| `fig4_cm_E1_basiccnn_test.png` | in-domain 混淆矩阵热图 | 结果节 |
| `fig4_cm_E5_plantdoc_eval_plantdoc.png` | 跨域混淆矩阵热图 | 跨域分析节 |
| `fig5_compare.png` | E1–E5 的 Acc / macro-F1 对比柱状图 | 结果节（**最有信息量的一张**） |
| `fig6_errors_E1_basiccnn.png` | 错例 6 张（真实→预测+概率） | 错误分析节 |
| `figures/index.md` | 所有图的英文图注 + 本次读到的数值 | 图注直接复制 |

只画某一组：加 `--only E1`。想指定错例用哪次 run：`--errors-run E5_plantdoc_eval`。

---

## 阶段 H：把结果填进 progress report

**顺序很重要：先填数字，再写文字。** 反过来会写出一堆"表现良好"但没有数字的句子。

### H1. 五张表，一张一张填

| 报告中的表 | 需要的数据 | 从哪个文件抄 | 什么时候有 |
|---|---|---|---|
| **Table 1** 数据构成 | 三类张数（两个数据集） | `stats/class_counts.md` | 阶段 C 后 |
| **Table 4** BasicCNN 参数 | 总参数量 | `--print-model` 输出 | 阶段 D 后 |
| **Table 5** ResNet18 参数 | 总/可训练参数量 | `--print-model` 输出 | 阶段 D 后 |
| **Table 6** 超参数实际值 | batch/lr/epochs/环境 | `runs/*/args.json` | 阶段 E 后 |
| **Table 9** 主结果（见下） | acc / macro-F1 / Δ | `runs/*/metrics_*.json` | 阶段 E、F 后 |

**Table 9 模板**（补丁包里没有，直接抄这个）：

| ID | Model | Init. | Aug. | PV-test Acc | PV-test macro-F1 | PD Acc | PD macro-F1 | Δ Acc |
|---|---|---|---|---|---|---|---|---|
| E1 | Basic CNN | Random | No | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| E2 | ResNet18 | Random | No | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| E3 | ResNet18 | ImageNet | No | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| E4 | ResNet18 | ImageNet | Yes | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| E5 | = E4 checkpoint | ImageNet | Yes | （同 E4） | （同 E4） | `[待填]` | `[待填]` | `[待填]` |

> 填表规则：数字照抄 `metrics_*.json` 的 `accuracy` / `macro_f1`，保留 4 位小数；Δ 自己减，别让读者算。

### H2. 文字按这个顺序改（每节都有模板）

1. **3.2 Dataset** → 补丁包 2.5 节（Table 1 + 两句英文）→ 再插 Fig. 1、Fig. 2
2. **3.4 Model Architecture** → 补丁包第三节（Table 4/5，参数量用 D2 的真实值覆盖）
3. **3.6 Experimental Design** → 补丁包第四节（Table 2/3/6；Table 6 的 `[待填]` 用 `args.json` 覆盖）
4. **4. Progress and Current Status** → 补丁包第五节（把你真正做完的条目留下，没做的删掉）
5. **Results（新增小节，建议 4.6 或单独第 5 节）** → 见 H3
6. **Next Steps** → 补丁包第七节周计划表
7. **Contribution** → 补丁包第八节

### H3. 结果节的写法（三段话，每段都要有数字）

> **In-domain.** On the PlantVillage test split (n = `[待填]`), the basic CNN reaches
> `[待填]` accuracy and `[待填]` macro-F1, while ResNet18 with random initialisation reaches
> `[待填]` / `[待填]` and the ImageNet-initialised model reaches `[待填]` / `[待填]` (Table 9).
> The difference between E2 and E3 isolates the effect of pretrained weights at a fixed
> architecture; the difference between E3 and E4 isolates the effect of training-time augmentation.
>
> **Cross-domain.** The same checkpoints, applied to PlantDoc without any retraining, reach
> `[待填]` accuracy (E4 checkpoint), a drop of `[待填]` points relative to the in-domain test split
> (Fig. 5). `[如果某类 recall 明显低，在这里点名]`.
>
> **Error analysis.** Of the `[待填]` misclassified PlantDoc images, `[待填]` are early-blight
> images predicted as late blight; Fig. 6 shows six examples, in which `[描述你看到的：背景杂乱 /
> 病斑占比小 / 逆光]`. This suggests the model relies on `[你的判断]` rather than on lesion shape.

> 没跑完就**不要写这三段**。宁可结果节只有"E1 已完成，其余见 Next Steps"，也不要编。

### H4. 图放哪里 + 图注怎么写

| 图 | 放在 | 图注（英文，可微调） |
|---|---|---|
| Fig. 1 样本网格 | 3.2 | "Randomly drawn samples from the three tomato classes (`[数据集]`, seed = 0)." |
| Fig. 2 类别分布 | 3.2 | "Number of images per class in PlantVillage (colour) and PlantDoc (mapped classes)." |
| Fig. 3 训练曲线 | 3.6 或结果节 | 直接抄 `figures/index.md` 里 fig3 那段 |
| Fig. 4 混淆矩阵 | 结果节 | 同 index.md（fig4），两行/两列并列放：左 in-domain、右 PlantDoc |
| Fig. 5 对比柱状图 | 结果节，Evidence 段 | 同 index.md（fig5） |
| Fig. 6 错例 | 错误分析段 | 同 index.md（fig6） |

> Word 里插图后**一定要检查图号顺序**：正文第一次提到它时用的编号，必须和图下方编号一致。
> 补丁包 1.5 节有一份一致性检查清单。

### H5. 最后三件事

1. 全文搜索 `[待填`，**必须为 0**。
2. 全文搜索 `will be`，逐句确认"确实没做 + 写了谁/何时"。
3. 用补丁包末尾"交付前最后一遍（10 分钟）"那 10 条过一遍。

---

## 阶段 I：打包提交

```
AI3003_组号_番茄叶病/
├── AI3003_report.pdf               报告（≤ 页面限制，相似度 ≤20%）
├── code/
│   ├── data_stats.py
│   ├── make_figures.py
│   ├── train_baseline.py
│   ├── plot_results.py
│   ├── requirements.txt
│   ├── README.md
│   ├── .gitignore
│   └── docs/dataset_sources.md     ← 数据集链接说明（课程要求的那份）
├── figures/                        报告里用到的 6 张图（可选，方便助教核对）
└── runs_summary/                   只放 log.csv + metrics_*.json（不要放 .pt 权重）
```

提交前：`git status` 确认 `data/`、`*.pt`、`.venv/` 都没被提交（`.gitignore` 已经挡住）；
ZIP 里不要放数据集本体（版权 + 体积）。

---

## 10. 时间不够怎么办

### 只有 4 小时：做完 C + D + E1

产出：Table 1、Fig. 1/2、Table 4、`runs/E1`、一段真实的 4.1。
报告里明确写"E2–E5 计划于 Week __ 完成"。这已经是**合格**的进度报告——有数据、有代码、有第一个 baseline。

### 只有 1 天：C + D + E1 + E3 + 出图

E3 和 E1 的对比就能回答"预训练有没有用"的一半问题；Fig. 5 有两根柱子也够了。

### 完整版（2–3 天）：全部九个阶段 + 3 seed

---

## 11. 一页纸检查清单（可打印）

```
[ ] A  torch 能 import，版本抄进 3.6.3
[ ] B  两个数据集就位；commit 抄进 4.1.2
[ ] B  --list-only：3 类匹配成功，6 个番茄目录显示未匹配（正常）
[ ] C  stats/class_counts.md 生成，Table 1 三个数字填好
[ ] C  figures/sample_images.png、class_distribution.png 能打开且不是纯色方块
[ ] D  --dry-run 通过，BAD 图片已删
[ ] D  --print-model 两个参数量抄进 Table 4/5
[ ] D  smoke_test/log.csv 有一行
[ ] E  E1 跑完，runs/E1 有 7 个文件
[ ] E  E2/E3 跑完（至少 E3）
[ ] E  E4 跑完（时间不够可砍，但要在 Limitations 写"未验证增强效果"）
[ ] F  E5 跨域跑完，Δ 算出来
[ ] G  plot_results.py 跑完，figures/index.md 有 5–6 图
[ ] H  Table 9 填完，结果节三段写完
[ ] H  全文 [待填 清零
[ ] H  Contribution 表填完，合计 100%
[ ] I  ZIP 结构符合要求，未打包 data/ 和 .pt
[ ] I  查重 ≤20%，AI 比例 ≤30%（按学校要求自查）
```

---

## 12. 命令速查（全流程复制版）

**Anaconda Prompt（cmd）版**——一行一条，直接粘贴，不用改续行符：

```bat
REM 0) 环境（本机 myenv 已装好 PyTorch，不需要 pip install）
conda activate myenv
cd /d "C:\Users\75342\Desktop\NNDL project\tomato-leaf-transfer"
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"

REM 1) 数据
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git data/PlantDoc-Dataset
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git data/PlantVillage-Dataset
cd data\PlantVillage-Dataset
git sparse-checkout set raw/color/Tomato___healthy raw/color/Tomato___Early_blight raw/color/Tomato___Late_blight
cd ..\..

REM 2) 核对 + 统计 + 基础图
python data_stats.py --root plantvillage=data/PlantVillage-Dataset/raw/color --root plantdoc=data/PlantDoc-Dataset --list-only
python data_stats.py --root plantvillage=data/PlantVillage-Dataset/raw/color --root plantdoc=data/PlantDoc-Dataset --out-csv stats/class_counts.csv --out-md stats/class_counts.md
python make_figures.py --root plantvillage=data/PlantVillage-Dataset/raw/color --root plantdoc=data/PlantDoc-Dataset --out-dir figures --seed 0

REM 3) 冒烟
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --dry-run
python train_baseline.py --model resnet18 --pretrained --print-model
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --model basiccnn --epochs 1 --limit-per-class 50 --max-steps 5 --out-dir runs/smoke_test

REM 4) 正式实验
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --model basiccnn --epochs 20 --seed 42 --out-dir runs/E1_basiccnn
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --model resnet18 --epochs 20 --seed 42 --out-dir runs/E2_resnet18_scratch
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --model resnet18 --pretrained --epochs 20 --seed 42 --out-dir runs/E3_resnet18_pre
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --model resnet18 --pretrained --augment --epochs 20 --seed 42 --out-dir runs/E4_resnet18_pre_aug

REM 5) 跨域
python train_baseline.py --model resnet18 --pretrained --augment --eval-only --ckpt runs/E4_resnet18_pre_aug/best.pt --plantdoc-root data/PlantDoc-Dataset --plantdoc-split test --out-dir runs/E5_plantdoc_eval

REM 6) 出图
python plot_results.py --runs runs --out-dir figures
```

**PowerShell 版**（如果你已经 `conda init powershell`，把反引号当续行符用）：

```powershell
# 0) 环境
conda activate myenv
cd "C:\Users\75342\Desktop\NNDL project\tomato-leaf-transfer"

# 1) 数据
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git data/PlantDoc-Dataset
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git data/PlantVillage-Dataset
cd data/PlantVillage-Dataset
git sparse-checkout set raw/color/Tomato___healthy raw/color/Tomato___Early_blight raw/color/Tomato___Late_blight
cd ..\..

# 2) 核对 + 统计 + 基础图
python data_stats.py --root plantvillage=data/PlantVillage-Dataset/raw/color --root plantdoc=data/PlantDoc-Dataset --list-only
python data_stats.py --root plantvillage=data/PlantVillage-Dataset/raw/color --root plantdoc=data/PlantDoc-Dataset --out-csv stats/class_counts.csv --out-md stats/class_counts.md
python make_figures.py --root plantvillage=data/PlantVillage-Dataset/raw/color --root plantdoc=data/PlantDoc-Dataset --out-dir figures --seed 0

# 3) 冒烟
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --dry-run
python train_baseline.py --model resnet18 --pretrained --print-model
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --model basiccnn --epochs 1 --limit-per-class 50 --max-steps 5 --out-dir runs/smoke_test

# 4) 正式实验
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --model basiccnn --epochs 20 --seed 42 --out-dir runs/E1_basiccnn
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --model resnet18 --epochs 20 --seed 42 --out-dir runs/E2_resnet18_scratch
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --model resnet18 --pretrained --epochs 20 --seed 42 --out-dir runs/E3_resnet18_pre
python train_baseline.py --data-root data/PlantVillage-Dataset/raw/color --model resnet18 --pretrained --augment --epochs 20 --seed 42 --out-dir runs/E4_resnet18_pre_aug

# 5) 跨域
python train_baseline.py --model resnet18 --pretrained --augment --eval-only --ckpt runs/E4_resnet18_pre_aug/best.pt --plantdoc-root data/PlantDoc-Dataset --plantdoc-split test --out-dir runs/E5_plantdoc_eval

# 6) 出图
python plot_results.py --runs runs --out-dir figures
```
