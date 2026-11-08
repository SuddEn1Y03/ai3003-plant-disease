# 数据集来源说明（提交 ZIP 时附上本文件）

> 课程要求"数据集链接说明"，本文件就是给助教看的那一页。
> 把方括号里的内容替换成你们真实的信息；没填完不要提交。

## 1. PlantVillage

| 项目 | 内容 |
|---|---|
| 来源 | https://github.com/spMohanty/PlantVillage-Dataset |
| 论文 | Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419 |
| 使用部分 | `raw/color/` 下的番茄三类：`Tomato___healthy`、`Tomato___Early_blight`、`Tomato___Late_blight` |
| 未使用部分 | `raw/grayscale/`、`raw/segmented/`（同一批叶片的变体，计入会重复计数） |
| 获取方式 | `[待填：git clone / 官网下载 ZIP]` |
| 版本 / 下载日期 | commit `[待填：git rev-parse HEAD]`，下载于 `[待填：YYYY-MM-DD]` |
| 未纳入的原因 | `[待填：例如无法解码 __ 张]` |

## 2. PlantDoc

| 项目 | 内容 |
|---|---|
| 来源 | https://github.com/pratikkayal/PlantDoc-Dataset |
| 论文 | Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. *Proceedings of the 7th ACM IKDD CoDS and 25th COMAD*, 249–253. https://doi.org/10.1145/3371158.3371196 |
| 使用部分 | `Tomato leaf`（健康）、`Tomato Early blight leaf`、`Tomato leaf late blight`（已按仓库 `master` 目录列表核对） |
| 未使用的番茄类 | `Tomato Septoria leaf spot`、`Tomato leaf bacterial spot`、`Tomato leaf mosaic virus`、`Tomato leaf yellow virus`、`Tomato mold leaf`、`Tomato two spotted spider mites leaf`（共 6 个，均丢弃） |
| 使用方式 | **仅用于评估**（跨域测试集），不参与任何训练、验证或模型选择 |
| 获取方式 | `[待填]` |
| 版本 / 下载日期 | commit `[待填]`，下载于 `[待填]` |
| 类别映射 | 见报告 Table 7 与仓库 `data_stats.py` 中的 `DEFAULT_MAP` |
| 未纳入的类 | 细菌性斑点病、Septoria 叶斑病、花叶病毒、黄化病毒、霉叶、红蜘蛛等番茄类，共 `[待填：__]` 张 |

## 3. 复现方式

```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset.git data/PlantVillage-Dataset
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git      data/PlantDoc-Dataset

python data_stats.py --root plantvillage=data/PlantVillage-Dataset/raw/color \
                     --root plantdoc=data/PlantDoc-Dataset \
                     --out-csv stats/class_counts.csv --out-md stats/class_counts.md
python make_figures.py --root plantvillage=data/PlantVillage-Dataset/raw/color \
                       --root plantdoc=data/PlantDoc-Dataset --out-dir figures
```

得到的样本数应与报告 Table 1 完全一致（`[待填：核对结果，一致 / 差异 __ 张，原因 __]`）。

## 4. 声明

两个数据集均为公开学术数据集，版权归原作者所有。本仓库**不重新分发图片**，只提供下载链接与统计脚本。使用范围限于 AI3003 课程项目。
