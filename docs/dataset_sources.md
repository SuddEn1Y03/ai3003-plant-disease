# Dataset sources

Two public datasets are used. Neither is redistributed in this repository; both are downloaded
from their original repositories.

## 1. PlantVillage

| Item | Value |
|---|---|
| Source | https://github.com/spMohanty/PlantVillage-Dataset |
| Paper | Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419 |
| Part used | `raw/color/` only, three tomato classes: `Tomato___healthy`, `Tomato___Early_blight`, `Tomato___Late_blight` |
| Not used | `raw/grayscale/` and `raw/segmented/`: the same leaves in other variants, which would be counted twice |
| Retrieval | `git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git`, then `git sparse-checkout set raw/color/<the three classes>` |
| Version | master branch, downloaded 2026-11-07 |
| Size | 4,500 images (1,591 / 1,000 / 1,909) |

## 2. PlantDoc

| Item | Value |
|---|---|
| Source | https://github.com/pratikkayal/PlantDoc-Dataset |
| Paper | Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. *Proceedings of the 7th ACM IKDD CoDS and 25th COMAD*, 249–253. https://doi.org/10.1145/3371158.3371196 |
| Part used | `Tomato leaf` (healthy), `Tomato Early blight leaf`, `Tomato leaf late blight`; verified against the repository listing |
| Not used | Six other tomato classes (Septoria leaf spot, bacterial spot, mosaic virus, yellow virus, mold, spider mites) and 21 other crops: 2,310 images in total |
| Role | External evaluation only. Never used for training, validation or checkpoint selection |
| Version | master branch (archive download), 2026-11-07 |
| Size | 2,572 images in total, of which 262 map to the three classes (63 / 88 / 111) |

## 3. Reproduction

```bash
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git data/PlantDoc-Dataset
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git data/PlantVillage-Dataset
cd data/PlantVillage-Dataset
git sparse-checkout set raw/color/Tomato___healthy raw/color/Tomato___Early_blight raw/color/Tomato___Late_blight

cd ../..
python data_stats.py --root plantvillage=data/PlantVillage-Dataset/raw/color \
                     --root plantdoc=data/PlantDoc-Dataset --list-only
```

The counts printed by `--list-only` should match Table 1 of the report.

## 4. Licence

Both datasets are public and academic. Copyright remains with their authors; this repository
stores links and statistics only, never the images.
