# 🗑️ RealWaste Classification — IML Course Project

**Team:** Harin Savaliya · Rushikesh Talaviya · Divyang Vanjara · Akshat Agrawal · Anish Reddy  
**Course:** Introduction to Machine Learning (IML), 5th Semester IT  
**Dataset:** [RealWaste — UCI ML Repository #908](https://archive.ics.uci.edu/dataset/908/realwaste) | CC BY 4.0  
**Reference repo:** [waste-classification-machine-learning](https://github.com/juliareinhart/waste-classification-machine-learning)

---

## Project Structure

```
IML_Realwaste/
│
├── notebooks/                          ← All our notebooks (run in order)
│   ├── phase0_1_setup_and_split.ipynb  ← Setup + shared data split
│   ├── phase2_classical_baselines.ipynb← Classical models (HOG + Color Hist)
│   ├── phase3_scratch_cnn.ipynb        ← Custom 4-layer CNN (from scratch)
│   ├── phase4_pretrained_cnns.ipynb    ← Transfer learning (MobileNet, EfficientNet, GoogLeNet)
│   ├── phase5_yolo.ipynb               ← YOLOv8n Classification
│   └── phase6_analysis.ipynb           ← Comparative evaluation & error analysis
│
├── waste-classification-machine-learning/  ← Original cloned repo (reference)
│
├── data/                               ← Shared split CSVs & metadata
│   ├── train.csv                       (3,326 images, 70%)
│   ├── val.csv                         (713 images, 15%)
│   ├── test.csv                        (713 images, 15%)
│   └── dataset_meta.json
│
└── results/
    ├── metrics/                        ← Per-model metric CSVs & reports
    ├── plots/                          ← Confusion matrices, comparisons, distributions
    └── models/                         ← Saved weights (.pth / .pt, gitignored)
```

## How to Run (Google Colab)

Click the badges to open notebooks directly in Colab:

| # | Notebook | Status | Open in Colab |
|---|---|---|---|
| 0+1 | Setup & Split | ✅ Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rushikesh-24-max/IML_Realwaste/blob/master/notebooks/phase0_1_setup_and_split.ipynb) |
| 2 | Classical Baselines | ✅ Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rushikesh-24-max/IML_Realwaste/blob/master/notebooks/phase2_classical_baselines.ipynb) |
| 3 | Scratch CNN (Custom 4-layer) | ✅ Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rushikesh-24-max/IML_Realwaste/blob/master/notebooks/phase3_scratch_cnn.ipynb) |
| 4 | Pretrained CNNs | Completed | [Open in Colab](https://colab.research.google.com/github/Rushikesh-24-max/IML_Realwaste/blob/master/notebooks/phase4_pretrained_cnns.ipynb) |
| 5 | YOLOv8n-cls | Completed | [Open in Colab](https://colab.research.google.com/github/Rushikesh-24-max/IML_Realwaste/blob/master/notebooks/phase5_yolo.ipynb) |
| 6 | Analysis | ⏳ Planned | — |

---

## Benchmark Results (Test Set: 713 Images)

### Phase 3: Custom 4-Layer Scratch CNN (Trained from Scratch)

| Model Variant | Test Accuracy (%) | Test Macro F1 | Test Loss | Key Characteristics |
|---|---:|---:|---:|---|
| **Scratch CNN (Baseline / No Aug)** | **73.49%** | **0.7424** | **0.7564** | No augmentation, Dropout=0.0, 20 epochs |
| **Scratch CNN (Regularized + Aug)** | 67.04% | 0.6697 | 0.8655 | Data Aug (flips, rot, jitter, crop) + Dropout=0.4 + Weight Decay, 25 epochs |

#### Key Insights from Scratch CNN:
- **Major Leap from Classical:** The baseline Scratch CNN achieved **73.49% accuracy and 0.7424 Macro F1**, outperforming the best classical model (Random Forest: 59.89%, RBF SVM: 0.5785 Macro F1) by **+13.6% accuracy and +0.164 Macro F1**. This proves the superiority of end-to-end learned visual representations over hand-crafted HOG/Color descriptors.
- **Ablation Observation (Baseline vs. Regularized):** The unaugmented baseline converged faster and reached higher test performance within 20–25 epochs, whereas heavy regularization (Dropout 0.4 + aggressive geometric/color transformations) constrained capacity in this compact 390k-parameter network. However, on **Vegetation**, the regularized model achieved higher performance (**0.910 vs 0.882 F1**), showing that augmentation improved robustness against natural visual variations.

---

### Phase 2: Classical Baselines (HOG + HSV Color Histogram)

| Model | Accuracy (%) | Macro F1 | Train Time (s) |
|---|---:|---:|---:|
| **Random Forest** (200 trees) | **59.89%** | 0.5465 | 79.3s |
| **RBF SVM** ($C=10, \gamma=\text{scale}$) | **59.47%** | **0.5785** | 148.7s |
| **Logistic Regression** ($C=1.0$) | 55.82% | 0.5507 | 5.4s |
| **Linear SVM** ($C=1.0$) | 46.14% | 0.4450 | 148.2s |
| *Majority Class Baseline (Plastic)* | *19.4%* | *—* | *—* |

#### Key Insights from Classical Models:
- **Best overall balance:** **RBF SVM** achieved the highest Macro F1 (0.5785) and 59.47% accuracy.
- **Easiest classes:** **Vegetation** (F1: 0.795) and **Cardboard** (F1: 0.767) — both have distinct color profiles and clear edge textures that HOG and Color Hist capture well.
- **Hardest classes:** **Miscellaneous Trash** (F1: 0.163–0.370) and **Textile Trash** (F1: 0.185–0.415) — high visual variance and fewer samples make hand-crafted features struggle.
- All models comfortably beat the 19.4% majority baseline, confirming the utility of hand-crafted feature baselines before moving to deep representations.

## Phase 4 transfer learning result

MobileNetV3 Large reached 87.24% test accuracy and 0.8761 macro F1. Validation selected fine-tuning epoch 10. The run took about 8.5 minutes on a Tesla T4. See [experiment progress](docs/phase4_progress.md) for settings and remaining experiments.

## Phase 5 YOLOv8n classification result

YOLOv8n reached 91.30% test accuracy and 0.9229 macro F1 on the shared 713-image test set. Validation macro F1 selected epoch 22 of 25. Training took about 13.6 minutes on Tesla T4. See [Phase 5 progress](docs/phase5_progress.md).

### Remaining Phase 4 CNN results

EfficientNet-B0 achieved 87.38% test accuracy and 0.8784 macro F1, with fine-tuning epoch 10 selected by validation macro F1. GoogLeNet achieved 84.99% accuracy and 0.8544 macro F1, with epoch 7 selected. All planned Phase 4 models are complete. Full results are in [the combined summary](results/metrics/phase4_summary.csv) and [experiment notes](docs/phase4_progress.md).
