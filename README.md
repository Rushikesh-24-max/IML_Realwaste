# RealWaste Classification

An Introduction to Machine Learning course project that classifies landfill photos into nine waste categories. We evaluated ten model configurations on the same 713-image test set. **YOLOv8n-cls performed best with 91.30% accuracy and 0.9229 macro F1**, and its checkpoint runs in the Streamlit demo.

- **Team:** Harin Savaliya, Rushikesh Talaviya, Divyang Vanjara, Akshat Agrawal, and Anish Reddy
- **Dataset:** [RealWaste, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste)
- **Prior work:** [RealWaste paper](https://doi.org/10.3390/info14120633) and [reference implementation](https://github.com/juliareinhart/waste-classification-machine-learning)

## Final deliverables

- [Project report (PDF)](reports/output/RealWaste_IEEE_Style_Project_Report.pdf)
- [Detailed presentation with speaker notes (PowerPoint)](presentations/output/RealWaste_Project_Presentation_v3_Detailed.pptx)
- [Results and error analysis](docs/final_evaluation.md)
- [Demo setup, deployment, and validation](demo/README.md)

## Results

All figures are from the fixed, stratified test split. Macro F1 gives each class equal weight.

| Model | Family | Test accuracy | Test macro F1 |
| --- | --- | ---: | ---: |
| **YOLOv8n-cls** | Pretrained YOLO classifier | **91.30%** | **0.9229** |
| EfficientNet-B0 | Transfer CNN | 87.38% | 0.8784 |
| MobileNetV3 Large | Transfer CNN | 87.24% | 0.8761 |
| GoogLeNet | Transfer CNN | 84.99% | 0.8544 |
| Scratch CNN baseline | Scratch CNN | 73.49% | 0.7424 |
| Scratch CNN regularized | Scratch CNN | 67.04% | 0.6697 |
| Random Forest | Classical | 59.89% | 0.5465 |
| RBF SVM | Classical | 59.47% | 0.5785 |
| Logistic Regression | Classical | 55.82% | 0.5507 |
| Linear SVM | Classical | 46.14% | 0.4450 |

YOLOv8n-cls classified **651 of 713** test photos correctly, 28 more than EfficientNet-B0. Random Forest led the classical baselines on accuracy, while RBF SVM led them on macro F1. See the [final evaluation](docs/final_evaluation.md) for class-level results and the limits of the comparison.

## Reproduce the workflow

The [shared manifests](data) contain 4,752 images across nine classes. The stratified split uses seed 42 and contains 3,326 training, 713 validation, and 713 test images.

| Step | Notebook | Outputs |
| --- | --- | --- |
| Prepare the dataset | [Setup and split](notebooks/phase0_1_setup_and_split.ipynb) | Check exact duplicates and create split manifests |
| Train classical baselines | [Classical classifiers](notebooks/phase2_classical_baselines.ipynb) | HOG and HSV features, four models, and per-class reports |
| Train scratch CNNs | [Scratch CNN](notebooks/phase3_scratch_cnn.ipynb) | Baseline and regularized variants, learning curves, and confusion matrices |
| Fine-tune pretrained CNNs | [Transfer learning](notebooks/phase4_pretrained_cnns.ipynb) | MobileNetV3 Large, EfficientNet-B0, and GoogLeNet |
| Fine-tune YOLOv8n-cls | [YOLO classification](notebooks/phase5_yolo.ipynb) | Best validation checkpoint, test metrics, and predictions |

The notebooks were run in Colab. Review the notebook setup cells for the expected project and dataset paths. For the original split, reuse the committed manifests. The transfer learning notebook defaults to EfficientNet-B0 and GoogLeNet. Add `mobilenet_v3_large` to `MODEL_NAMES` to run that model too. Dataset images and trained checkpoints are stored separately; the repository contains the code, manifests, [metric exports](results/metrics), and [plots](results/plots).

## Run the demo

The Streamlit app lets you classify a JPG or PNG, compare predictions from the deployed models, and browse all ten saved evaluations. The Modal deployment used YOLOv8n and the two scratch CNN checkpoints for live inference.

```powershell
python -m pip install -r demo/requirements.txt
python -m streamlit run demo/app.py
```

Follow the [demo guide](demo/README.md) to set up the authenticated Modal endpoint, add checkpoints, and configure local Streamlit secrets.

The recorded CPU benchmark measured **11.00 s** for the first SDK request, a **1.68 s** median for ten warm requests, and **12.59 ms** median warm model processing. The separate three-model HTTP check took **3.22 s** end to end. See the [timing methodology](docs/modal_cpu_benchmark.md) and [HTTP results](results/metrics/demo_http_check.json).
