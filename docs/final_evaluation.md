# Final evaluation

The ten trained configurations used the same RealWaste split. The [manifests](../data) contain 3,326 training images, 713 validation images, and 713 held-out test images across nine classes. We checked exact file duplicates before splitting and stratified by label with seed 42. The neural experiments used validation macro F1 to choose checkpoints. The results table in the [project README](../README.md#results) reports the selected models' performance on the test split.

## Results by model family

- **Classical features.** The classifiers used HOG edge features and HSV color histograms. Random Forest achieved the highest classical accuracy (59.89%) and RBF SVM achieved the highest classical macro F1 (0.5785). The majority-class baseline was 138 of 713 (19.35%). See the [summary](../results/metrics/classical_summary.csv) and [per-class F1 scores](../results/metrics/classical_perclass_f1.csv).
- **Scratch CNN.** The baseline reached 73.49% accuracy and 0.7424 macro F1. The regularized configuration reached 67.04% and 0.6697. The regularized configuration changed augmentation, dropout, weight decay, learning-rate scheduling, and epoch count together. The 6.45 percentage-point gap measures the difference between those configurations. See the [learning curves](../results/plots/scratch_cnn_learning_curves.png) and [summary](../results/metrics/scratch_cnn_summary.csv).
- **Transfer CNNs.** MobileNetV3 Large, EfficientNet-B0, and GoogLeNet began with ImageNet weights. Each trained its classification head for six epochs with the backbone frozen, then fine-tuned the head and selected backbone blocks for ten epochs. EfficientNet-B0 led this family at 87.38% accuracy and 0.8784 macro F1. It classified one more test image correctly than MobileNetV3 Large. See the [training notes](phase4_progress.md) and [summary](../results/metrics/phase4_summary.csv).
- **YOLOv8n-cls.** The model started from pretrained classification weights and trained its backbone and head together for 25 epochs. Validation macro F1 peaked at epoch 22. The selected checkpoint achieved 91.30% accuracy (651 of 713) and 0.9229 macro F1, 3.93 percentage points ahead of EfficientNet-B0. The training schedules and architectures both changed. See the [training notes](phase5_progress.md) and [summary](../results/metrics/phase5_yolov8n_summary.csv).

## Errors and class-level results

YOLOv8n-cls had its highest F1 on Textile Trash (0.9796), followed by Vegetation (0.9764) and Food Organics (0.9677). It found all 48 true Textile Trash examples and made two extra predictions of that class. Miscellaneous Trash had the lowest F1 (0.8671), with 62 of 75 examples correctly classified. The [YOLO classification report](../results/metrics/phase5_yolov8n_report.csv) includes precision, recall, F1, and support for all nine classes.

The [scratch CNN confusion matrices](../results/plots/scratch_cnn_confusion_matrices.png) show the predicted-class distribution for each true class. In the regularized run, 49% of Miscellaneous Trash and 30% of Glass examples were predicted as Plastic. Vegetation recall rose from 86% to 94%. The [classical confusion matrices](../results/plots/classical_confusion_matrices.png) provide the same breakdown for the four baseline classifiers.

## Demo evidence

The separate [HTTP check](../results/metrics/demo_http_check.json) classified one held-out example, `Food Organics_157.jpg`, with all three deployed models in 3.22 seconds. All three predicted Food Organics. Confidence scores were 99.55% for YOLOv8n, 66.24% for the scratch baseline, and 54.87% for the regularized scratch CNN. These scores describe that example. The [CPU benchmark](modal_cpu_benchmark.md) separately measures first and warm requests for YOLOv8n.

## Scope

The reported ranking comes from one seed and one held-out split. Repeat runs and external photos would help assess the stability of the ranking and generalization to other collection sites. A controlled experiment with matched training settings would help isolate which parts of the YOLO setup contributed to its gain. The original [RealWaste study](https://doi.org/10.3390/info14120633) used a different data split and set of models.
