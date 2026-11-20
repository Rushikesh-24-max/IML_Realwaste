# Phase 4 completed

All three pretrained CNNs used the shared seed-42 manifests with 3,326 training, 713 validation, and 713 test images. Each ran six head-training epochs and ten fine-tuning epochs with batch size 32. Checkpoints were selected by validation macro F1.

EfficientNet-B0 achieved 87.38% test accuracy, 0.8784 macro F1, and 0.3746 test loss. Fine-tuning epoch 10 was selected with validation macro F1 0.8779. Training took 651.05 seconds (10.85 minutes).

GoogLeNet achieved 84.99% test accuracy, 0.8544 macro F1, and 0.4468 test loss. Fine-tuning epoch 7 was selected with validation macro F1 0.8621. Training took 558.59 seconds (9.31 minutes).

Both models passed a two-image forward/backward check on the T4 before training. GoogLeNet pretrained weights were loaded with auxiliary heads present, then those heads were disabled and removed. This fixes the original constructor argument incompatibility.

The two runs executed notebook revision fcb0dcc through a launch cell in the already connected Colab GPU session. A notebook copy is saved under the Drive project's notebooks folder. Model checkpoints, histories, per-class reports, 713-image prediction CSVs, and confusion matrices are stored under results in Drive. Prediction counts and accuracy were verified against the summary, and checkpoint and plot files were checked. The runtime environment was saved in results/metrics/phase4_remaining_environment.json.

GitHub metrics were transcribed from the Drive readback, including the full-precision MobileNetV3 summary. The combined results are in results/metrics/phase4_summary.csv. Both remaining CNNs used ImageNet IMAGENET1K_V1 weights and the same deterministic preprocessing for training and evaluation as the earlier Phase 4 run.

All planned Phase 4 models have completed. EfficientNet-B0's accuracy exceeds MobileNetV3 by one test image out of 713, so this single-seed difference is small. The [final evaluation](final_evaluation.md) compares these runs with the classical, scratch CNN, and YOLOv8n results.
