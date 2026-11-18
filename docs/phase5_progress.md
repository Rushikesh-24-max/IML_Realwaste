# Phase 5 completed

YOLOv8n classification completed 25 epochs on Tesla T4 with the shared seed-42 manifests. Validation macro F1 selected epoch 22. Test accuracy was 91.30435% and macro F1 was 0.922906. Training took 815.478 seconds. The run used Ultralytics 8.4.215, torch 2.14.1+cu130, Python 3.13.16, 224-pixel inputs, batch 32, and AdamW.

The run was launched from an added cell in the existing Phase 4 Colab session to reuse its connected GPU. It executed the Phase 5 notebook at commit 88da8f3. A copy of that notebook is saved under the Drive project's notebooks folder. The original Phase 4 training was not rerun.

Drive exports were verified in IML_RealWaste/results/metrics, with weights and full run artifacts in results/yolo/phase5_yolov8n_20261118_182913. Both Phase 4 and Phase 5 progress notes were written and read back in the Drive docs folder. GitHub contains the summary and report transcribed from that readback. Full prediction CSVs, plots, and weights remain in Drive.

YOLO improved accuracy over MobileNetV3 by about 4.07 percentage points and macro F1 by 0.0469 on the common test set. Strongest class F1 scores were Textile Trash (0.9796), Vegetation (0.9764), and Food Organics (0.9677). Miscellaneous Trash remained the lowest at 0.8671.

EfficientNet-B0 and GoogLeNet remain pending. Phase 6 can compare completed models using reports and predictions, then inspect misclassified images. Report these as single-seed comparisons.
