# RealWaste experiment progress

MobileNetV3 Large completed on a Tesla T4 using the shared seed-42 split of 3,326 training, 713 validation, and 713 test images.

The selected checkpoint was fine-tuning epoch 10, selected by validation macro F1 of 0.894055. Test accuracy was 0.872370, macro F1 was 0.876053, and cross-entropy loss was 0.555245. Training took 512.50817 seconds (about 8.5 minutes). These values were transcribed from the saved Phase 4 notebook output in Git commit d618b8d.

The classifier-head stage reached validation macro F1 of about 0.8903. Fine-tuning improved this to about 0.8941. Test accuracy improved by 13.75 percentage points over the scratch CNN baseline, and macro F1 improved by about 0.1337.

The run used ImageNet IMAGENET1K_V2 weights, six head-training epochs, ten fine-tuning epochs, and batch size 32. Images were loaded from the runtime-local official dataset cache. Manifests, checkpoints, reports, predictions, and the confusion matrix were saved under the Drive project folder by the notebook.

MobileNetV3 is the only Phase 4 architecture evaluated so far. EfficientNet-B0 and GoogLeNet remain pending. Per-class findings require reading the exported report and predictions. A single seed provides a point estimate of performance.

Phase 5 compares pretrained YOLOv8n classification using the same manifests. Its checkpoint selection uses validation macro F1. The final test predictions will be exported for Phase 6 comparative error analysis.

References
- Saved run https://github.com/Rushikesh-24-max/IML_Realwaste/blob/d618b8d/notebooks/phase4_pretrained_cnns.ipynb
- YOLO classification https://docs.ultralytics.com/tasks/classify/
