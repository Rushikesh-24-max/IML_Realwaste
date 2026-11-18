"""Checkpoint names and class order shared by the UI and inference service."""

CLASSES = [
    "Cardboard", "Food Organics", "Glass", "Metal", "Miscellaneous Trash",
    "Paper", "Plastic", "Textile Trash", "Vegetation",
]

MODELS = {
    "yolov8n": {"label": "YOLOv8n", "checkpoint": "realwaste_yolov8n_best.pt"},
    "mobilenet_v3_large": {"label": "MobileNetV3 Large", "checkpoint": "phase4_mobilenet_v3_large_finetune.pth"},
    "efficientnet_b0": {"label": "EfficientNet-B0", "checkpoint": "phase4_efficientnet_b0_finetune.pth"},
    "googlenet": {"label": "GoogLeNet", "checkpoint": "phase4_googlenet_finetune.pth"},
    "scratch_baseline": {"label": "Scratch CNN baseline", "checkpoint": "best_scratch_cnn_baseline.pth"},
    "scratch_regularized": {"label": "Scratch CNN regularized", "checkpoint": "best_scratch_cnn_regularized.pth"},
}

MAX_IMAGE_BYTES = 10 * 1024 * 1024
