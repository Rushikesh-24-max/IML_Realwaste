"""CPU inference using the architectures and evaluation transforms from our notebooks."""

from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torchvision import models, transforms

from demo.catalog import CLASSES, MODELS


class RealWasteScratchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = self.stage(3, 32)
        self.stage2 = self.stage(32, 64)
        self.stage3 = self.stage(64, 128)
        self.stage4 = self.stage(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(256, len(CLASSES))

    @staticmethod
    def stage(inputs, outputs):
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outputs), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )

    def forward(self, inputs):
        features = self.stage4(self.stage3(self.stage2(self.stage1(inputs))))
        features = torch.flatten(self.global_pool(features), 1)
        return self.classifier(self.dropout(features))


def load_classifier(model_id, checkpoint):
    if model_id == "yolov8n":
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
        model.to("cpu")
        if set(model.names.values()) != set(CLASSES):
            raise ValueError("YOLO checkpoint class names differ from RealWaste.")
        return model, None
    if model_id == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
        transform = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()
    elif model_id == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
        transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    elif model_id == "googlenet":
        model = models.googlenet(weights=None, aux_logits=False, init_weights=False, transform_input=True)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        transform = models.GoogLeNet_Weights.IMAGENET1K_V1.transforms()
    elif model_id in ("scratch_baseline", "scratch_regularized"):
        model = RealWasteScratchCNN()
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError("Unknown model.")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval(), transform


class ModelStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.loaded = {}

    def available(self):
        return [key for key, spec in MODELS.items() if (self.directory / spec["checkpoint"]).is_file()]

    def predict(self, model_id, image):
        load_ms = 0.0
        if model_id not in self.loaded:
            start = perf_counter()
            self.loaded[model_id] = load_classifier(model_id, self.directory / MODELS[model_id]["checkpoint"])
            load_ms = (perf_counter() - start) * 1000
        model, transform = self.loaded[model_id]
        start = perf_counter()
        with torch.inference_mode():
            if model_id == "yolov8n":
                output = model.predict(image, device="cpu", imgsz=224, verbose=False)[0]
                probabilities = output.probs.data.cpu().tolist()
                names = [output.names[index] for index in range(len(probabilities))]
            else:
                probabilities = model(transform(image).unsqueeze(0)).softmax(dim=1)[0].tolist()
                names = CLASSES
        scores = sorted(
            [{"class": name, "score": float(score)} for name, score in zip(names, probabilities)],
            key=lambda item: item["score"], reverse=True,
        )
        return {
            "model_id": model_id, "model": MODELS[model_id]["label"],
            "predicted_class": scores[0]["class"], "confidence": scores[0]["score"],
            "scores": scores, "predict_ms": (perf_counter() - start) * 1000,
            "model_load_ms": load_ms,
        }
