"""Benchmark the trained YOLO classifier on Modal CPU.

Run from the repository root with `modal run demo/benchmark_modal.py`.
Requires results/models/realwaste_yolov8n_best.pt.
"""

import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import median
from time import perf_counter

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "results/models/realwaste_yolov8n_best.pt"
REMOTE_CHECKPOINT = "/models/realwaste_yolov8n_best.pt"
app = modal.App("realwaste-cpu-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("ultralytics==8.4.215")
    .add_local_file(CHECKPOINT, remote_path=REMOTE_CHECKPOINT)
)


@app.cls(image=image, cpu=1, memory=2048, timeout=120, scaledown_window=15, max_containers=1)
class Classifier:
    @modal.enter()
    def load(self):
        import os
        import uuid

        start = perf_counter()
        import torch
        import ultralytics
        from ultralytics import YOLO

        self.import_ms = (perf_counter() - start) * 1000
        torch.set_num_threads(2)
        start = perf_counter()
        self.model = YOLO(REMOTE_CHECKPOINT)
        self.model.to("cpu")
        self.load_ms = (perf_counter() - start) * 1000
        self.container_id = uuid.uuid4().hex
        self.call_count = 0
        self.environment = {
            "torch": str(torch.__version__),
            "ultralytics": ultralytics.__version__,
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        }

    @modal.method()
    def predict(self, image_bytes: bytes):
        import io

        from PIL import Image

        start = perf_counter()
        with Image.open(io.BytesIO(image_bytes)) as source:
            uploaded_image = source.convert("RGB")
        decode_ms = (perf_counter() - start) * 1000
        predict_start = perf_counter()
        result = self.model.predict(uploaded_image, device="cpu", imgsz=224, verbose=False)[0]
        predict_ms = (perf_counter() - predict_start) * 1000
        self.call_count += 1
        return {
            "predicted_class": result.names[result.probs.top1],
            "confidence": float(result.probs.top1conf),
            "decode_ms": decode_ms,
            "predict_call_ms": predict_ms,
            "preprocess_ms": result.speed["preprocess"],
            "inference_ms": result.speed["inference"],
            "postprocess_ms": result.speed["postprocess"],
            "server_total_ms": (perf_counter() - start) * 1000,
            "import_ms": self.import_ms,
            "model_load_ms": self.load_ms,
            "container_id": self.container_id,
            "container_call": self.call_count,
            "environment": self.environment,
        }


@app.local_entrypoint()
def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Save the trained YOLOv8n best.pt as {CHECKPOINT}")
    with (PROJECT_ROOT / "data/test.csv").open(newline="", encoding="utf-8") as manifest:
        samples = list(csv.DictReader(manifest))
    sample = samples[0]
    relative_path = PurePosixPath(sample["label"]) / PurePosixPath(sample["filepath"]).name
    source = PROJECT_ROOT / "realwaste-main/RealWaste" / str(relative_path)
    image_bytes = source.read_bytes()
    classifier = Classifier()
    records = []
    for index in range(11):
        start = perf_counter()
        result = classifier.predict.remote(image_bytes)
        result["roundtrip_ms"] = (perf_counter() - start) * 1000
        result["request"] = index + 1
        result["sample"] = str(relative_path)
        result["true_class"] = sample["label"]
        records.append(result)
        print(json.dumps(result), flush=True)

    warm = [record for record in records if record["container_call"] > 1]
    if len(warm) != 10 or len({record["container_id"] for record in records}) != 1:
        raise RuntimeError("Container changed during benchmarking; inspect logs and repeat.")
    if len({record["predicted_class"] for record in records}) != 1:
        raise RuntimeError("Repeated predictions disagree; inspect the results before summarizing.")
    summary = {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "model": "trained RealWaste YOLOv8n classification",
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "sample": str(relative_path),
        "true_class": sample["label"],
        "predicted_class": records[0]["predicted_class"],
        "cpu_physical_cores_requested": 1,
        "memory_mib_requested": 2048,
        "warm_requests": len(warm),
        "first_request_ms": records[0]["roundtrip_ms"],
        "import_ms": records[0]["import_ms"],
        "model_load_ms": records[0]["model_load_ms"],
        "warm_roundtrip_median_ms": median(record["roundtrip_ms"] for record in warm),
        "warm_roundtrip_min_ms": min(record["roundtrip_ms"] for record in warm),
        "warm_roundtrip_max_ms": max(record["roundtrip_ms"] for record in warm),
        "warm_predict_median_ms": median(record["predict_call_ms"] for record in warm),
        "warm_inference_median_ms": median(record["inference_ms"] for record in warm),
        "environment": records[0]["environment"],
        "scope": "One image repeated. Modal SDK RPC includes upload, scheduling and response; no Streamlit/HTTP UI. Image build and deployment setup excluded.",
    }
    output = PROJECT_ROOT / "results/metrics/modal_cpu_benchmark"
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    flattened = [{key: value for key, value in record.items() if key != "environment"} for record in records]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=flattened[0].keys())
        writer.writeheader()
        writer.writerows(flattened)
    print("BENCHMARK SUMMARY", json.dumps(summary, indent=2), flush=True)
