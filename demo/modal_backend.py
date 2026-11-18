"""Deploy with `python -m modal deploy demo/modal_backend.py` from the repo root."""

import os
from pathlib import Path

import modal

from demo.catalog import MAX_IMAGE_BYTES, MODELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = PROJECT_ROOT / "results/models"
if modal.is_local():
    import tomllib

    settings = tomllib.loads((PROJECT_ROOT / ".streamlit/secrets.toml").read_text())
    api_key = settings["api"]["key"]
else:
    api_key = os.environ.get("REALWASTE_API_KEY", "")

app = modal.App("realwaste-demo")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.15.0+cpu", "torchvision==0.29.2+cpu", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("ultralytics==8.4.215", "fastapi[standard]>=0.115,<1")
    .add_local_python_source("demo")
)
for spec in MODELS.values():
    checkpoint = WEIGHTS / spec["checkpoint"]
    if checkpoint.is_file():
        image = image.add_local_file(checkpoint, f"/models/{checkpoint.name}")


@app.cls(
    image=image, cpu=1, memory=2048, timeout=120, scaledown_window=15, max_containers=1,
    secrets=[modal.Secret.from_dict({"REALWASTE_API_KEY": api_key})],
)
class PredictionService:
    @modal.enter()
    def load(self):
        import torch
        from demo.inference import ModelStore

        torch.set_num_threads(2)
        self.models = ModelStore("/models")

    @modal.asgi_app()
    def web(self):
        import io
        import logging
        import secrets
        import warnings
        from time import perf_counter

        from fastapi import Depends, FastAPI, HTTPException, Request
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
        from PIL import Image, ImageOps, UnidentifiedImageError

        api = FastAPI(title="RealWaste classifier", docs_url=None, redoc_url=None, openapi_url=None)
        bearer = HTTPBearer(auto_error=False)

        def authenticate(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
            expected = os.environ.get("REALWASTE_API_KEY", "")
            if not expected or credentials is None or not secrets.compare_digest(credentials.credentials, expected):
                raise HTTPException(401, "Authentication required.")

        @api.get("/models", dependencies=[Depends(authenticate)])
        def models():
            return {"models": self.models.available(), "device": "cpu"}

        @api.post("/predict", dependencies=[Depends(authenticate)])
        async def predict(request: Request, models: str = "yolov8n"):
            start = perf_counter()
            selected = list(dict.fromkeys(models.split(",")))
            available = self.models.available()
            if not selected or any(model_id not in available for model_id in selected):
                raise HTTPException(400, "Choose an available model.")
            chunks = bytearray()
            async for chunk in request.stream():
                chunks.extend(chunk)
                if len(chunks) > MAX_IMAGE_BYTES:
                    raise HTTPException(413, "Image exceeds the 10 MB limit.")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(chunks)) as uploaded:
                        if uploaded.format not in ("JPEG", "PNG"):
                            raise HTTPException(415, "Upload a JPG or PNG image.")
                        image = ImageOps.exif_transpose(uploaded).convert("RGB")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning, ValueError):
                raise HTTPException(400, "The image could not be read. Choose a smaller JPG or PNG.")
            results = []
            for model_id in selected:
                try:
                    results.append(self.models.predict(model_id, image))
                except Exception:
                    logging.exception("Prediction failed for %s", model_id)
                    results.append({"model_id": model_id, "model": MODELS[model_id]["label"],
                                    "error": "Prediction failed for this model. Try again."})
            return {"results": results, "server_ms": (perf_counter() - start) * 1000, "device": "cpu"}

        return api
