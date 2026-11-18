"""Streamlit interface. All model inference runs on Modal."""

import hashlib
import io
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from demo.catalog import CLASSES, MAX_IMAGE_BYTES, MODELS


ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="RealWaste | Waste classification", page_icon="♻", layout="wide")
st.markdown("""
<style>
 .block-container { max-width: 1100px; padding-top: 2rem; }
 [data-testid="stMetricValue"] { font-size: 1.65rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60, show_spinner=False)
def available_models(url, key):
    response = requests.get(url + "/models", headers={"Authorization": "Bearer " + key}, timeout=(10, 90))
    response.raise_for_status()
    return [name for name in response.json()["models"] if name in MODELS]


@st.cache_data(show_spinner=False)
def test_results():
    metrics = ROOT / "results/metrics"
    rows = []
    for filename in ("classical_summary.csv", "scratch_cnn_summary.csv"):
        for row in pd.read_csv(metrics / filename).to_dict("records"):
            rows.append({"Model": row["Model"], "Accuracy (%)": row["Accuracy (%)"], "Macro F1": row["Macro F1"]})
    for row in pd.read_csv(metrics / "phase4_summary.csv").to_dict("records"):
        rows.append({"Model": MODELS[row["model"]]["label"], "Accuracy (%)": row["test_accuracy"] * 100,
                     "Macro F1": row["test_macro_f1"]})
    row = pd.read_csv(metrics / "phase5_yolov8n_summary.csv").iloc[0]
    rows.append({"Model": "YOLOv8n", "Accuracy (%)": row["test_accuracy"] * 100, "Macro F1": row["test_macro_f1"]})
    return pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)


def predict(image_bytes, selected, url, key):
    digest = hashlib.sha256(image_bytes).hexdigest()
    cache = st.session_state.setdefault("predictions", {})
    missing = [name for name in selected if digest + ":" + name not in cache]
    if missing:
        start = time.perf_counter()
        response = requests.post(
            url + "/predict", params={"models": ",".join(missing)}, data=image_bytes,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/octet-stream"},
            timeout=(10, 120),
        )
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        body = response.json()
        for result in body["results"]:
            if "error" not in result:
                cache[digest + ":" + result["model_id"]] = result
            else:
                st.error(result["model"] + ": " + result["error"])
        st.session_state["last_request"] = {"seconds": elapsed, "count": len(missing), "digest": digest}
        while len(cache) > 36:
            del cache[next(iter(cache))]
    return [cache[digest + ":" + name] for name in selected if digest + ":" + name in cache]


def run_prediction(image_bytes, selected, url, key):
    try:
        with st.spinner("Classifying on Modal. The first request may take longer while models load."):
            return predict(image_bytes, selected, url, key)
    except requests.Timeout:
        st.error("The service took too long to respond. Please try again.")
    except requests.HTTPError as error:
        code = error.response.status_code
        if code in (401, 403):
            st.error("The API credentials were rejected. Check the local secrets configuration.")
        elif code in (400, 413, 415):
            st.error("The service could not accept this image or model selection. Try a smaller JPG or PNG.")
        else:
            st.error("The prediction service is temporarily unavailable. Please retry.")
    except (requests.RequestException, ValueError, KeyError):
        st.error("Could not reach the prediction service. Check your connection and retry.")
    return []


st.title("RealWaste classification")
st.write("Upload a waste photograph and explore predictions from our trained models.")
st.caption("Nine waste categories · CPU inference on Modal")

try:
    settings = st.secrets["api"]
    api_url, api_key = settings["url"].rstrip("/"), settings["key"]
except (KeyError, FileNotFoundError):
    st.error("Set the Modal API URL and key in .streamlit/secrets.toml. Setup instructions are in demo/README.md.")
    st.stop()

try:
    with st.spinner("Connecting to the prediction service…"):
        live_models = available_models(api_url, api_key)
except (requests.RequestException, ValueError, KeyError):
    live_models = []
    st.warning("The prediction service is unavailable. Saved results and project information remain accessible.")

classify_tab, compare_tab, about_tab = st.tabs(["Classify", "Compare models", "About the project"])
image_bytes = None

with classify_tab:
    input_column, result_column = st.columns([1, 1], gap="large")
    with input_column:
        source = st.radio("Image source", ["Upload", "Sample image"], horizontal=True)
        if source == "Upload":
            upload = st.file_uploader("Choose a waste image", type=["jpg", "jpeg", "png"], max_upload_size=10)
            if upload is not None:
                image_bytes = upload.getvalue()
        else:
            samples = sorted((ROOT / "demo/samples").glob("*.jpg"))
            if samples:
                sample = st.selectbox("Try a test image", samples, format_func=lambda path: path.stem.replace("_", " "))
                image_bytes = sample.read_bytes()
        if image_bytes:
            try:
                if len(image_bytes) > MAX_IMAGE_BYTES:
                    raise ValueError("Image exceeds 10 MB.")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(image_bytes)) as uploaded:
                        if uploaded.format not in ("JPEG", "PNG"):
                            raise ValueError("Unsupported image format.")
                        preview = ImageOps.exif_transpose(uploaded).convert("RGB")
                st.image(preview, width="stretch")
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                st.error("Choose a readable JPG or PNG image up to 10 MB.")
                image_bytes = None
    with result_column:
        st.subheader("Prediction")
        if live_models:
            chosen = st.selectbox("Model", live_models, format_func=lambda name: MODELS[name]["label"])
            if st.button("Classify image", type="primary", disabled=image_bytes is None, width="stretch"):
                run_prediction(image_bytes, [chosen], api_url, api_key)
            if image_bytes:
                digest = hashlib.sha256(image_bytes).hexdigest()
                result = st.session_state.get("predictions", {}).get(digest + ":" + chosen)
                if result:
                    st.subheader(result["predicted_class"])
                    confidence, latency = st.columns(2)
                    confidence.metric("Model confidence", f"{result['confidence']:.1%}")
                    latency.metric("Prediction processing", f"{result['predict_ms']:.1f} ms")
                    st.caption(f"Model loading for this response · {result['model_load_ms'] / 1000:.2f} s")
                    for score in result["scores"][:3]:
                        st.progress(score["score"], text=f"{score['class']} · {score['score']:.1%}")
                    request = st.session_state.get("last_request", {})
                    if request.get("digest") == digest:
                        st.caption(f"Last request · {request['seconds']:.2f} s for {request['count']} model(s), including transfer and startup when needed.")
                    st.caption("Predictions for this image are retained in this session. Confidence is the model’s score for this image.")

with compare_tab:
    st.subheader("Predict the same image with different models")
    st.caption("Uses the image selected in Classify. Models run sequentially on the same cloud CPU.")
    if live_models:
        selected = st.multiselect("Models to compare", live_models, default=live_models[:2],
                                 format_func=lambda name: MODELS[name]["label"])
        if st.button("Compare this image", disabled=image_bytes is None or not selected, type="primary"):
            run_prediction(image_bytes, selected, api_url, api_key)
        if image_bytes:
            digest = hashlib.sha256(image_bytes).hexdigest()
            cache = st.session_state.get("predictions", {})
            comparison = [cache[digest + ":" + name] for name in selected if digest + ":" + name in cache]
            if comparison:
                if len(comparison) < len(selected):
                    st.caption(f"{len(comparison)} of {len(selected)} selected models has a saved prediction. Compare this image to run the remaining models.")
                st.dataframe(pd.DataFrame([{
                    "Model": result["model"], "Prediction": result["predicted_class"],
                    "Confidence (%)": result["confidence"] * 100, "Processing (ms)": result["predict_ms"],
                    "Model loading (s)": result["model_load_ms"] / 1000,
                } for result in comparison]), hide_index=True, width="stretch",
                    column_config={"Confidence (%)": st.column_config.NumberColumn(format="%.1f"),
                                   "Processing (ms)": st.column_config.NumberColumn(format="%.1f"),
                                   "Model loading (s)": st.column_config.NumberColumn(format="%.2f")})
                request = st.session_state.get("last_request", {})
                if request.get("digest") == digest:
                    st.caption(f"Last request · {request['seconds']:.2f} s for {request['count']} newly requested model(s). Existing predictions are reused.")
    st.divider()
    st.subheader("Saved test-set results")
    st.caption("713 shared test images · seed 42 · Accuracy and macro F1 from the recorded experiments.")
    st.dataframe(test_results(), hide_index=True, width="stretch", column_config={
        "Accuracy (%)": st.column_config.NumberColumn(format="%.2f"),
        "Macro F1": st.column_config.NumberColumn(format="%.4f"),
    })
    st.caption("Live selection lists the checkpoints deployed to Modal. Classical models currently have saved evaluation results; their fitted feature pipelines have yet to be exported.")

with about_tab:
    st.subheader("Machine learning for waste sorting")
    st.write("Our Introduction to Machine Learning course project classifies photographs of landfill waste into nine material categories.")
    st.write(" · ".join(CLASSES))
    st.subheader("Dataset and experiments")
    st.write("RealWaste provides 4,752 images in our project subset. We use a shared stratified split of 3,326 training, 713 validation, and 713 test images with seed 42.")
    st.write("We compared HOG and HSV color features with classical classifiers, a four-stage CNN trained from scratch, ImageNet transfer learning, and YOLOv8n classification. Pretrained model checkpoints were selected using validation macro F1.")
    st.write("Use photographs with one dominant waste category. Predictions on mixed scenes and images outside the dataset can be less reliable. Reported test results are from a single seed.")
    st.subheader("Team")
    st.write("Harin Savaliya · Rushikesh Talaviya · Divyang Vanjara · Akshat Agrawal · Anish Reddy")
    st.markdown("[Project repository](https://github.com/Rushikesh-24-max/IML_Realwaste) · [RealWaste dataset](https://archive.ics.uci.edu/dataset/908/realwaste) · [Original paper](https://doi.org/10.3390/info14120633)")
    st.caption("Dataset and sample images attributed to Sam Single, Saeid Iranmanesh, and Raad Raad. See the dataset source for license terms.")
