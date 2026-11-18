# RealWaste working demo

The Streamlit interface runs on the presentation computer. An authenticated Modal HTTP API performs CPU inference using our trained checkpoints.

## Interface

- **Classify** accepts a JPG/PNG up to 10 MB or one of three attributed RealWaste test examples. It shows the image, predicted class, confidence, three highest scores, model processing time, model-loading time, and the last request time.
- **Compare models** runs the selected live models sequentially on the same image. It also displays all ten recorded experiment configurations on the shared 713-image test set.
- **About the project** explains the dataset, categories, methods, team, evaluation protocol, and intended inputs.

Each new image/model pair is sent for inference when a prediction button is clicked. Successful predictions are retained in the browser's Streamlit session, so revisiting tabs or repeating a comparison reuses those results. A new session or image causes fresh requests. Processing time covers model prediction and preprocessing. Request time also includes image transfer, server handling, and cold starts when applicable. The loading measurement is stored with the original result when it is cached.

## Files

```text
demo/
  app.py                 Streamlit controls, preview, API calls, and session cache
  modal_backend.py       Authenticated /models and /predict HTTP routes on Modal
  inference.py           Lazy checkpoint loading and original evaluation transforms
  catalog.py             Model identifiers, checkpoint filenames, and class order
  requirements.txt       Local UI and deployment dependencies
  samples/               Three RealWaste test photographs
  benchmark_modal.py     Earlier single-image CPU timing experiment
.streamlit/
  config.toml            UI theme and upload-size setting
  secrets.toml           Local API URL and key, excluded from Git
results/models/          Trained weights, excluded from Git
results/metrics/         Recorded model evaluations
```

## Run locally

From the repository root, use a working Python environment.

```powershell
python -m pip install -r demo/requirements.txt
python -m streamlit run demo/app.py
```

The configured deployment's URL and API key are read from `.streamlit/secrets.toml`. The key stays on the Streamlit server. On another computer, create that file locally.

```toml
[api]
url = "YOUR_DEPLOYED_MODAL_URL"
key = "YOUR_APP_SPECIFIC_KEY"
```

## Deploy or add checkpoints

Authenticate the local Modal CLI with `modal setup`. Put each checkpoint under `results/models` using the names below. Create a random app-specific API key in the secrets file before deploying.

```powershell
python -m modal deploy demo/modal_backend.py
```

The deployment uploads the available checkpoints to Modal. Update `api.url` in the local secrets file to the URL returned by deployment. Existing keys are retained during redeployment. A fresh key requires updating the client and redeploying together.

- YOLOv8n uses `realwaste_yolov8n_best.pt`.
- Scratch baseline uses `best_scratch_cnn_baseline.pth`.
- Scratch regularized uses `best_scratch_cnn_regularized.pth`.
- MobileNetV3 uses `phase4_mobilenet_v3_large_finetune.pth`.
- EfficientNet-B0 uses `phase4_efficientnet_b0_finetune.pth`.
- GoogLeNet uses `phase4_googlenet_finetune.pth`.

The fine-tuned checkpoints listed here are the variants selected in our experiments. The API advertises models with checkpoint files present in its deployment, and the UI lists those models for live inference. Adding files requires redeployment. Live classical-model inference requires exporting the fitted classifiers and scaler alongside their HOG/HSV feature pipeline. Their evaluation results are already shown.

## Request flow

The browser uploads an image to the local Streamlit server. Streamlit validates and previews it, then posts the original bytes to `/predict?models=yolov8n` with an app-specific bearer key. Modal validates the image, loads the selected model if needed, applies its notebook evaluation preprocessing, and returns JSON scores and timings. Multiple selected model IDs are comma-separated. `/models` returns the available checkpoint IDs. Both routes require authentication.

The backend uses one physical CPU core, 2 GiB RAM, two PyTorch threads, and a maximum of one container. Models load lazily and are reused in that container. It scales down after 15 seconds without requests. Modal can terminate/recreate containers, which repeats initialization. Usage is drawn from the account's compute allowance.

## Sample attribution

The sample photos are unchanged images from the shared RealWaste test manifest. They are included for classroom demonstration and attributed to Sam Single, Saeid Iranmanesh, and Raad Raad. Dataset source and license information are available at https://archive.ics.uci.edu/dataset/908/realwaste. The local dataset README states CC BY-NC-SA 4.0; retain that attribution and restriction for the bundled photos.

## Validation

The prior YOLO benchmark measured an 11.00-second first SDK request and a 1.68-second median for ten warm SDK requests. The HTTP and UI flow is checked separately. Those SDK timings are recorded in `docs/modal_cpu_benchmark.md`.

The deployed HTTP service passed authentication checks, invalid-image and unknown-model rejection, and predictions from all three available checkpoints. The three-model response took 3.22 seconds in the recorded API check. Each returned distribution summed to one. The result is saved in `results/metrics/demo_http_check.json`.

Streamlit AppTest passed the Classify and Compare workflows against the real API. Repeating Classify and Compare reused predictions, and a comparison requested only its missing model. Browser review verified the sample-image preview, prediction display, and comparison screen. The browser automation extension could not supply a file to the native file picker because its file-access permission is disabled. Manual file selection remains available in the UI; the image upload control and server validation were checked separately.

The current deployment contains YOLOv8n and both scratch CNN checkpoints. The other three pretrained models require their saved checkpoint files to be copied from Drive before redeployment.
