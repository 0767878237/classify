# Dog vs Cat Classification with ResNet
DEMO link: https://dog-cat-classify-app.streamlit.app/
## Overview

This project classifies dog and cat images from:

- `data/train`: `25,000` labeled images
- `data/test`: `12,500` unlabeled images

Current project goals:

- maximize accuracy
- use GPU automatically when available, otherwise fall back to CPU
- provide outputs suitable for both competition-style scoring and business demos

The current project includes:

- `ResNet34` as the accuracy-first option
- `ResNet18` as the lighter and faster option for CPU usage

## Project Structure

```text
classify/
|-- data/
|   |-- train/
|   `-- test/
|-- src/
|   |-- __init__.py
|   |-- datasets.py
|   |-- transforms.py
|   |-- model.py
|   |-- predictor.py
|   |-- train.py
|   |-- infer.py
|   `-- utils.py
|-- configs/
|   |-- resnet18_cpu.yaml
|   |-- resnet18_cpu_fast.yaml
|   |-- resnet18_cpu_ultrafast.yaml
|   `-- resnet34_accuracy.yaml
|-- artifacts/
|-- outputs/
|-- api.py
|-- app.py
|-- requirements.txt
`-- README.md
```

## End-to-End Flow

```text
1. Read images from data/train
2. Split train/valid with stratified sampling
3. Apply augmentation and ImageNet normalization
4. Load a pretrained ResNet
5. Freeze the backbone for warmup
6. Unfreeze and fine-tune
7. Save the best checkpoint by validation F1
8. Run inference on data/test
9. Export:
   - submission.csv
   - predictions.jsonl
```

## Main Components

- `src/datasets.py`
  - Extract labels from `cat.xxx.jpg` and `dog.xxx.jpg`
  - Build datasets for train, validation, and inference

- `src/transforms.py`
  - Training transforms with augmentation
  - Stable evaluation transforms for validation and inference

- `src/model.py`
  - Supports `resnet18` and `resnet34`
  - Replaces the final head for binary classification

- `src/train.py`
  - Automatically chooses `cuda` when available, else `cpu`
  - Supports freeze/unfreeze fine-tuning
  - Uses early stopping
  - Saves `artifacts/best.pt`

- `src/infer.py`
  - Loads the best checkpoint
  - Runs batch inference
  - Exports two output formats

- `src/predictor.py`
  - Shared prediction helper for Streamlit and optional API usage

## Output Files

The project exports two useful output formats.

### `outputs/submission.csv`

Competition-style output:

```csv
id,label
1,0.998421
2,0.031552
```

Meaning:

- `id`: test image id
- `label`: probability that the image is a dog

### `outputs/predictions.jsonl`

Business-demo-friendly output with richer details:

```json
{"image_id":"1","file_path":"data/test/1.jpg","predicted_label":"dog","confidence":0.998421,"dog_probability":0.998421,"cat_probability":0.001579,"review_recommended":false}
```

Useful for:

- dashboards
- API responses
- manual review workflows
- internal demo reports

## Available Configs

### Accuracy-first

File: `configs/resnet34_accuracy.yaml`

- Model: `resnet34`
- Batch size: `8`
- Epochs: `24`
- Freeze warmup: `2`
- Early stopping: `5`
- Best when your main goal is top performance

### CPU-friendly

File: `configs/resnet18_cpu.yaml`

- Model: `resnet18`
- Batch size: `16`
- Epochs: `18`
- Lighter than the accuracy-first setup

### CPU-fast

File: `configs/resnet18_cpu_fast.yaml`

- Model: `resnet18`
- Batch size: `32`
- Image size: `160`
- Trains only the classification head
- Validates less often to reduce epoch time
- Recommended first choice on machines without a GPU
- Uses `num_workers: 0` based on Windows benchmarking

### CPU-ultrafast

File: `configs/resnet18_cpu_ultrafast.yaml`

- Model: `resnet18`
- Batch size: `64`
- Image size: `128`
- Trains only the classification head
- Very sparse validation
- Best for quick experiments and debugging

## Installation

If your current `venv` is broken, recreate it:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you have multiple Python versions:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training

Default training command:

```powershell
python -m src.train
```

Accuracy-first config:

```powershell
python -m src.train --config configs/resnet34_accuracy.yaml
```

CPU-friendly config:

```powershell
python -m src.train --config configs/resnet18_cpu.yaml
```

CPU-fast config:

```powershell
python -m src.train --config configs/resnet18_cpu_fast.yaml
```

CPU-ultrafast config:

```powershell
python -m src.train --config configs/resnet18_cpu_ultrafast.yaml
```

## Inference

After you have `artifacts/best.pt`:

```powershell
python -m src.infer --checkpoint artifacts/best.pt
```

Generated files:

- `outputs/submission.csv`
- `outputs/predictions.jsonl`

## Device Selection

The pipeline automatically chooses:

- `cuda` if `torch.cuda.is_available() == True`
- `cpu` otherwise

You do not need to manually edit the code for CPU-only machines.

## Accuracy Strategy

If you want to push accuracy higher, try this order:

1. Train `resnet34_accuracy.yaml` first.
2. Increase `epochs` if validation metrics are still improving.
3. Increase `tta_passes` during inference.
4. Reduce backbone learning rate slightly.
5. Review failure cases and hard examples.

## CPU Speed Optimization

This project was optimized for CPU-first usage:

- reduced image size in fast modes
- lighter augmentation
- skipped train metrics when not needed
- optional head-only training
- validation not required every epoch
- `num_workers: 0` on Windows when benchmarked faster overall

If your machine is still slow, try in this order:

1. `python -m src.train --config configs/resnet18_cpu_fast.yaml`
2. `python -m src.train --config configs/resnet18_cpu_ultrafast.yaml`
3. `python -m src.train --config configs/resnet18_cpu.yaml`
4. `python -m src.train --config configs/resnet34_accuracy.yaml`

## Training Outputs

- `artifacts/best.pt`: best checkpoint
- `artifacts/best_metrics.json`: best validation metrics
- `outputs/training_history.csv`: per-epoch training history

## Important Notes

- `train.py` saves the best model by `valid_f1`, which is a balanced choice for this binary task.
- On this balanced dataset, higher `f1` usually tracks closely with higher accuracy.
- `ResNet34` on CPU will still be noticeably slower than `ResNet18`.
- If your old epochs took around 30 minutes, the main causes were usually `224x224` images, heavier augmentation, and full validation every epoch.
- Benchmarking on your Windows machine showed `num_workers=0` gave the best total runtime compared with `1` or `2`, mainly because worker startup cost was too high.

## Streamlit Demo

The project now includes:

- `app.py`: English Streamlit UI for demos
- `src/predictor.py`: shared helper that loads the model and predicts directly inside Streamlit
- `api.py`: optional FastAPI backend for future separated deployments

### Recommended Deployment Mode

For Streamlit deployment, the best option is:

```powershell
streamlit run app.py
```

The app will:

- load `artifacts/best.pt` directly
- cache the model with `st.cache_resource`
- avoid any dependency on `127.0.0.1:8000`
- avoid reloading the model on every upload

### Demo Flow

1. Start Streamlit.
2. Open the app in the browser.
3. Upload a new dog or cat image.
4. Click `Run Prediction`.
5. Review:
   - predicted label
   - confidence
   - dog probability
   - cat probability
   - review recommendation

### Streamlit Screens

- `Model Overview`
  - reads best metrics from `artifacts/best_metrics.json`
  - reads charts from `outputs/training_history.csv`

- `Predict New Image`
  - upload a new image
  - predict directly inside the app

- `Batch Predictions`
  - reads `outputs/predictions.jsonl`
  - filters by predicted label
  - filters flagged rows
  - downloads `submission.csv` and `predictions.jsonl`

### When to Use the API

Only use the separate API if you later want:

- a separate frontend
- a mobile app calling prediction over HTTP
- another business system integrating with your model service

Then you can run:

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

For a deployed Streamlit demo, direct prediction inside `app.py` is the simplest and most reliable setup.
