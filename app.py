from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Dog vs Cat Demo", page_icon="🐾", layout="wide")

API_URL = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")
METRICS_PATH = Path("artifacts/best_metrics.json")
HISTORY_PATH = Path("outputs/training_history.csv")
PREDICTIONS_PATH = Path("outputs/predictions.jsonl")


@st.cache_data
def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_history() -> pd.DataFrame | None:
    if not HISTORY_PATH.exists():
        return None
    return pd.read_csv(HISTORY_PATH)


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    if not PREDICTIONS_PATH.exists():
        return None
    records = []
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return None
    return pd.DataFrame(records)


def check_api_health(api_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("model_loaded", False):
            return False, "API is running but the model checkpoint is not loaded."
        return True, "API is ready."
    except Exception as exc:  # noqa: BLE001
        return False, f"API is unavailable: {exc}"


def render_overview() -> None:
    st.subheader("Model Overview")

    metrics = load_metrics()
    history = load_history()

    col1, col2, col3 = st.columns(3)
    if metrics:
        col1.metric("Best F1", f"{metrics['best_score']:.4f}")
        col2.metric("Best Epoch", metrics["epoch"])
        col3.metric("Device", metrics["device"].upper())
    else:
        st.info("No `artifacts/best_metrics.json` file found yet.")

    if history is not None and not history.empty:
        chart_df = history.copy()
        st.line_chart(chart_df.set_index("epoch")[["train_loss", "valid_loss"]], height=260)
        st.line_chart(chart_df.set_index("epoch")[["valid_accuracy", "valid_f1"]], height=260)
    else:
        st.info("No training history available yet.")


def render_prediction(api_url: str) -> None:
    st.subheader("Predict a New Image")

    is_ready, message = check_api_health(api_url)
    if is_ready:
        st.success(message)
    else:
        st.warning(message)

    uploaded_file = st.file_uploader("Upload a cat or dog image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    col1.image(image, caption="Uploaded image", use_container_width=True)

    if col2.button("Run Prediction", type="primary", use_container_width=True):
        if not is_ready:
            col2.error("The API is not ready. Start the backend first.")
            return

        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            response = requests.post(f"{api_url.rstrip('/')}/predict", files=files, timeout=30)
            response.raise_for_status()
            result = response.json()
        except Exception as exc:  # noqa: BLE001
            col2.error(f"Prediction failed: {exc}")
            return

        col2.metric("Predicted Label", result["predicted_label"].title())
        col2.metric("Confidence", f"{result['confidence']:.4f}")
        col2.progress(result["dog_probability"], text=f"Dog probability: {result['dog_probability']:.4f}")
        col2.progress(result["cat_probability"], text=f"Cat probability: {result['cat_probability']:.4f}")

        if result["review_recommended"]:
            col2.warning("Low confidence. Human review is recommended.")
        else:
            col2.success("Prediction confidence is strong enough for demo use.")


def render_predictions_table() -> None:
    st.subheader("Latest Batch Predictions")

    predictions = load_predictions()
    if predictions is None or predictions.empty:
        st.info("No `outputs/predictions.jsonl` file found yet.")
        return

    label_filter = st.selectbox("Filter by label", ["all", "cat", "dog"], index=0)
    review_only = st.checkbox("Show only flagged rows", value=False)

    filtered = predictions.copy()
    if label_filter != "all":
        filtered = filtered[filtered["predicted_label"] == label_filter]
    if review_only:
        filtered = filtered[filtered["review_recommended"]]

    st.dataframe(filtered, use_container_width=True, height=360)

    if Path("outputs/submission.csv").exists():
        st.download_button(
            "Download submission.csv",
            data=Path("outputs/submission.csv").read_bytes(),
            file_name="submission.csv",
            mime="text/csv",
        )

    if PREDICTIONS_PATH.exists():
        st.download_button(
            "Download predictions.jsonl",
            data=PREDICTIONS_PATH.read_bytes(),
            file_name="predictions.jsonl",
            mime="application/json",
        )


def main() -> None:
    st.title("Dog vs Cat Prediction Demo")
    st.caption("Simple English demo UI powered by FastAPI + Streamlit.")

    overview_tab, predict_tab, table_tab = st.tabs(
        ["Model Overview", "Predict New Image", "Batch Predictions"]
    )

    with overview_tab:
        render_overview()

    with predict_tab:
        render_prediction(API_URL)

    with table_tab:
        render_predictions_table()


if __name__ == "__main__":
    main()
