from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.predictor import DogCatPredictor


app = FastAPI(title="Dog vs Cat Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: DogCatPredictor | None = None


@app.on_event("startup")
def load_model() -> None:
    global predictor
    checkpoint_path = Path("artifacts/best.pt")
    if not checkpoint_path.exists():
        predictor = None
        return
    predictor = DogCatPredictor(checkpoint_path=checkpoint_path)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model checkpoint is not available.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        result = predictor.predict_bytes(image_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to process image: {exc}") from exc

    return JSONResponse(
        {
            "filename": file.filename,
            "predicted_label": result.predicted_label,
            "confidence": result.confidence,
            "dog_probability": result.dog_probability,
            "cat_probability": result.cat_probability,
            "review_recommended": result.review_recommended,
        }
    )
