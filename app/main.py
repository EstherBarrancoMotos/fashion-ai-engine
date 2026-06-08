from fastapi import FastAPI, File, HTTPException, UploadFile

from fashion_ai_engine.inference import FashionPredictor

app = FastAPI(
    title="Fashion AI Engine",
    version="0.1.0",
    description="Image classification API for fashion products.",
)

predictor = FashionPredictor.from_environment()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": predictor.model_loaded,
        "model_path": str(predictor.model_path),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return predictor.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
