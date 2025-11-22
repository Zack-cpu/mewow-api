"""
FastAPI server for Cat Meow classifier.

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from cat_model_server import predict_from_bytes


app = FastAPI(
    title="Cat Meow Classifier API",
    description="API server for CLAP-based cat meow classifier",
    version="1.0.0",
)

# Allow your iOS app to call this (you can restrict origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev; in production, use your actual domain / IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Cat Meow Classifier API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded audio file and return prediction.
    The iOS app should send multipart/form-data with field name 'file'.
    """
    # Read audio bytes
    audio_bytes = await file.read()

    # Run model
    label, confidence, probs = predict_from_bytes(audio_bytes)

    # Return JSON response
    return {
        "label": label,
        "confidence": confidence,
        "probs": probs,
    }