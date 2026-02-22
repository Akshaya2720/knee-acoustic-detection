"""
app.py  –  FastAPI backend for Knee Acoustic Signal Classifier
Run with:  uvicorn app:app --reload --port 8000
# Model reloaded: 2026-02-22
"""
import os
import json
import tempfile
import warnings
import numpy as np
import librosa
import joblib

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
# App Configuration
# ──────────────────────────────────────────────────
app = FastAPI(
    title="Knee Acoustic Signal Classifier",
    description="Classifies knee joint acoustic signals as Normal or Abnormal",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (frontend served locally)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────
# Load model artifacts at startup
# ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = os.path.join(BASE_DIR, "knee_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "knee_scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "knee_model_meta.json")

model  = None
scaler = None
meta   = {}

def load_artifacts():
    global model, scaler, meta
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("⚠️  Model files not found. Run the Colab notebook first and")
        print("    place knee_model.pkl + knee_scaler.pkl in this directory.")
        return
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    print(f"✅ Loaded model: {meta.get('model_name', 'Unknown')}")
    print(f"   Accuracy: {meta.get('test_accuracy', 0)*100:.1f}% | AUC: {meta.get('roc_auc', 0):.3f}")

load_artifacts()

# ──────────────────────────────────────────────────
# Classification threshold
# Lower than 0.5 to catch more Abnormal cases —
# in a medical context, a false negative (missing
# an abnormal condition) is worse than a false alarm.
# Range: 0.0 – 0.5  (default 0.5 = standard majority vote)
# ──────────────────────────────────────────────────
ABNORMAL_THRESHOLD = 0.40  # classify as Abnormal if P(Abnormal) >= this

# ──────────────────────────────────────────────────
# Signal Validation — detects non-joint audio
# ──────────────────────────────────────────────────
MIN_CONFIDENCE_FOR_VALID = 0.55  # if max(proba) < this → uncertain
MAX_VALID_DURATION_SEC   = 60.0  # recordings longer than 1 min are suspicious
MIN_RMS_ENERGY           = 1e-5  # effectively silent
MAX_RMS_ENERGY           = 0.85  # clipping / very loud non-joint audio (music etc.)

def validate_signal(signal: np.ndarray, sr: int, proba) -> tuple:
    """
    Returns (is_uncertain: bool, warning: str)
    Checks signal characteristics to detect non-joint recordings.
    """
    duration   = len(signal) / sr
    rms        = float(np.sqrt(np.mean(signal ** 2)))
    # Zero-crossing rate — joint sounds are broadband/impulsive;
    # music/speech have lower or more structured ZCR
    zcr        = float(np.mean(librosa.feature.zero_crossing_rate(signal)[0]))
    max_conf   = float(max(proba))

    # 1. Too long — joint recordings are short bursts
    if duration > MAX_VALID_DURATION_SEC:
        return True, (f"Audio is {duration:.0f}s long. Knee joint recordings are "
                      "typically short bursts (< 60s). This may not be a joint signal.")

    # 2. Nearly silent
    if rms < MIN_RMS_ENERGY:
        return True, "The audio appears to be silent or nearly silent. Please upload a valid knee joint recording."

    # 3. Very high RMS — typical of music, speech, ambient noise
    if rms > MAX_RMS_ENERGY:
        return True, (f"Signal energy is very high (RMS={rms:.2f}). This looks like "
                      "music, loud speech, or ambient noise — not a knee joint recording.")

    # 4. High ZCR combined with long duration — characteristic of speech/music
    if zcr > 0.25 and duration > 5.0:
        return True, ("The signal has speech/music-like characteristics (high zero-crossing rate). "
                      "Please upload a knee joint acoustic recording.")

    # 5. Model is uncertain (close to 50/50 split)
    if max_conf < MIN_CONFIDENCE_FOR_VALID:
        return True, (f"The model is uncertain (confidence={max_conf*100:.0f}%). "
                      "The signal may not match the knee joint acoustic patterns the model was trained on.")

    return False, ""

# ──────────────────────────────────────────────────
# Feature extraction (mirrors Colab notebook exactly)
# ──────────────────────────────────────────────────
from feature_extractor import extract_features, SR


# ──────────────────────────────────────────────────
# Response Schema
# ──────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    label: str            # "Normal" or "Abnormal"
    confidence: float     # 0.0 – 1.0
    prob_normal: float
    prob_abnormal: float
    model_name: str
    model_accuracy: float
    threshold_used: float # Abnormal decision threshold
    is_uncertain: bool    # True if signal may not be a knee joint recording
    warning: str          # Human-readable reason when is_uncertain=True


# ──────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "model_name": meta.get("model_name", "Not loaded"),
        "test_accuracy": meta.get("test_accuracy", 0),
        "abnormal_threshold": ABNORMAL_THRESHOLD,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None,
            "abnormal_threshold": ABNORMAL_THRESHOLD}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload a .wav audio file and get a Normal / Abnormal prediction.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Place knee_model.pkl & knee_scaler.pkl "
                   "in the backend folder and restart the server."
        )

    # Accept only audio files
    allowed_types = {"audio/wav", "audio/wave", "audio/x-wav",
                     "audio/mp3", "audio/mpeg", "audio/ogg",
                     "audio/flac", "application/octet-stream"}
    if file.content_type not in allowed_types:
        # Still try to process — browser content types vary
        pass

    # Save uploaded file to a temp location
    suffix = os.path.splitext(file.filename or "signal.wav")[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Load audio
        signal, sr = librosa.load(tmp_path, sr=SR, mono=True)

        if len(signal) == 0:
            raise HTTPException(status_code=422,
                                detail="The audio file appears to be empty.")

        # Tile (repeat) very short signals so librosa features are computable.
        # Knee .mat-converted WAVs can be very short (< 100ms / few hundred samples).
        # IMPORTANT: we TILE (not zero-pad) to preserve spectral characteristics —
        # zero-padding would make every short clip look like silence (→ all "Normal").
        MIN_SAMPLES = 4096
        if len(signal) < MIN_SAMPLES:
            repeats = int(np.ceil(MIN_SAMPLES / len(signal)))
            signal = np.tile(signal, repeats)[:MIN_SAMPLES]

        # Extract features
        feats = extract_features(signal, sr=SR)
        feats_scaled = scaler.transform(feats.reshape(1, -1))

        # ── Custom threshold predict ──────────────────────────────────────
        proba = model.predict_proba(feats_scaled)[0]   # [P_normal, P_abnormal]
        prob_abnormal = float(proba[1])

        if prob_abnormal >= ABNORMAL_THRESHOLD:
            pred, label = 1, "Abnormal"
        else:
            pred, label = 0, "Normal"

        confidence = float(proba[pred])
        # ─────────────────────────────────────────────────────────────────

        # ── Signal validation — detect non-joint audio ───────────────────
        is_uncertain, warning = validate_signal(signal, SR, proba)
        # ─────────────────────────────────────────────────────────────────

        return PredictionResponse(
            label=label,
            confidence=round(confidence, 4),
            prob_normal=round(float(proba[0]), 4),
            prob_abnormal=round(prob_abnormal, 4),
            model_name=meta.get("model_name", "Unknown"),
            model_accuracy=round(float(meta.get("test_accuracy", 0)), 4),
            threshold_used=ABNORMAL_THRESHOLD,
            is_uncertain=is_uncertain,
            warning=warning,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        os.unlink(tmp_path)
