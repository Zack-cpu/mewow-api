"""
Model wrapper for FastAPI server.

- Ensures best_model.pth is present by downloading from GitHub release if needed.
- Loads the CLAP-based classifier and exposes predict_from_bytes() for server.py.
"""

import io
import os
from typing import Tuple, Dict

import requests
import torch
import torchaudio

import config
from model import create_model, load_clap_processor

# ----------------- Paths & device -----------------

DEVICE = torch.device(
    "cuda" if (getattr(config, "DEVICE", "cpu") == "cuda" and torch.cuda.is_available())
    else "cpu"
)

# Where the checkpoint should live locally (same as your training code)
CHECKPOINT_PATH = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")

# GitHub release URL you gave me
MODEL_DOWNLOAD_URL = (
    "https://github.com/Zack-cpu/mewow-api/releases/download/v1.0.0/best_model.pth"
)


def ensure_checkpoint() -> None:
    """
    Make sure best_model.pth exists locally.
    If not, download it from the GitHub release URL.
    """
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"[cat_model_server] Found checkpoint at {CHECKPOINT_PATH}")
        return

    print(
        f"[cat_model_server] No checkpoint at {CHECKPOINT_PATH}, "
        f"downloading from {MODEL_DOWNLOAD_URL}..."
    )

    resp = requests.get(MODEL_DOWNLOAD_URL, stream=True)
    resp.raise_for_status()

    total_bytes = 0
    with open(CHECKPOINT_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
            if not chunk:
                continue
            f.write(chunk)
            total_bytes += len(chunk)

    print(
        f"[cat_model_server] Downloaded checkpoint "
        f"({total_bytes / (1024 * 1024):.1f} MB) to {CHECKPOINT_PATH}"
    )


# ----------------- Load model & processor -----------------

ensure_checkpoint()

print(f"[cat_model_server] Loading model from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model = create_model(num_classes=config.NUM_CLASSES)

# Your training code might have saved a dict with model_state_dict
state = checkpoint
if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
elif isinstance(state, dict) and "state_dict" in state:
    model.load_state_dict(state["state_dict"])
else:
    # Fallback: assume it's directly a state_dict
    model.load_state_dict(state)

model.to(DEVICE)
model.eval()

processor = load_clap_processor()  # ClapProcessor.from_pretrained(config.MODEL_NAME)


# ----------------- Audio preprocessing (matches inference.py) -----------------

SAMPLE_RATE = config.SAMPLE_RATE
MAX_AUDIO_SECONDS = config.MAX_AUDIO_LENGTH
CLASS_NAMES = config.CLASS_NAMES


def _preprocess_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Match CatMeowInference.preprocess_audio(), but operate on a waveform tensor.

    waveform: [channels, samples]
    returns: 1D tensor [samples] at config.SAMPLE_RATE, padded/trimmed to MAX_AUDIO_SECONDS
    """
    # Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=SAMPLE_RATE,
        )
        waveform = resampler(waveform)

    max_samples = int(MAX_AUDIO_SECONDS * SAMPLE_RATE)

    # Trim or pad
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    elif waveform.shape[1] < max_samples:
        padding = max_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform.squeeze(0)  # [samples]


def _load_waveform_from_bytes(audio_bytes: bytes) -> Tuple[torch.Tensor, int]:
    """
    Load waveform + sample rate from raw bytes using torchaudio.
    Works with wav/mp3/etc as long as torchaudio has the backend.
    """
    file_like = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(file_like)
    return waveform, sample_rate


# ----------------- Public prediction API -----------------

@torch.inference_mode()
def predict_from_bytes(audio_bytes: bytes) -> Tuple[str, float, Dict[str, float]]:
    """
    Given raw audio bytes, return:
    - predicted class name
    - confidence for that class
    - dict of {class_name: probability}
    """

    # 1. Decode audio
    waveform, sample_rate = _load_waveform_from_bytes(audio_bytes)

    # 2. Preprocess to fixed-length mono at SAMPLE_RATE
    audio = _preprocess_waveform(waveform, sample_rate)  # 1D tensor

    # 3. Use CLAP processor (same as inference.py)
    inputs = processor(
        audios=audio.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 4. Run model
    logits = model(inputs)
    probabilities = torch.softmax(logits, dim=-1)[0]  # [NUM_CLASSES]

    # 5. Decode prediction
    pred_idx = int(torch.argmax(probabilities))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(probabilities[pred_idx])

    probs_dict = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, probs_dict
