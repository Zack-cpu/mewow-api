"""
Model wrapper for FastAPI server.

This reuses the same logic as inference.py:
- load checkpoint best_model.pth
- preprocess audio the same way
- run CLAP + classifier
"""

import io
import os
from typing import Tuple, Dict

import torch
import torchaudio

import config
from model import create_model, load_clap_processor


# ----------------- Device & checkpoint -----------------

# Use config.DEVICE but fall back to CPU if CUDA not available
DEVICE = torch.device(
    "cuda" if (config.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
)

# Default checkpoint path: .../output/checkpoints/best_model.pth
CHECKPOINT_PATH = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")


# ----------------- Load model & processor once -----------------

print(f"[cat_model_server] Loading model from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model = create_model(num_classes=config.NUM_CLASSES)
# Your training code saves state as 'model_state_dict' :contentReference[oaicite:3]{index=3}
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

processor = load_clap_processor()  # ClapProcessor.from_pretrained(config.MODEL_NAME) 


# ----------------- Audio preprocessing (same as inference.py) -----------------

def _preprocess_waveform(
    waveform: torch.Tensor, sample_rate: int
) -> torch.Tensor:
    """
    Match CatMeowInference.preprocess_audio() but operate on a waveform tensor
    instead of loading from path.

    waveform: shape [channels, samples]
    returns: 1D tensor [samples]
    """
    # Convert to mono (mean over channels) 
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=config.SAMPLE_RATE,
        )
        waveform = resampler(waveform)

    # Trim or pad to MAX_AUDIO_LENGTH seconds
    max_samples = config.MAX_AUDIO_LENGTH * config.SAMPLE_RATE
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    elif waveform.shape[1] < max_samples:
        padding = max_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    # Return as 1D tensor [samples]
    return waveform.squeeze(0)


def _load_waveform_from_bytes(audio_bytes: bytes) -> Tuple[torch.Tensor, int]:
    """
    Load waveform + sample rate from raw bytes using torchaudio.

    NOTE: For easiest compatibility, have your iOS app upload WAV files.
    """
    file_like = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(file_like)
    return waveform, sample_rate


# ----------------- Public API for server -----------------

@torch.inference_mode()
def predict_from_bytes(audio_bytes: bytes) -> Tuple[str, float, Dict[str, float]]:
    """
    Main entry: given raw audio bytes, return:
    - predicted class name (string)
    - confidence (float)
    - dict of {class_name: probability}
    """

    # 1. Decode audio
    waveform, sample_rate = _load_waveform_from_bytes(audio_bytes)

    # 2. Preprocess like inference.py
    audio = _preprocess_waveform(waveform, sample_rate)  # 1D tensor

    # 3. Use ClapProcessor exactly as in inference.py :contentReference[oaicite:6]{index=6}
    inputs = processor(
        audios=audio.numpy(),               # 1D numpy array
        sampling_rate=config.SAMPLE_RATE,
        return_tensors="pt",
    )

    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 4. Run model
    logits = model(inputs)
    probabilities = torch.softmax(logits, dim=-1)[0]  # shape [NUM_CLASSES]

    # 5. Decode prediction using config.CLASS_NAMES :contentReference[oaicite:7]{index=7}
    pred_idx = int(torch.argmax(probabilities))
    predicted_class = config.CLASS_NAMES[pred_idx]
    confidence = float(probabilities[pred_idx])

    probs_dict = {
        config.CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(config.CLASS_NAMES))
    }

    return predicted_class, confidence, probs_dict
