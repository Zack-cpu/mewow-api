"""
Configuration file for Cat Meow Classifier
"""
import os

# Model configuration
MODEL_NAME = "laion/clap-htsat-fused"
NUM_CLASSES = 10
MAX_AUDIO_LENGTH = 5  # seconds (not used - keeping variable length audio)

# Class names for the 10 cat meow types
CLASS_NAMES = [
    "Angry",
    "Defence",
    "Fighting",
    "Happy",
    "HuntingMind",
    "Mating",
    "MotherCall",
    "Paining",
    "Resting",
    "Warning"
]

# Training configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-5"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "500"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))

# Data paths
DATA_DIR = os.getenv("DATA_DIR", "./data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Output paths
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "checkpoints")
COREML_MODEL_PATH = os.path.join(OUTPUT_DIR, "CatMeowClassifier.mlmodel")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Audio processing
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "48000"))
N_MELS = int(os.getenv("N_MELS", "128"))

# Device
DEVICE = os.getenv("DEVICE", "cuda" if os.path.exists("/proc/driver/nvidia/version") else "cpu")

