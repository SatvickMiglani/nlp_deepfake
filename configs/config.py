"""
Central configuration for the Multimodal Deepfake Detection project.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"D:\deepfake"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw_videos")
SCORE_CACHE_DIR = os.path.join(PROJECT_ROOT, "outputs", "scores")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

for d in [DATA_DIR, RAW_VIDEO_DIR, SCORE_CACHE_DIR, RESULTS_DIR, FIGURES_DIR, SPLITS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Using HuggingFace datasets API for FakeAVCeleb
HF_DATASET_NAME = "DASH-Lab/FakeAVCeleb"  # Primary candidate
DATASET_CACHE_DIR = os.path.join(DATA_DIR, "hf_cache")

# FakeAVCeleb label categories
LABEL_MAP = {
    "real": 0,           # Real audio + Real video
    "fake_audio": 1,     # Fake audio + Real video  (audio-only deepfake)
    "fake_video": 2,     # Real audio + Fake video  (video-only deepfake)
    "fake_both": 3,      # Fake audio + Fake video  (both manipulated)
}
# Binary label for main detection task
BINARY_LABEL_MAP = {
    "real": 0,
    "fake": 1,
}

# ─── Segment / Window ────────────────────────────────────────────────────────
SEGMENT_DURATION_SEC = 2.5     # Window length in seconds (tunable on val)
SEGMENT_HOP_SEC = 1.0          # Hop between windows
SAMPLE_RATE = 16000             # Audio sample rate
FPS_TARGET = 10                 # Frames per second for visual branch

# ─── Models ───────────────────────────────────────────────────────────────────
# Audio forensic branch
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"  # Feature extractor
AUDIO_CLASSIFIER_HIDDEN = 256

# Visual forensic branch  (MediaPipe + frozen CNN)
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
VISUAL_MODEL_NAME = "google/efficientnet-b0"  # Pretrained visual encoder
VISUAL_CLASSIFIER_HIDDEN = 256

# Emotion branch
SPEECH_EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# Facial emotion via MediaPipe face mesh + simple classifier
FACIAL_EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS_FUSION = 50
EARLY_STOP_PATIENCE = 10
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ─── Ablation configs ────────────────────────────────────────────────────────
ABLATION_CONFIGS = {
    "A":     {"audio": True,  "visual": False, "emotion": False},
    "V":     {"audio": False, "visual": True,  "emotion": False},
    "E":     {"audio": False, "visual": False, "emotion": True},
    "A+V":   {"audio": True,  "visual": True,  "emotion": False},
    "A+E":   {"audio": True,  "visual": False, "emotion": True},
    "V+E":   {"audio": False, "visual": True,  "emotion": True},
    "A+V+E": {"audio": True,  "visual": True,  "emotion": True},
}

# ─── Temporal aggregation strategies ─────────────────────────────────────────
TEMPORAL_AGG_STRATEGIES = ["mean", "max", "top_k"]
TOP_K = 3  # for top-k aggregation

# ─── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
