import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
(DATA_DIR / "faces").mkdir(exist_ok=True)
(DATA_DIR / "logs").mkdir(exist_ok=True)

# Configuration
BACKEND = "cpu"  # "cpu" (ONNX) or "npu" (RKNN)

# Model Paths
DETECTOR_MODEL_CPU = MODELS_DIR / "yolov8n-face.onnx"
DETECTOR_MODEL_NPU = MODELS_DIR / "yolov8n-face.rknn"

RECOGNIZER_MODEL_CPU = MODELS_DIR / "w600k_r50.onnx"
RECOGNIZER_MODEL_NPU = MODELS_DIR / "arcface.rknn"

# YOLO General Object Detection (for overlay)
YOLO_MODEL_PATH = BASE_DIR / "yolo26n.pt"
YOLO_CONF_THRESHOLD = 0.25

# Detection Settings
DETECTOR_CONF_THRESHOLD = 0.5
DETECTOR_NMS_THRESHOLD = 0.45

# Face Recognition Settings
RECOGNIZER_THRESHOLD = 0.45  # Cosine similarity threshold for w600k_r50 (more discriminative than generic ArcFace)

# Database
DB_PATH = DATA_DIR / "database.db"

# Camera settings
CAMERA_INDEX = 0  # Can be an integer or path to a video file
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
