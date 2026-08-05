import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Device / Hardware Configuration
# BISA DIUBAH VIA ENVIRONMENT VARIABLE ATAU FILE .env
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CAMERA_PREVIEW = os.environ.get("CAMERA_PREVIEW", "false").lower() in ("true", "1", "yes")
CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "720"))

# Manual exposure / shutter speed tuning (opsional, per-de&perangkat).
# None = jangan sentuh, kamera pakai nilai default-nya sendiri.
# Isi di .env bila ingin mengunci shutter lebih cepat (mengurangi blur):
#   CAMERA_MANUAL_EXPOSURE=true   # nonaktifkan auto-exposure (Manual Mode)
#   CAMERA_EXPOSURE_TIME=200      # kecil = shutter cepat; range bervariasi per webcam (cek via `v4l2-ctl -L`)
#   CAMERA_SHARPNESS=3            # naikkan ketajaman (0-10)
CAMERA_MANUAL_EXPOSURE = os.environ.get("CAMERA_MANUAL_EXPOSURE", "").lower() in ("true", "1", "yes")
CAMERA_EXPOSURE_TIME   = os.environ.get("CAMERA_EXPOSURE_TIME")
CAMERA_SHARPNESS       = os.environ.get("CAMERA_SHARPNESS")

# Audio Device Settings (PipeWire / ALSA / PulseAudio / SoundCard)
# AUDIO_INPUT_DEVICE: Nama/ID node target pw-record atau ALSA device (misal: "alsa_input.usb-xxx", "1", "default")
AUDIO_INPUT_DEVICE = os.environ.get("AUDIO_INPUT_DEVICE", None)

# AUDIO_OUTPUT_DEVICE: Nama/ID audio output (misal: "alsa_output.usb-xxx", "default")
AUDIO_OUTPUT_DEVICE = os.environ.get("AUDIO_OUTPUT_DEVICE", None)

# Cloud Model Configuration
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-3.5-flash")

# Local Models Configuration (bisa diubah langsung di bawah atau lewat .env)
LLAMA_SERVER_PATH = os.environ.get("LLAMA_SERVER_PATH", "./llama/llama-server")
GEMMA4_MODEL_PATH = os.environ.get("GEMMA4_MODEL_PATH", "models/Gemma4/gemma-4-E2B-it-UD-Q4_K_XL.gguf")
GEMMA4_MMPROJ_PATH = os.environ.get("GEMMA4_MMPROJ_PATH", "models/Gemma4/mmproj-F16.gguf")
GEMMA4_MTP_PATH = os.environ.get("GEMMA4_MTP_PATH", "models/Gemma4/mtp-gemma-4-E2B-it.gguf")
TTS_MODEL_PATH = os.environ.get("TTS_MODEL_PATH", "models/piper/id_ID-news_tts-medium.onnx")
VAD_MODEL_PATH = os.environ.get("VAD_MODEL_PATH", "models/silero_vad.onnx")
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "models/yolov26n.onnx")
GEMMA4_PORT = int(os.environ.get("GEMMA4_PORT", "8080"))
GEMMA4_CHAT_URL = f"http://localhost:{GEMMA4_PORT}/v1/chat/completions"
SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 500
VAD_SPEECH_PAD_MS = 30
VAD_MIN_SPEECH_MS = 250
MAX_HISTORY_TURNS = 10

# Vision LLM request timeout (seconds) — increase for slow SBC inference
VISION_TIMEOUT = int(os.environ.get("VISION_TIMEOUT", "180"))

# Temp / Debug Output
TEMP_DIR = "temp"  # directory for saved debug images (f-button captures)

# YOLO Object Detection
YOLO_MODEL_PATH          = "models/yolov26n.onnx"
YOLO_INPUT_SIZE          = 640
YOLO_CONF_THRESHOLD      = 0.5
YOLO_IOU_THRESHOLD       = 0.45
YOLO_INFER_INTERVAL      = 0.5        # seconds between YOLO inference runs
YOLO_CLASSES_FILTER: list[str] = []   # [] = all classes; fill to filter
YOLO_TARGET_CLASSES: list[str] = []   # [] = auto-announce off
YOLO_POLL_INTERVAL       = 1.0        # seconds between proactive polls in main.py
YOLO_ANNOUNCE_COOLDOWN   = 15.0       # seconds before same class announced again
