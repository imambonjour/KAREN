#!/usr/bin/env python3
"""
MCP Server: Vision — Camera capture + describe / OCR / YOLO detection.
Tools: cek_sekitar, baca_teks, analisa_foto, deteksi_objek.

Architecture:
  - _CameraThread (daemon): satu-satunya yang membuka cv2.VideoCapture.
    Terus membaca frame → _latest_frame (shared, Lock).
    YOLO inference setiap YOLO_INFER_INTERVAL → _latest_detections (shared, Lock).
  - Semua tools membaca frame/detections dari shared state (tidak buka kamera baru).

Run standalone:  python -m karen_mcp.servers.vision_server
"""

import base64
import logging
import os
import sys
import threading
import time
from typing import Any

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import config  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

log = logging.getLogger("karen.vision_server")

# ---------------------------------------------------------------------------
# Gemma 4 endpoint
# ---------------------------------------------------------------------------
GEMMA4_CHAT_URL = config.GEMMA4_CHAT_URL

# ---------------------------------------------------------------------------
# COCO 80-class labels (yolov8 standard order)
# ---------------------------------------------------------------------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_frame_lock = threading.Lock()
_latest_frame: bytes | None = None          # JPEG bytes

_detections_lock = threading.Lock()
_latest_detections: list[dict] = []         # [{label, confidence, timestamp}]

_camera_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# YOLO ONNX helpers
# ---------------------------------------------------------------------------

def _load_yolo_session():
    """Load yolov8n ONNX model. Returns ort.InferenceSession or None."""
    model_path = config.YOLO_MODEL_PATH
    if not os.path.exists(model_path):
        log.warning(f"YOLO model not found at {model_path}. YOLO disabled.")
        return None
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        log.info(f"YOLO model loaded: {model_path}")
        return session
    except Exception as e:
        log.warning(f"Failed to load YOLO model: {e}. YOLO disabled.")
        return None


def _nms(boxes, scores, iou_threshold: float) -> list[int]:
    """Simple NMS. boxes: [[x1,y1,x2,y2], ...], scores: [float, ...]."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def _run_yolo(session, jpeg_bytes: bytes) -> list[dict]:
    """Run YOLO inference on jpeg_bytes, return list of detection dicts."""
    try:
        import cv2
        nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return []

        size = config.YOLO_INPUT_SIZE
        h0, w0 = frame.shape[:2]
        blob = cv2.resize(frame, (size, size))
        blob = blob[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, /255
        blob = blob.transpose(2, 0, 1)[np.newaxis]           # HWC→NCHW

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: blob})
        # YOLOv8 output shape: [1, 84, 8400] (cx,cy,w,h + 80 class scores)
        preds = outputs[0][0]  # (84, 8400)
        preds = preds.T        # (8400, 84)

        conf_thresh = config.YOLO_CONF_THRESHOLD
        classes_filter = set(config.YOLO_CLASSES_FILTER)

        boxes, scores, class_ids = [], [], []
        for row in preds:
            cx, cy, w, h = row[:4]
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < conf_thresh:
                continue
            label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
            if classes_filter and label not in classes_filter:
                continue
            # Convert cx,cy,w,h (normalized) → x1,y1,x2,y2 (pixel)
            x1 = (cx - w / 2) / size * w0
            y1 = (cy - h / 2) / size * h0
            x2 = (cx + w / 2) / size * w0
            y2 = (cy + h / 2) / size * h0
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []

        keep = _nms(boxes, scores, config.YOLO_IOU_THRESHOLD)
        ts = time.time()
        results = []
        for i in keep:
            cid = class_ids[i]
            label = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else str(cid)
            results.append({"label": label, "confidence": round(scores[i], 3), "timestamp": ts})
        return results
    except Exception as e:
        log.warning(f"YOLO inference error: {e}")
        return []


# ---------------------------------------------------------------------------
# Camera Thread
# ---------------------------------------------------------------------------

class _CameraThread(threading.Thread):
    """Daemon thread: membuka kamera sekali, loop baca frame, jalankan YOLO."""

    def __init__(self):
        super().__init__(name="karen-camera", daemon=True)
        self._stop_event = threading.Event()
        self._yolo_session = _load_yolo_session()

    def stop(self):
        self._stop_event.set()

    def run(self):
        global _latest_frame, _latest_detections
        import cv2

        idx = config.CAMERA_INDEX
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            log.error(f"_CameraThread: cannot open camera index={idx}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"_CameraThread started (camera index={idx}, capture {actual_w}x{actual_h})")
        last_yolo_time = 0.0

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log.warning("_CameraThread: failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue

                # Encode to JPEG for shared state
                _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpeg_bytes = jpeg_buf.tobytes()

                with _frame_lock:
                    _latest_frame = jpeg_bytes

                # YOLO inference (throttled)
                if self._yolo_session is not None:
                    now = time.monotonic()
                    if now - last_yolo_time >= config.YOLO_INFER_INTERVAL:
                        last_yolo_time = now
                        detections = _run_yolo(self._yolo_session, jpeg_bytes)
                        with _detections_lock:
                            _latest_detections = detections

                time.sleep(0.03)  # ~30fps cap
        finally:
            cap.release()
            log.info("_CameraThread stopped, camera released.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_latest_frame_b64() -> str:
    """Return base64-encoded JPEG of the latest frame, or raise RuntimeError."""
    with _frame_lock:
        frame = _latest_frame
    if frame is None:
        raise RuntimeError("Kamera belum siap atau tidak ada frame tersedia.")
    return base64.b64encode(frame).decode("ascii")


def _ask_gemma_vision(image_b64: str, prompt: str) -> str:
    """Send image + prompt to Gemma 4 vision, return text response."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(GEMMA4_CHAT_URL, json=payload, timeout=config.VISION_TIMEOUT)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# MCP Server & Tools
# ---------------------------------------------------------------------------

server = FastMCP(
    "karen-vision",
    instructions=(
        "Menyediakan kemampuan penglihatan (kamera) untuk mendeskripsikan lingkungan, "
        "membaca teks, menganalisis foto, dan mendeteksi objek secara real-time."
    ),
)


@server.tool()
def cek_sekitar() -> dict:
    """Mengambil foto dari kamera dan mendeskripsikan apa yang terlihat di sekitar.
    Gunakan ketika pengguna bertanya 'apa yang kamu lihat', 'lihat sekitar', 'ada apa di depan', dll."""
    try:
        image_b64 = _get_latest_frame_b64()
        description = _ask_gemma_vision(
            image_b64,
            "Deskripsikan apa yang kamu lihat di gambar ini dalam Bahasa Indonesia. "
            "Jelaskan objek, orang, dan suasana secara ringkas.",
        )
        return {"deskripsi": description}
    except Exception as e:
        log.exception("cek_sekitar failed")
        return {"error": str(e)}


@server.tool()
def baca_teks() -> dict:
    """Mengambil foto dari kamera dan membaca/mengekstrak teks yang terlihat (OCR).
    Gunakan ketika pengguna minta 'baca tulisan itu', 'apa yang tertulis', 'baca teks', dll."""
    try:
        image_b64 = _get_latest_frame_b64()
        extracted = _ask_gemma_vision(
            image_b64,
            "Baca dan tuliskan SEMUA teks yang terlihat di gambar ini secara verbatim. "
            "Jika ada beberapa baris, pisahkan dengan newline. "
            "Jika tidak ada teks, jawab 'Tidak ada teks yang terlihat.'",
        )
        return {"teks": extracted}
    except Exception as e:
        log.exception("baca_teks failed")
        return {"error": str(e)}


@server.tool()
def analisa_foto() -> dict:
    """Mengambil foto dari kamera dan meminta LLM memutuskan: OCR atau deskripsi suasana.
    Gunakan saat pengguna menekan tombol foto atau meminta analisis umum gambar."""
    try:
        image_b64 = _get_latest_frame_b64()
        hasil = _ask_gemma_vision(
            image_b64,
            "Lihat gambar ini dengan seksama. "
            "Jika terdapat teks/tulisan yang signifikan di dalamnya, "
            "bacakan dan transkripsikan seluruh isi teks secara verbatim dalam Bahasa Indonesia. "
            "Jika tidak ada teks yang signifikan, deskripsikan objek dan suasana yang terlihat secara ringkas. "
            "Jawab hanya dengan hasil analisis, tanpa pengantar.",
        )
        return {"hasil": hasil}
    except Exception as e:
        log.exception("analisa_foto failed")
        return {"error": str(e)}


@server.tool()
def simpan_frame() -> dict:
    """Menyimpan frame kamera terbaru ke direktori temp/ di workspace (untuk keperluan tes/debug).
    Mengembalikan path file yang tersimpan."""
    try:
        image_b64 = _get_latest_frame_b64()
        jpeg_bytes = base64.b64decode(image_b64)
        temp_dir = config.TEMP_DIR
        os.makedirs(temp_dir, exist_ok=True)
        filename = f"foto_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.abspath(os.path.join(temp_dir, filename))
        with open(filepath, "wb") as f:
            f.write(jpeg_bytes)
        log.info(f"simpan_frame: saved {filepath} ({len(jpeg_bytes)} bytes)")
        return {"path": filepath, "bytes": len(jpeg_bytes)}
    except Exception as e:
        log.exception("simpan_frame failed")
        return {"error": str(e)}


@server.tool()
def deteksi_objek() -> dict:
    """Mengembalikan daftar objek yang terdeteksi YOLO dari frame kamera terbaru (tanpa LLM).
    Gunakan saat pengguna bertanya objek apa yang ada di sekitar, atau saat proactive monitoring."""
    with _detections_lock:
        detections = list(_latest_detections)
    return {"detections": detections}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cam_thread = _CameraThread()
    cam_thread.start()
    log.info("Camera thread started. Warming up (2s)...")
    time.sleep(2)
    server.run(transport="stdio")
