"""
Anime Eye Tracker
==================
Mata anime "hidup" yang pupil & arah pandangnya mengikuti posisi tangan
(ujung telunjuk) yang terdeteksi dari webcam, menggunakan OpenCV untuk
rendering mata dan MediaPipe Hands untuk tracking tangan.

Install dulu:
    pip install opencv-python mediapipe numpy

Jalankan:
    python anime_eye_tracker.py

Kontrol:
    q  -> keluar
    b  -> kedip manual (blink)
"""

import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import os
import threading
import time
import random

# ------------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 540          # ukuran window kamera (untuk preview kecil)
EYE_CANVAS_W, EYE_CANVAS_H = 900, 600  # ukuran window mata

NUM_EYES = 2
EYE_RADIUS = 140          # radius sclera
IRIS_RADIUS = 60          # radius iris
PUPIL_RADIUS = 26         # radius pupil
MAX_PUPIL_OFFSET = EYE_RADIUS - IRIS_RADIUS - 6  # batas gerak iris di dalam sclera

IRIS_COLOR = (60, 130, 210)   # BGR - warna dasar iris (cokelat kemerahan bisa diganti)
SCLERA_TINT = (235, 235, 245)  # BGR - putih agak kebiruan

SMOOTH = 0.18   # faktor smoothing gerak pupil (0-1, makin kecil makin halus/lambat)
BLINK_INTERVAL_RANGE = (2.5, 6.0)  # detik, kedip otomatis acak
BLINK_DURATION = 0.15

# ------------------------------------------------------------------
# MEDIAPIPE HANDS — shared state & callback untuk LIVE_STREAM mode
# ------------------------------------------------------------------
_INDEX_FINGER_TIP = 8
_lock = threading.Lock()
_hand_point_raw = None


def _hand_callback(result, image, timestamp_ms):
    global _hand_point_raw
    if result.hand_landmarks:
        lm = result.hand_landmarks[0][_INDEX_FINGER_TIP]
        with _lock:
            _hand_point_raw = (lm.x, lm.y)
    else:
        with _lock:
            _hand_point_raw = None


# ------------------------------------------------------------------
# RENDERING MATA REALISTIS
# ------------------------------------------------------------------
def draw_radial_gradient(canvas, center, radius, color_inner, color_outer):
    """Gambar lingkaran dengan gradient radial dari color_inner (tengah) ke color_outer (tepi)."""
    x0, y0 = center
    y_idx, x_idx = np.ogrid[max(0, y0 - radius):y0 + radius, max(0, x0 - radius):x0 + radius]
    dist = np.sqrt((x_idx - x0) ** 2 + (y_idx - y0) ** 2)
    mask = dist <= radius
    t = np.clip(dist / radius, 0, 1)

    region = canvas[max(0, y0 - radius):y0 + radius, max(0, x0 - radius):x0 + radius]
    for c in range(3):
        grad = color_inner[c] * (1 - t) + color_outer[c] * t
        region[..., c] = np.where(mask, grad, region[..., c])


def draw_iris_texture(canvas, center, radius, base_color, n_lines=40):
    """Tambahkan garis-garis serat radial tipis di iris agar terlihat tekstur."""
    x0, y0 = center
    for i in range(n_lines):
        angle = (2 * np.pi / n_lines) * i + random.uniform(-0.02, 0.02)
        r1 = radius * 0.25
        r2 = radius * random.uniform(0.85, 1.0)
        x1 = int(x0 + r1 * np.cos(angle))
        y1 = int(y0 + r1 * np.sin(angle))
        x2 = int(x0 + r2 * np.cos(angle))
        y2 = int(y0 + r2 * np.sin(angle))
        shade = random.uniform(0.6, 1.3)
        col = tuple(int(min(255, c * shade)) for c in base_color)
        cv2.line(canvas, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)


def draw_eye(canvas, center, look_offset, blink_amount):
    """
    Gambar satu mata realistis di 'canvas' pada posisi 'center'.
    look_offset: (dx, dy) offset pupil relatif dari center, sudah dibatasi.
    blink_amount: 0 = mata terbuka penuh, 1 = mata tertutup penuh.
    """
    cx, cy = center

    # --- Sclera (radial gradient putih -> abu muda di tepi) ---
    draw_radial_gradient(canvas, (cx, cy), EYE_RADIUS,
                          (250, 250, 255), SCLERA_TINT)

    # sedikit shading atas (efek kelopak/bayangan)
    overlay = canvas.copy()
    cv2.ellipse(overlay, (cx, cy), (EYE_RADIUS, int(EYE_RADIUS * 0.9)),
                0, 180, 360, (200, 200, 210), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

    # --- Posisi iris mengikuti arah pandang ---
    ix, iy = cx + look_offset[0], cy + look_offset[1]

    # iris gradient (gelap di tepi, terang di tengah)
    inner = tuple(min(255, int(c * 1.4)) for c in IRIS_COLOR)
    outer = tuple(int(c * 0.35) for c in IRIS_COLOR)
    draw_radial_gradient(canvas, (int(ix), int(iy)), IRIS_RADIUS, inner, outer)
    draw_iris_texture(canvas, (int(ix), int(iy)), IRIS_RADIUS, IRIS_COLOR)

    # cincin tepi iris (limbal ring) biar tegas kayak anime
    cv2.circle(canvas, (int(ix), int(iy)), IRIS_RADIUS, (20, 20, 20), 2, cv2.LINE_AA)

    # --- Pupil ---
    cv2.circle(canvas, (int(ix), int(iy)), PUPIL_RADIUS, (10, 10, 10), -1, cv2.LINE_AA)

    # --- Highlight / glare khas anime (2 titik: besar + kecil) ---
    hl1 = (int(ix - IRIS_RADIUS * 0.35), int(iy - IRIS_RADIUS * 0.4))
    hl2 = (int(ix + IRIS_RADIUS * 0.15), int(iy + IRIS_RADIUS * 0.35))
    cv2.circle(canvas, hl1, 14, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, hl2, 6, (255, 255, 255), -1, cv2.LINE_AA)

    # --- Outline sclera (garis luar mata, khas anime tegas) ---
    cv2.circle(canvas, (cx, cy), EYE_RADIUS, (15, 15, 15), 4, cv2.LINE_AA)

    # --- Blink: tutup pakai dua kelopak (atas & bawah) mengikuti blink_amount ---
    if blink_amount > 0.01:
        skin = (225, 200, 190)  # warna kelopak (bisa disesuaikan skin tone)
        x1, y1 = cx - EYE_RADIUS - 2, cy - EYE_RADIUS - 2
        x2, y2 = cx + EYE_RADIUS + 2, cy + EYE_RADIUS + 2
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(canvas.shape[1], x2)
        y2 = min(canvas.shape[0], y2)

        cover_h_top = int((EYE_RADIUS * 2) * (blink_amount / 2))
        cover_h_bot = int((EYE_RADIUS * 2) * (blink_amount / 2))

        if cover_h_top > 0:
            cv2.rectangle(canvas, (x1, y1), (x2, y1 + cover_h_top), skin, -1)
            cv2.line(canvas, (x1, y1 + cover_h_top), (x2, y1 + cover_h_top), (60, 40, 40), 3, cv2.LINE_AA)
        if cover_h_bot > 0:
            cv2.rectangle(canvas, (x1, y2 - cover_h_bot), (x2, y2), skin, -1)
            cv2.line(canvas, (x1, y2 - cover_h_bot), (x2, y2 - cover_h_bot), (60, 40, 40), 3, cv2.LINE_AA)


# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    eye_centers = [
        (EYE_CANVAS_W // 2 - 220, EYE_CANVAS_H // 2),
        (EYE_CANVAS_W // 2 + 220, EYE_CANVAS_H // 2),
    ]

    model_path = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        result_callback=_hand_callback,
    )
    hand_landmarker = HandLandmarker.create_from_options(options)

    current_offset = np.array([0.0, 0.0])
    target_offset = np.array([0.0, 0.0])

    next_blink_time = time.time() + random.uniform(*BLINK_INTERVAL_RANGE)
    blink_start = None
    manual_blink = False

    print("Tekan 'q' untuk keluar, 'b' untuk kedip manual.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca webcam.")
            break
        frame = cv2.flip(frame, 1)  # mirror biar natural
        frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))

        now = time.time()
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        mp_img = Image(ImageFormat.SRGB, rgb)
        hand_landmarker.detect_async(mp_img, int(now * 1000))

        with _lock:
            hand_point = _hand_point_raw

        if hand_point is not None:
            hx, hy = hand_point  # 0..1
            dx = (hx - 0.5) * 2
            dy = (hy - 0.5) * 2
            dx = np.clip(dx, -1, 1)
            dy = np.clip(dy, -1, 1)
            target_offset = np.array([dx, dy]) * MAX_PUPIL_OFFSET

            px, py = int(hx * FRAME_W), int(hy * FRAME_H)
            cv2.circle(frame_small, (px, py), 10, (0, 255, 0), -1)

        # smoothing gerak pupil
        current_offset += (target_offset - current_offset) * SMOOTH

        # --- blink logic ---
        if blink_start is None and (now >= next_blink_time or manual_blink):
            blink_start = now
            manual_blink = False
            next_blink_time = now + random.uniform(*BLINK_INTERVAL_RANGE)

        blink_amount = 0.0
        if blink_start is not None:
            elapsed = now - blink_start
            half = BLINK_DURATION / 2
            if elapsed < half:
                blink_amount = elapsed / half
            elif elapsed < BLINK_DURATION:
                blink_amount = 1 - (elapsed - half) / half
            else:
                blink_start = None
                blink_amount = 0.0
            blink_amount = np.clip(blink_amount, 0, 1)

        # --- render mata ---
        canvas = np.full((EYE_CANVAS_H, EYE_CANVAS_W, 3), (30, 25, 25), dtype=np.uint8)
        for c in eye_centers:
            draw_eye(canvas, c, current_offset, blink_amount)

        cv2.imshow("Anime Eyes", canvas)
        cv2.imshow("Webcam (hand tracking)", frame_small)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            manual_blink = True

    cap.release()
    hand_landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
