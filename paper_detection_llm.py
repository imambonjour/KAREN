#!/usr/bin/env python3
"""
paper_detection_llm.py
----------------------
Aplikasi deteksi kertas real-time menggunakan kamera + LLM Vision (Gemma 4 / llama-server).

Cara kerja parsing gambar ke llama-server instance:
1. Gambar hasil crop (atau file gambar) di-encode ke format JPEG / PNG.
2. Binary gambar diubah menjadi string Base64 (`base64.b64encode(jpeg_bytes)`).
3. Payload dikirim ke endpoint OpenAI-compatible `/v1/chat/completions` milik `llama-server`.
4. Format payload:
   messages: [
     {
       "role": "user",
       "content": [
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64_STRING>"}},
         {"type": "text", "text": "Bacakan seluruh teks yang ada pada gambar ini."}
       ]
     }
   ]
5. Hasil ekstraksi teks dari LLM lalu dibacakan oleh Piper TTS.
"""

import os
import sys
import time
import base64
import requests
import subprocess
import tempfile
import numpy as np
import cv2

# Import config jika tersedia
try:
    import config
    GEMMA4_CHAT_URL = getattr(config, "GEMMA4_CHAT_URL", "http://localhost:8080/v1/chat/completions")
    PIPER_MODEL_PATH = getattr(config, "TTS_MODEL_PATH", "models/piper/id_ID-news_tts-medium.onnx")
except ImportError:
    GEMMA4_CHAT_URL = "http://localhost:8080/v1/chat/completions"
    PIPER_MODEL_PATH = "models/piper/id_ID-news_tts-medium.onnx"


def order_points(pts):
    """Mengurutkan 4 titik sudut: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """Melakukan Perspective Transform untuk meng-crop dan meratakan posisi kertas."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB), 100)

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB), 100)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_paper_contour(frame, min_area_ratio=0.10):
    """Mendeteksi kontur segiempat kertas terbesar pada frame."""
    height, width = frame.shape[:2]
    frame_area = height * width

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 40, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        area = cv2.contourArea(c)
        if area < frame_area * min_area_ratio:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx, area

    return None, 0


def numpy_image_to_base64(image_np):
    """
    Mengubah gambar Numpy array (OpenCV BGR) menjadi string Base64 Data URI JPEG.
    """
    success, buffer = cv2.imencode(".jpg", image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        raise ValueError("Gagal meng-encode gambar ke format JPEG.")
    image_bytes = buffer.tobytes()
    return base64.b64encode(image_bytes).decode("ascii")


def file_to_base64(file_path):
    """
    Mengubah file gambar (PNG/JPG) di disk menjadi string Base64.
    """
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def parse_image_with_llm(image_input, prompt="Tuliskan dan bacakan seluruh teks yang tertulis pada dokumen ini secara lengkap dan jelas dalam Bahasa Indonesia.", server_url=GEMMA4_CHAT_URL):
    """
    Mengirim gambar (Numpy Array BGR atau Path File) ke llama-server instance (Gemma 4 E2B Vision).
    
    Format JSON OpenAI Chat Completions dengan multimodal image_url:
    {
       "messages": [
          {
             "role": "user",
             "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                {"type": "text", "text": "..."}
             ]
          }
       ]
    }
    """
    print(f"[LLM] Menyiapkan gambar dan mengirim permintaan ke llama-server ({server_url})...")
    
    if isinstance(image_input, str):
        image_b64 = file_to_base64(image_input)
    elif isinstance(image_input, np.ndarray):
        image_b64 = numpy_image_to_base64(image_input)
    else:
        raise TypeError("image_input harus berupa file path (str) atau OpenCV numpy array.")

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
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        response = requests.post(server_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        result_text = data["choices"][0]["message"]["content"].strip()
        
        # Format pembersihan tag <think> jika ada
        result_text = result_text.replace("<think>", "").replace("</think>", "").strip()
        return result_text

    except requests.exceptions.ConnectionError:
        print("\n[LLM Error] Gagal terhubung ke llama-server di localhost:8080!")
        print("Pastikan llama-server sudah berjalan (misal via `uv run main.py` atau script server).")
        return None
    except Exception as e:
        print(f"\n[LLM Error] Terjadi kesalahan saat memanggil LLM: {e}")
        return None


def speak_with_piper(text):
    """Mengubah teks hasil LLM ke audio suara menggunakan Piper TTS dan memutarnya."""
    if not text:
        print("[TTS] Tidak ada teks yang terdeteksi untuk dibaca.")
        return

    print(f"\n==================== HASIL LLM VISION ====================")
    print(text)
    print(f"==========================================================\n")

    model_path = PIPER_MODEL_PATH
    if not os.path.exists(model_path):
        alt_paths = ["models/piper/id_ID-news_tts-medium.onnx", "../models/piper/id_ID-news_tts-medium.onnx"]
        for p in alt_paths:
            if os.path.exists(p):
                model_path = p
                break

    if not os.path.exists(model_path):
        print(f"[TTS Error] Model Piper tidak ditemukan di '{model_path}'. Gagal memutar suara.")
        return

    print("[TTS] Menggenerasi audio menggunakan Piper TTS...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        temp_wav_path = tf.name

    try:
        piper_bin = os.path.join(sys.prefix, "bin", "piper")
        if not os.path.exists(piper_bin):
            piper_bin = "piper"

        cmd = [piper_bin, "--model", model_path, "--output_file", temp_wav_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        proc.communicate(input=text)

        if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
            print("[TTS] Memutar suara...")
            play_audio(temp_wav_path)
        else:
            print("[TTS Error] Penggenerasian audio WAV gagal.")
    except Exception as e:
        print(f"[TTS Error] Gagal memutar Piper TTS: {e}")
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def play_audio(wav_path):
    """Memutar audio menggunakan player sistem."""
    for player in ["aplay", "paplay", "ffplay"]:
        try:
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path] if player == "ffplay" else [player, wav_path]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                return
        except FileNotFoundError:
            continue
    print("[Audio Warning] Tidak ada audio player yang ditemukan di sistem.")


def main():
    print("[INIT] Membuka kamera untuk Paper Detection + LLM Vision...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat dibuka!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n=======================================================")
    print(" PAPER DETECTION + GEMMA4 E2B LLM VISION (LLAMA-SERVER)")
    print("=======================================================")
    print(" Petunjuk:")
    print(" - Arahkan kertas dokumen ke depan kamera.")
    print(" - Tahan posisi kertas selama 1.5 detik untuk Auto Capture.")
    print(" - Tekan tombol 'SPASI' untuk Capture manual.")
    print(" - Tekan tombol 'Q' atau 'ESC' untuk keluar.")
    print("=======================================================\n")

    stable_start_time = None
    STABLE_DURATION = 1.5
    COOLDOWN_DURATION = 4.0
    last_capture_time = 0

    prev_center = None
    SHIFT_THRESHOLD = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame kamera.")
            break

        display_frame = frame.copy()
        paper_cnt, area = detect_paper_contour(frame)

        current_time = time.time()
        in_cooldown = (current_time - last_capture_time) < COOLDOWN_DURATION
        trigger_capture = False

        if paper_cnt is not None:
            M = cv2.moments(paper_cnt)
            curr_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else None

            color = (0, 165, 255) if in_cooldown else (0, 255, 0)
            cv2.drawContours(display_frame, [paper_cnt], -1, color, 3)

            for pt in paper_cnt.reshape(4, 2):
                cv2.circle(display_frame, tuple(pt), 6, (0, 0, 255), -1)

            if not in_cooldown:
                if prev_center is not None and curr_center is not None:
                    dist = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                    if dist < SHIFT_THRESHOLD:
                        if stable_start_time is None:
                            stable_start_time = current_time
                        
                        elapsed = current_time - stable_start_time
                        progress = min(1.0, elapsed / STABLE_DURATION)

                        bar_w, bar_h = 300, 20
                        bx, by = 30, 80
                        cv2.rectangle(display_frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
                        cv2.rectangle(display_frame, (bx, by), (bx + int(bar_w * progress), by + bar_h), (0, 255, 0), -1)
                        cv2.putText(display_frame, f"Holding paper... {int(progress * 100)}%", (bx, by - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        if elapsed >= STABLE_DURATION:
                            trigger_capture = True
                    else:
                        stable_start_time = None
                else:
                    stable_start_time = None
                prev_center = curr_center
            else:
                stable_start_time = None
                prev_center = None
                cooldown_left = COOLDOWN_DURATION - (current_time - last_capture_time)
                cv2.putText(display_frame, f"Selesai LLM Vision. Cooldown ({cooldown_left:.1f}s)...", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.putText(display_frame, "KERTAS TERDETEKSI", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            stable_start_time = None
            prev_center = None
            cv2.putText(display_frame, "Mencari kertas...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(display_frame, "[SPASI] Capture Manual | [Q/ESC] Keluar", (30, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Paper Detection & LLM Vision Reader", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("[EXIT] Menutup aplikasi.")
            break
        elif key == 32:
            if paper_cnt is not None:
                trigger_capture = True
            else:
                h, w = frame.shape[:2]
                paper_cnt = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.int32)
                trigger_capture = True

        if trigger_capture and paper_cnt is not None:
            last_capture_time = time.time()
            stable_start_time = None

            print("\n[CAPTURE] Kertas ditangkap! Mengirim ke LLM Vision...")
            warped = four_point_transform(frame, paper_cnt.reshape(4, 2))

            cv2.imshow("Hasil Crop Dokumen", warped)
            cv2.waitKey(1)

            # Kirim gambar crop ke llama-server instance (Gemma 4 Vision)
            llm_text = parse_image_with_llm(
                warped,
                prompt="Bacakan dan transkripsikan seluruh isi teks pada gambar dokumen ini dengan jelas dan tepat."
            )

            if llm_text:
                speak_with_piper(llm_text)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
