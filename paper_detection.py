#!/usr/bin/env python3
"""
paper_detection.py
------------------
Aplikasi deteksi kertas real-time menggunakan kamera.
Fitur:
1. Menyalakan kamera terus-menerus dengan visualisasi overlay kontur kertas.
2. Otomatis capture saat kertas terdeteksi stabil (atau manual dengan tombol SPASI).
3. Perspective transform (crop & ratakan posisi kertas).
4. Penyesuaian Kecerahan & Kontras (Brightness & Contrast Adjustment).
5. OCR menggunakan Tesseract (Bahasa Indonesia + Inggris).
6. Text-to-Speech (TTS) menggunakan Piper.
"""

import os
import sys
import time
import subprocess
import tempfile
import numpy as np
import cv2
import pytesseract

# Import config jika tersedia
try:
    import config
    PIPER_MODEL_PATH = getattr(config, "TTS_MODEL_PATH", "models/piper/id_ID-news_tts-medium.onnx")
except ImportError:
    PIPER_MODEL_PATH = "models/piper/id_ID-news_tts-medium.onnx"


def order_points(pts):
    """
    Mengurutkan 4 titik sudut: [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)

    # Top-left memiliki jumlah (x + y) terkecil, bottom-right terbesar
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right memiliki selisih (y - x) terkecil, bottom-left terbesar
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    Melakukan Perspective Transform untuk meng-crop dan meratakan gambar sesuai kertas.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Hitung lebar maksimum gambar baru
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Hitung tinggi maksimum gambar baru
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Pastikan ukuran w & h valid
    maxWidth = max(maxWidth, 100)
    maxHeight = max(maxHeight, 100)

    # Matriks transformasi tujuan (top-down view)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Hitung matriks transformasi & terapkan warpPerspective
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def detect_paper_contour(frame, min_area_ratio=0.12):
    """
    Mendeteksi kontur kertas (segiempat terbesar) pada frame.
    Returns: (screen_cnt, area) atau (None, 0) jika tidak ditemukan.
    """
    height, width = frame.shape[:2]
    frame_area = height * width

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Dilasi untuk menutup celah pada garis kontur
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    # Cari kontur
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screen_cnt = None
    max_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < frame_area * min_area_ratio:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Kertas harus memiliki 4 sudut (quadrilateral) dan cembung (convex)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            screen_cnt = approx
            max_area = area
            break

    return screen_cnt, max_area


def adjust_brightness_contrast(image, alpha=1.05, beta=5):
    """
    Menyesuaikan kecerahan (beta) dan kontras (alpha) gambar secara wajar (tidak over-contrast).
    formula: g(x) = alpha * f(x) + beta
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def enhance_for_ocr(image):
    """
    Preprocessing lembut untuk mengoptimalkan hasil Tesseract OCR tanpa over-contrast:
    1. Penyesuaian kecerahan & kontras lembut (alpha=1.05, beta=5)
    2. Ubah ke Grayscale
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization) agar pencahayaan merata
    """
    # 1. Penyesuaian Kecerahan & Kontras Lembut
    adj = adjust_brightness_contrast(image, alpha=1.05, beta=5)

    # 2. Ubah ke Grayscale
    gray = cv2.cvtColor(adj, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE untuk perataan kontras lokal secara alami
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 4. Otsu Thresholding lembut (opsional untuk binarisasi jika diperlukan)
    _, binary = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return adj, enhanced_gray


def run_ocr(image):
    """
    Menjalankan Tesseract OCR pada gambar yang telah di-preprocess.
    """
    print("[OCR] Menjalankan Tesseract OCR...")
    config_ocr = "--psm 3"
    try:
        text = pytesseract.image_to_string(image, lang="ind+eng", config=config_ocr)
    except Exception as e:
        print(f"[OCR Warning] Gagal dengan lang 'ind+eng', mencoba 'eng': {e}")
        text = pytesseract.image_to_string(image, lang="eng", config=config_ocr)

    text = text.strip()
    return text


def speak_with_piper(text):
    """
    Mengubah teks menjadi suara menggunakan Piper TTS dan memutarnya.
    """
    if not text:
        print("[TTS] Tidak ada teks yang terdeteksi untuk dibaca.")
        return

    print(f"\n==================== HASIL OCR ====================")
    print(text)
    print(f"===================================================\n")

    # Cari file model Piper
    model_path = PIPER_MODEL_PATH
    if not os.path.exists(model_path):
        # Fallback pencarian model di direktori projek
        alt_paths = [
            "models/piper/id_ID-news_tts-medium.onnx",
            "../models/piper/id_ID-news_tts-medium.onnx"
        ]
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
        # Panggil executable piper via uv / python / CLI
        piper_bin = os.path.join(sys.prefix, "bin", "piper")
        if not os.path.exists(piper_bin):
            piper_bin = "piper"

        cmd = [
            piper_bin,
            "--model", model_path,
            "--output_file", temp_wav_path
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        proc.communicate(input=text)

        if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
            print("[TTS] Memutar suara hasil bacaan OCR...")
            play_audio(temp_wav_path)
        else:
            print("[TTS Error] Penggenerasian file WAV Piper gagal.")

    except Exception as e:
        print(f"[TTS Error] Gagal menjalankan Piper: {e}")
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def play_audio(wav_path):
    """
    Memutar audio WAV menggunakan player bawaan sistem (aplay, paplay, ffplay).
    """
    players = ["aplay", "paplay", "ffplay"]
    played = False
    for player in players:
        try:
            if player == "ffplay":
                cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path]
            else:
                cmd = [player, wav_path]
            
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                played = True
                break
        except FileNotFoundError:
            continue

    if not played:
        print("[Audio Warning] Tidak menemukan audio player (aplay/paplay/ffplay) di sistem.")


def main():
    print("[INIT] Membuka kamera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat diakses! Pastikan webcam terhubung.")
        return

    # Atur resolusi kamera jika memungkinkan
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n=======================================================")
    print(" PAPER DETECTION & AUTOMATIC OCR + PIPER TTS BACA TEKS")
    print("=======================================================")
    print(" Petunjuk:")
    print(" - Arahkan kertas dokumen ke depan kamera.")
    print(" - Tahan posisi kertas dengan stabil selama 1.5 detik untuk Auto Capture.")
    print(" - Tekan tombol 'SPASI' untuk Capture manual kapan saja.")
    print(" - Tekan tombol 'Q' atau 'ESC' untuk keluar.")
    print("=======================================================\n")

    stable_start_time = None
    STABLE_DURATION = 1.5  # Waktu tahan (detik) sebelum auto-capture
    COOLDOWN_DURATION = 4.0 # Cooldown setelah capture (detik)
    last_capture_time = 0

    prev_center = None
    SHIFT_THRESHOLD = 30 # Toleransi pergeseran pusat kertas (pixel)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal mengambil frame dari kamera.")
            break

        display_frame = frame.copy()
        paper_cnt, area = detect_paper_contour(frame)

        current_time = time.time()
        in_cooldown = (current_time - last_capture_time) < COOLDOWN_DURATION
        trigger_capture = False

        if paper_cnt is not None:
            # Hitung titik pusat kontur kertas
            M = cv2.moments(paper_cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                curr_center = (cx, cy)
            else:
                curr_center = None

            # Gambar overlay kontur di layar
            color = (0, 165, 255) if in_cooldown else (0, 255, 0) # Orange jika cooldown, Hijau jika siap
            cv2.drawContours(display_frame, [paper_cnt], -1, color, 3)

            # Gambar titik sudut
            for pt in paper_cnt.reshape(4, 2):
                cv2.circle(display_frame, tuple(pt), 6, (0, 0, 255), -1)

            if not in_cooldown:
                # Cek kestabilan posisi kertas
                if prev_center is not None and curr_center is not None:
                    dist = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                    if dist < SHIFT_THRESHOLD:
                        if stable_start_time is None:
                            stable_start_time = current_time
                        
                        elapsed = current_time - stable_start_time
                        progress = min(1.0, elapsed / STABLE_DURATION)

                        # Tampilkan Progress Bar Stabil
                        bar_w = 300
                        bar_h = 20
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
                # Tampilkan teks Cooldown
                stable_start_time = None
                prev_center = None
                cooldown_left = COOLDOWN_DURATION - (current_time - last_capture_time)
                cv2.putText(display_frame, f"Selesai OCR. Cooldown ({cooldown_left:.1f}s)...", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.putText(display_frame, "KERTAS TERDETEKSI", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            stable_start_time = None
            prev_center = None
            cv2.putText(display_frame, "Mencari kertas...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Tampilkan instruksi di bagian bawah
        cv2.putText(display_frame, "[SPASI] Capture Manual | [Q/ESC] Keluar", (30, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Paper Detection & Reader", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): # ESC atau Q
            print("[EXIT] Menutup aplikasi.")
            break
        elif key == 32: # SPASI (Manual capture)
            if paper_cnt is not None:
                print("[MANUAL] Tombol SPASI ditekan, melakukan capture...")
                trigger_capture = True
            else:
                print("[MANUAL] Kertas belum terdeteksi! Gunakan seluruh frame sebagai dokumen...")
                # Fallback ke seluruh frame jika tidak ada kontur segiempat terdeteksi
                h, w = frame.shape[:2]
                paper_cnt = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.int32)
                trigger_capture = True

        if trigger_capture and paper_cnt is not None:
            last_capture_time = time.time()
            stable_start_time = None

            print("\n[CAPTURE] Kertas ditangkap! Memproses gambar...")
            
            # Flash efek visual di layar
            flash_frame = display_frame.copy()
            cv2.rectangle(flash_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (255, 255, 255), -1)
            cv2.imshow("Paper Detection & Reader", flash_frame)
            cv2.waitKey(100)

            # 1. Perspective Transform (Crop & Flatten)
            warped = four_point_transform(frame, paper_cnt.reshape(4, 2))

            # 2. Penyesuaian Brightness & Contrast Lembut (CLAHE)
            adjusted_img, ocr_img = enhance_for_ocr(warped)

            # Tampilkan hasil Crop & Processing
            cv2.imshow("Hasil Crop (Natural)", adjusted_img)
            cv2.imshow("Hasil Preprocessing OCR", ocr_img)
            cv2.waitKey(1)

            # 3. Jalankan Tesseract OCR
            extracted_text = run_ocr(ocr_img)
            if not extracted_text:
                extracted_text = run_ocr(adjusted_img)

            # 4. Bacakan teks menggunakan Piper TTS
            if extracted_text:
                speak_with_piper(extracted_text)
            else:
                print("[OCR Warning] Tidak ada teks yang dapat dibaca dari kertas.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
