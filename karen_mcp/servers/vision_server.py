#!/usr/bin/env python3
"""
MCP Server: Vision — Camera capture + describe / OCR.
Tools: cek_sekitar, baca_teks.
Uses USB webcam (/dev/video0) via OpenCV + Gemma 4 vision (multimodal).
Run standalone:  python -m karen_mcp.servers.vision_server
"""

import base64
import logging
import os

import requests

from mcp.server.fastmcp import FastMCP

log = logging.getLogger("karen.vision_server")

# Gemma 4 llama-server endpoint (same server as main pipeline)
GEMMA4_PORT = int(os.environ.get("GEMMA4_PORT", "8080"))
GEMMA4_CHAT_URL = f"http://localhost:{GEMMA4_PORT}/v1/chat/completions"

# Camera device index (USB webcam)
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))

server = FastMCP(
    "karen-vision",
    instructions="Menyediakan kemampuan penglihatan (kamera) untuk mendeskripsikan lingkungan dan membaca teks.",
)


def _capture_frame() -> bytes:
    """Capture a single frame from the USB webcam, return as JPEG bytes."""
    import cv2

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (index={CAMERA_INDEX})")
    try:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        _, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes()
    finally:
        cap.release()


def _ask_gemma_vision(image_b64: str, prompt: str) -> str:
    """Send an image + prompt to Gemma 4 vision and return the text response."""
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
    response = requests.post(GEMMA4_CHAT_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


@server.tool()
def cek_sekitar() -> dict:
    """Mengambil foto dari kamera dan mendeskripsikan apa yang terlihat di sekitar.
    Gunakan ketika pengguna bertanya 'apa yang kamu lihat', 'lihat sekitar', 'ada apa di depan', dll."""
    try:
        jpeg_bytes = _capture_frame()
        image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
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
        jpeg_bytes = _capture_frame()
        image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
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


if __name__ == "__main__":
    server.run(transport="stdio")
