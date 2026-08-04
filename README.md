# KAREN (Karya Asisten Registrasi & Edukasi Nirkabel) - Embedded AI Voice & Vision Assistant

KAREN adalah asisten suara dan penglihatan pintar berbasis AI Bahasa Indonesia yang dirancang khusus untuk berjalan di perangkat **Single Board Computer (SBC)** seperti Orange Pi / Raspberry Pi maupun PC/Laptop secara **headless** atau interaktif.

Proyek ini mengintegrasikan **Model Context Protocol (MCP)**, **Gemma 4 VLM (Multimodal)** lokal via `llama-server`, **YOLO Object Detection (ONNX)**, **ArcFace & YOLOv8 Face Recognition**, **Silero VAD**, dan **Piper TTS**.

---

## 🛠️ Arsitektur & Fitur Utama

```
                     ┌────────────────────────┐
                     │   main.py (CLI Main)   │
                     └───────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       ┌──────────────────┐           ┌──────────────────┐
       │ core/audio.py    │           │ core/speak_queue │
       │ (VAD Recording)  │           │   (Piper TTS)    │
       └──────────────────┘           └──────────────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │ core/gemma_pipeline   │
     └───────────┬───────────┘
                 │
                 ├──────────────────────────┐
                 ▼                          ▼
      ┌────────────────────┐     ┌─────────────────────┐
      │ llama-server       │     │ MCP Host & Servers  │
      │ (Gemma 4 Multimodal│     │ - info_server       │
      │  ASR, Chat, VLM)   │     │ - weather_server    │
      └────────────────────┘     │ - search_server     │
                                 │ - vision_server     │
                                 │   (YOLO & Camera)   │
                                 └─────────────────────┘
```

### 1. 🎤 Voice Activity Detection (VAD) & Input Audio
* **Silero VAD v5** (`models/silero_vad.onnx`) via ONNX Runtime.
* Otomatis memotong rekaman audio saat pengguna berhenti berbicara.
* Kompatibel dengan USB Soundcard / Microphone Adapter di SBC (`pw-record` / PipeWire).

### 2. 🧠 Native Audio ASR & LLM (Gemma 4 Multimodal)
* Menggunakan **Gemma 4 E2B** (`models/Gemma4/gemma-4-E2B-it-UD-Q4_K_XL.gguf`) diproses via `llama-server`.
* **Native Audio Understanding**: Menerjemahkan audio WAV langsung tanpa perlu engine Whisper terpisah.
* **Function Calling / Tool Use**: Terhubung otomatis ke MCP server untuk mengambil data dunia nyata.

### 3. 🔌 MCP Tool Architecture (Model Context Protocol)
Semua kapabilitas eksternal dikemas ke dalam server MCP independen:
* **`info_server`**: Informasi organisasi/pendaftaran KIR.
* **`weather_server`**: Informasi cuaca real-time.
* **`search_server`**: DuckDuckGo web search.
* **`vision_server`**: Penglihatan kamera & deteksi objek.

### 4. 👁️ Camera & Vision (YOLO + Gemma 4 VLM)
* **Single Camera Ownership**: Camera device (`/dev/video0`) di-lock secara eksklusif oleh daemon thread di `vision_server.py`.
* **YOLO Object Detection (ONNX)**: Berjalan secara kontinu di background untuk mendeteksi 80 kelas COCO tanpa membebani LLM.
* **VLM On-Demand**: Menggunakan Gemma 4 Vision untuk membaca teks (OCR) atau mendeskripsikan suasana sekitar saat tombol `f` ditekan atau diminta lewat suara.

### 5. 🔊 Speech Synthesis (TTS)
* **Piper TTS** (`models/piper/id_ID-news_tts-medium.onnx`) untuk menghasilkan respons suara Bahasa Indonesia yang natural.
* Non-blocking audio queue playback (`core/speak_queue.py`).

### 6. 👤 Face Recognition (Fitur Terpisah)
* [face.py](file:///home/normies/Projects/KAREN/face.py) & [app/registration/register_person.py](file:///home/normies/Projects/KAREN/app/registration/register_person.py): Sistem pengenalan wajah real-time berbasis YOLOv8-Face + ArcFace/w600k_r50 ONNX dengan SQLite database.

---

## 📁 Struktur Direktori

```bash
KAREN/
├── main.py                # Main Entry Point (Voice + Vision Assistant CLI)
├── config.py              # Konfigurasi terpusat (Path, Port, Threshold, Device)
├── face.py                # Standalone Real-Time Face Recognition GUI
├── paper_detection_llm.py # Standalone OCR/Document Reader + LLM
├── core/                  # Core modules
│   ├── audio.py           # Silero VAD + Recording
│   ├── gemma_pipeline.py  # Lifecycle llama-server, ASR, Chat, Vision & MCP Host
│   └── speak_queue.py     # Piper TTS synthesizer & audio player
├── karen_mcp/             # System MCP (Model Context Protocol)
│   ├── host.py            # MCP Host Client orchestrator
│   └── servers/           # Individual MCP Servers
│       ├── info_server.py
│       ├── weather_server.py
│       ├── search_server.py
│       └── vision_server.py # Camera ownership, YOLO loop, VLM tools
├── app/                   # Module Face Recognition & Overlay Renderer
├── models/                # Folder penyimpanan model ONNX & GGUF
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Persiapan Model & Environment
Pastikan file model berada di folder `models/`:
* `models/Gemma4/gemma-4-E2B-it-UD-Q4_K_XL.gguf`
* `models/Gemma4/mmproj-F16.gguf`
* `models/silero_vad.onnx`
* `models/piper/id_ID-news_tts-medium.onnx`
* `models/yolov26n.onnx`

Variabel lingkungan dapat disesuaikan di `.env` atau `config.py`:
```env
CAMERA_INDEX=0
GEMMA4_PORT=8080
```

### 2. Menjalankan KAREN Assistant
Gunakan `uv` untuk menjalankannya:
```bash
uv run main.py
```

### 3. Kontrol Keyboard (Interaktif CLI)
* `r` : Rekam suara (VAD akan mendeteksi ketika Anda selesai berbicara).
* `f` : Ambil foto dari kamera & analisis dengan LLM (Deskripsi / OCR).
* `c` : Reset riwayat percakapan.
* `q` : Keluar dari program.
