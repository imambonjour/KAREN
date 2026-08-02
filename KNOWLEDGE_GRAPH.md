# KAREN — Knowledge Graph

**KAREN** = *Karya Asisten Registrasi & Edukasi Nirkabel*  
Voice & Vision Assistant lokal berbasis **Gemma 4 E2B** dengan arsitektur **MCP (Model Context Protocol)**. Python 3.12, modular & crash-isolated.

---

## 🧠 Entities & Relations

### 1. Core Architecture (MCP-Based)

```
main.py (orchestrator)
  ├──► core/audio.py ─────────► Silero VAD + pw-record (Mic)
  ├──► core/gemma_pipeline.py ─► llama-server (Gemma 4 E2B GGUF)
  │                              └── hosts: LLM + native ASR + Multimodal Vision
  ├──► core/speak_queue.py ────► Piper TTS (Indonesian voice)
  │                              └── playback: pygame.mixer
  └──► karen_mcp/host.py ─────► MCP Host (stdio JSON-RPC manager)
                                 ├──► karen_mcp/servers/web_server.py    (cari_web, cek_cuaca)
                                 ├──► karen_mcp/servers/info_server.py   (cari_info_organisasi, get_semua_info_organisasi)
                                 └──► karen_mcp/servers/vision_server.py (cek_sekitar, baca_teks via USB webcam)
```

### 2. MCP Servers & Tools

| Server | Tools | Fungsi | Data / Hardware Source |
|---|---|---|---|
| `karen-web` | `cari_web`, `cek_cuaca` | Pencarian web & info cuaca | DuckDuckGo via `ddgs` |
| `karen-info` | `cari_info_organisasi`, `get_semua_info_organisasi` | Informasi KIR MAN 2 Kota Bogor | `data/organisasi.json` |
| `karen-vision` | `cek_sekitar`, `baca_teks` | Deskripsi pemandangan & OCR | USB Webcam (`/dev/video0`) + Gemma 4 Vision |

### 3. Models (Local, di `models/`)

```
models/
  ├── Gemma4/
  │   ├── gemma-4-E2B-it-UD-Q4_K_XL.gguf  # LLM & ASR (Q4)
  │   └── mmproj-F16.gguf                  # Multimodal projector (Vision)
  ├── piper/
  │   └── id_ID-news_tts-medium.onnx      # TTS voice (ID)
  └── silero_vad.onnx                      # VAD
```

### 4. Data Flow

```
Mic ──(pw-record)──► WAV ──► Silero VAD ──► Base64 WAV ──► llama-server (ASR)
  ──► Transkripsi ──► Gemma 4 + MCP Host ──► [Tool Execution via MCP Server Subprocess]
  ──► Respons Teks ──► Piper TTS ──► Speaker (pygame)
```

### 5. Project Layout

```
KAREN/
  ├── main.py                   # Entry point
  ├── config.py                 # Terpusat: path model, port, VAD params, TTS path
  ├── core/
  │   ├── audio.py              # VAD + mic recording
  │   ├── gemma_pipeline.py     # Wrapper llama-server + MCP integration
  │   └── speak_queue.py        # TTS synthesis & playback
  ├── karen_mcp/
  │   ├── host.py               # MCP Host (stdio client manager & tool aggregator)
  │   └── servers/
  │       ├── web_server.py     # Web search & weather MCP server
  │       ├── info_server.py    # Organisasi KIR info MCP server
  │       └── vision_server.py  # Camera capture & OCR MCP server
  ├── data/
  │   └── organisasi.json       # Database organisasi KIR
  └── models/                   # Local GGUF, ONNX, and model files
```

---

**Ringkasan:** KAREN bertransisi ke KAREN-Vision dengan arsitektur MCP modular. Setiap tool berjalan di proses terpisah (stdio JSON-RPC), menjamin crash isolation dan modularitas penuh untuk integrasi suara dan penglihatan (vision).
