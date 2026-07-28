# KAREN — Knowledge Graph

**KAREN** = *Karya Asisten Registrasi & Edukasi Nirkabel*  
Voice assistant untuk KIR MAN 2 Kota Bogor → sedang transisi ke **KAREN-Vision** (asisten kelas lokal + vision). Python 3.12, pipeline tunggal (Gemma 4 lokal).

---

## 🧠 Entities & Relations

### 1. Pipeline (Saat Ini)

```
Pipeline ──hanya──► gemma4.py ──uses──► llama-server (local LLM)
                                        ├──hosts──► Gemma 4 (GGUF)
                                        └──uses──► Gemma 4 native audio (ASR)

              ──share──► Silero VAD (voice activity detection)
              ──share──► Piper TTS (Indonesian TTS)
              ──runs──► headless (console log, no GUI)
```

### 2. Tools (Function Calling)

```
LLM ──calls──► Tool Registry (tools/registry.py)
                  │
                  ├──► cari_web(query) ──searches──► DuckDuckGo
                  ├──► cek_cuaca(kota) ──fetches──► DuckDuckGo Weather
                  ├──► cari_info_organisasi(query) ──searches──► data/organisasi.json
                  └──► get_semua_info_organisasi() ──returns──► data/organisasi.json
```

### 3. Independent Modules

```
speak.py   ──plays──► WAV files ──with──► Terminal waveform (plotext)
```

### 4. Models (Local, di `models/`)

```
models/
  ├── Gemma4/
  │   ├── gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf   # LLM (Q2)
  │   ├── gemma-4-E2B-it-UD-Q4_K_XL.gguf        # LLM (Q4) ← active
  │   ├── mmproj-F16.gguf                        # Multimodal projector
  │   └── mtp-gemma-4-E2B-it.gguf               # Multi-turn prediction
  ├── piper/
  │   └── id_ID-news_tts-medium.onnx            # TTS voice (ID)
  ├── silero_vad.onnx                            # VAD
  └── hand_landmarker.task                       # MediaPipe hand (unused)
```

### 5. Data Flow

```
Mic ──(pw-record)──► WAV ──► Silero VAD ──► base64 WAV ──► llama-server (Gemma4)
  ──► teks ASR ──► Gemma4 + tools ──► respons teks ──► Piper TTS ──► speaker
  ──► console log (DEBUG level)
```

### 6. Agent Skills (Developer)

```
.agents/skills/
  ├── gemini-api-dev/SKILL.md                    # Panduan Gemini API SDK
  └── gemini-interactions-api/
      ├── SKILL.md                               # Panduan Interactions API
      └── references/migration.md                # Migrasi generateContent → Interactions
```

### 7. Config

```
config.py ──holds──► LLAMA_SERVER_PATH, model paths, port, VAD params, TTS path
.env      ──holds──► SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY (legacy, unused)
pyproject.toml ──lists──► dependencies (ddgs, google-genai, mediapipe, piper, dll)
```

### 8. Audio Pipeline

```
Recording:   pw-record (16kHz, mono, S16LE) ──► WAV file
Playback:    pygame.mixer.Sound ──► speaker
```

### 9. Architecture Target (MCP — lihat MASTERPLAN.md)

```
main.py ──► llama-server ──► MCP Host ──► MCP Servers (vision, ocr, web, info)
                                            └──tiap server proses terpisah
```

---

**Ringkasan:** KAREN sekarang pipeline tunggal (Gemma 4 lokal, headless), 4 tools (web, cuaca, organisasi), tanpa cloud/Supabase/visualizer. Sedang bertransisi ke arsitektur MCP untuk vision & OCR — lihat `MASTERPLAN.md`.
