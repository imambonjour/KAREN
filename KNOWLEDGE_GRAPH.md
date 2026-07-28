# KAREN — Knowledge Graph

**KAREN** = *Karya Asisten Registrasi & Edukasi Nirkabel*  
Voice assistant untuk KIR MAN 2 Kota Bogor. Python 3.12, dual-pipeline (cloud + local).

---

## 🧠 Entities & Relations

### 1. Pipeline

```
Pipeline ──has──► Pipeline A (gemini.py) ──uses──► Gemini API (cloud LLM)
                │                                  └──uses──► pywhispercpp (local ASR)
                │
                └──► Pipeline B (gemma4.py) ──uses──► llama-server (local LLM)
                                                     ├──hosts──► Gemma 4 (GGUF)
                                                     └──uses──► Gemma 4 native audio (ASR)

Keduanya ──share──► Silero VAD (voice activity detection)
                  └──share──► Piper TTS (Indonesian TTS)
```

### 2. Tools (Function Calling)

```
LLM ──calls──► Tool Registry (tools/registry.py)
                  │
                  ├──► cari_pendaftar(nama, sekolah) ──queries──► Supabase (registrations)
                  ├──► hitung_total_pendaftar() ──queries──► Supabase
                  ├──► cari_web(query) ──searches──► DuckDuckGo
                  ├──► cek_cuaca(kota) ──fetches──► DuckDuckGo Weather
                  ├──► cari_info_organisasi(query) ──searches──► data/organisasi.json
                  └──► get_semua_info_organisasi() ──returns──► data/organisasi.json
```

### 3. Database (Supabase)

```
Table: registrations
  ├── id (PK)
  ├── full_name
  ├── school
  ├── whatsapp
  └── created_at

Digunakan oleh: tools/database_search.py, greeter.py
```

### 4. Independent Modules

```
greeter.py ──polls──► Supabase (every 5s) ──triggers──► Piper TTS (greet new registrant)

anime.py   ──captures──► Webcam (MediaPipe hand tracking)
           └──renders──► Anime eyes following finger (OpenCV)

visualizer.py ──displays──► VoiceScope (Pygame GUI)
              ├──mode: STANDBY │ LISTENING │ PROCESSING │ SPEAKING
              └──input: unified keyboard queue

speak.py   ──plays──► WAV files ──with──► Terminal waveform (plotext)

database-search.py ──CLI──► Supabase queries (manual testing)
```

### 5. Models (Local, di `models/`)

```
models/
  ├── Gemma4/
  │   ├── gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf   # LLM (Q2)
  │   ├── gemma-4-E2B-it-UD-Q4_K_XL.gguf        # LLM (Q4)
  │   ├── mmproj-F16.gguf                        # Multimodal projector
  │   └── mtp-gemma-4-E2B-it.gguf               # Multi-turn prediction
  ├── asr/
  │   ├── Qwen3-ASR-0.6B-Q8_0.gguf              # ASR (belum dipakai)
  │   └── mmproj-Qwen3-ASR-0.6B-Q8_0.gguf
  ├── piper/
  │   └── id_ID-news_tts-medium.onnx            # TTS voice (ID)
  ├── silero_vad.onnx                            # VAD
  └── hand_landmarker.task                       # MediaPipe hand
```

### 6. Data Flow (Pipeline A — Gemini)

```
Mic ──(pw-record)──► WAV ──► Silero VAD ──► pywhispercpp ──► teks
  ──► Gemini API + tools ──► respons teks ──► Piper TTS ──► speaker
  ──► visualizer.py (mode indicator + waveform)
```

### 7. Data Flow (Pipeline B — Gemma 4)

```
Mic ──(pw-record)──► WAV ──► Silero VAD ──► base64 WAV ──► llama-server (Gemma4)
  ──► teks ASR ──► Gemma4 + tools ──► respons teks ──► Piper TTS ──► speaker
  ──► visualizer.py
```

### 8. Agent Skills (Developer)

```
.agents/skills/
  ├── gemini-api-dev/SKILL.md                    # Panduan Gemini API SDK
  └── gemini-interactions-api/
      ├── SKILL.md                               # Panduan Interactions API
      └── references/migration.md                # Migrasi generateContent → Interactions
```

### 9. Config

```
.env ──holds──► SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY
pyproject.toml ──lists──► 15 dependencies (ddgs, google-genai, mediapipe, piper, dll)
```

### 10. Audio Pipeline

```
Recording:   pw-record (16kHz, mono, S16LE) ──► WAV file
Playback:    pygame.mixer.Sound ──► speaker
Greeting:    greeter.py ──► Piper ──► greet_temp.wav ──► pygame
```

---

**Ringkasan:** KAREN adalah voice assistant Indonesia dengan 2 mode (Gemini cloud atau Gemma 4 local), 6 tools, database Supabase, TTS Piper, VAD Silero, visualizer Pygame, dan eye-tracker anime — semuanya untuk membantu pendaftaran dan informasi organisasi KIR MAN 2 Bogor.
