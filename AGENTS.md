# KAREN — Project Index

> **KAREN** = *Karya Asisten Registrasi & Edukasi Nirkabel*
> Lokal AI voice + vision assistant (Bahasa Indonesia) for SBC/PC (headless or interactive).
> Python 3.12, modular & crash-isolated via MCP. Docs: `README.md`, `MASTERPLAN.md`, `PLAN_YOLO_VISION_KEY.md`, `KNOWLEDGE_GRAPH.md`, `konteks-proyek-KAREN-OPSI.md`.

## Architecture Overview

```
main.py (orchestrator, local backend)          gemini.py (orchestrator, cloud backend)
  ├── r → core/audio.py (Silero VAD + pw-record)   ├── r → core/audio.py → core/gemini_pipeline.py (Gemini API) ASR
  │        └── core/gemma_pipeline.py                 └── chat + MCP tool calling → karen_mcp/host.py
  │             (llama-server: Gemma 4 E2B) ASR
  │             └── chat + tool calling → karen_mcp/host.py (MCP stdio)
  │                  └── servers: karen-web / karen-info / karen-vision
  │             └── core/speak_queue.py (Piper TTS + pygame playback)
  ├── f → pipeline.vision_scan() → MCP tool `analisa_foto` → TTS
  ├── c → reset conversation history
  └── q → quit
```

Both backends share the same MCP servers, VAD, TTS, and keyboard loop (`r`/`f`/`c`/`q`).
`main.py` = fully local (llama-server Gemma 4). `gemini.py` = cloud (Google Gemini API, no llama-server; requires `GEMINI_API_KEY`).

Key design principles (see MASTERPLAN):
1. **One pipeline** — all voice & vision requests go through `core/gemma_pipeline.py`.
2. **Tool = MCP server** — each tool runs in its own process (stdio JSON-RPC), crash-isolated.
3. **Camera exclusive** — only `vision_server.py` opens `cv2.VideoCapture`; YOLO + tools read shared state.
4. **No image storage** — frames stay in memory, never written to disk.
5. **Central config** — all paths/ports/thresholds in `config.py`.

## Directory Map

| Path | Purpose |
|---|---|
| `main.py` | Entry point (local backend). Keyboard loop: `r` record, `f` photo scan, `c` clear history, `q` quit. YOLO proactive announce loop. |
| `gemini.py` | Entry point (cloud backend, Google Gemini API). Same keyboard loop; no llama-server. Requires `GEMINI_API_KEY`. Logs to `gemini_assistant.log`. |
| `config.py` | Central config: loads `.env` (dotenv), llama-server path/port, model paths, device/audio settings, VAD/YOLO params, `GEMINI_MODEL_NAME`. |
| `core/audio.py` | `SileroVAD` (ONNX), `StreamingVADIterator`, `record_speech()` via `pw-record` (honors `AUDIO_INPUT_DEVICE` → `--target`). |
| `core/gemma_pipeline.py` | `GemmaPipeline`: llama-server lifecycle, native audio ASR, `query_llm` (chat + MCP tool calls), `vision_scan`, `detect_objects`, history mgmt. |
| `core/gemini_pipeline.py` | `GeminiPipeline`: Google `genai` client, ASR via multimodal audio, chat + MCP tool calling (converts OpenAI→Gemini function decls), `vision_scan`, `vision_scan_direct`, `detect_objects`, history mgmt. |
| `core/speak_queue.py` | `SpeakQueue`: Piper TTS synth + blocking `play_audio` with `r`/`q` interruption. |
| `karen_mcp/host.py` | `MCPHost`: launches server subprocesses, handshake, aggregates OpenAI tool schemas, `call_tool()`. |
| `karen_mcp/servers/web_server.py` | `karen-web`: `cari_web`, `cek_cuaca` (DuckDuckGo via `ddgs`). |
| `karen_mcp/servers/info_server.py` | `karen-info`: `cari_info_organisasi`, `get_semua_info_organisasi` (queries `data/organisasi.json`). |
| `karen_mcp/servers/vision_server.py` | `karen-vision`: `cek_sekitar`, `baca_teks`, `analisa_foto`, `deteksi_objek`. Camera thread + YOLO ONNX loop (camera-exclusive owner). |
| `face.py` | Standalone real-time face recognition GUI (YOLOv8-Face + ArcFace). Keys: `Q` quit, `R` reload DB, `E` expand samples, `U` assign unknown. |
| `paper_detection.py` | Standalone paper contour detection → Tesseract OCR → Piper TTS. |
| `paper_detection_llm.py` | Standalone paper detection → perspective transform → Gemma 4 vision OCR → Piper TTS. |
| `speak.py` | CLI waveform visualizer for WAV files (plotext). |
| `konteks-proyek-KAREN-OPSI.md` | Project context: OPSI 2026 proposal "Smart-Vision Pedagogy" (blind-teacher assistive glasses), scoping decisions, OPSI compliance notes. |
| `app/config.py` | Face-recognition subsystem config (models, thresholds, DB, camera). |
| `app/database/db.py` | SQLite (`data/database.db`): `persons` + `person_embeddings` tables, legacy migration, add/get/delete. |
| `app/detection/face_detector.py` | `FaceDetector`: YOLOv8-Face ONNX with letterboxing, landmarks, NMS. |
| `app/detection/yolo_detector.py` | `YOLODetector` (Ultralytics, for overlay). |
| `app/recognition/face_recognizer.py` | `FaceRecognizer`: ArcFace w600k_r50, 5-point alignment, 512-d L2-normalized embeddings. |
| `app/tracking/bytetrack.py` | `SimpleTracker`/`Track`: greedy IoU tracking. |
| `app/cache/track_cache.py` | `TrackCache`: track_id→name cache + ghost cache for lost-then-reappearing faces. |
| `app/overlay/renderer.py` | `Renderer`: HUD + corner-rect overlays, merges YOLO person boxes with face tracks. |
| `app/utils/similarity.py` | `compute_similarity`, `find_best_match` (MAX-similarity across samples, threshold 0.45). |
| `app/utils/download_models.py` | Downloads yolov8n-face.onnx + buffalo_s ArcFace. |
| `app/registration/register_person.py` | CLI `--name` registration: captures N samples → mean embedding → DB. |

## MCP Tools Exposed to LLM

| Tool | Server | Description |
|---|---|---|
| `cari_web(query, max_results)` | karen-web | DuckDuckGo web search |
| `cek_cuaca(kota)` | karen-web | Current weather (DDG search) |
| `cari_info_organisasi(query)` | karen-info | KIR organization info search |
| `get_semua_info_organisasi()` | karen-info | All org data dump |
| `cek_sekitar()` | karen-vision | Describe camera view (Gemma 4 VLM) |
| `baca_teks()` | karen-vision | OCR via Gemma 4 VLM |
| `analisa_foto()` | karen-vision | LLM decides OCR vs description (used by `f` key) |
| `deteksi_objek()` | karen-vision | Latest YOLO detections, no LLM |

## Models (in `models/`, gitignored)

- `Gemma4/gemma-4-E2B-it-UD-Q4_K_XL.gguf` — LLM + native ASR (Q4)
- `Gemma4/mmproj-F16.gguf` — multimodal vision projector
- `Gemma4/mtp-gemma-4-E2B-it.gguf` — multi-token prediction (unused in config)
- `piper/id_ID-news_tts-medium.onnx` — Indonesian TTS voice
- `silero_vad.onnx` — VAD v5
- `yolov26n.onnx` — YOLO detection (config `YOLO_MODEL_PATH`, used by vision_server)
- Face subsystem: `yolov8n-face.onnx`, `w600k_r50.onnx` (ArcFace), `arcface.onnx`

Note: `config.YOLO_MODEL_PATH` points to `models/yolov26n.onnx`; `models/yolo/` dir is empty. `app/detection/yolo_detector.py` uses `BASE_DIR/yolo26n.pt` (Ultralytics, `.pt`).

## Commands

```bash
uv run main.py                          # main voice+vision assistant (local: r/f/c/q)
uv run gemini.py                        # cloud backend (Gemini API; requires GEMINI_API_KEY)
uv run face.py                          # face recognition GUI
uv run app/registration/register_person.py --name "Nama" [--samples 5]
uv run paper_detection.py               # contour + Tesseract OCR
uv run paper_detection_llm.py           # contour + Gemma 4 vision OCR
uv run python -m karen_mcp.servers.vision_server   # run MCP server standalone
uv run speak.py output_response.wav     # play + visualize WAV
```

Compile check: `uv run python -m py_compile main.py core/*.py config.py karen_mcp/servers/*.py`

## Config Highlights (`config.py`)

- Loads `.env` via dotenv at import; `CAMERA_INDEX`, `GEMMA4_PORT`, `GEMINI_MODEL_NAME` overridable via env.
- Device/audio: `CAMERA_PREVIEW` (bool), `AUDIO_INPUT_DEVICE`, `AUDIO_OUTPUT_DEVICE` (used as `--target` for `pw-record`).
- Cloud: `GEMINI_MODEL_NAME` default `gemini-3.5-flash`.
- YOLO: `YOLO_MODEL_PATH=models/yolov26n.onnx`, input 640, conf 0.5, IoU 0.45, infer interval 0.5s.
- `YOLO_TARGET_CLASSES=[]` → auto-announce off. Fill e.g. `["person"]` + `YOLO_POLL_INTERVAL=1.0`, `YOLO_ANNOUNCE_COOLDOWN=15.0` to enable proactive announce.
- VAD: threshold 0.5, min silence 500ms, speech pad 30ms, min speech 250ms, 16kHz.
- `MAX_HISTORY_TURNS=10`.

## Conventions & Gotchas

- All assistant speech must be in Bahasa Indonesia (system prompt enforces it).
- `GemmaPipeline._loop` / `GeminiPipeline._loop` are single asyncio event loops; MCP calls use `run_until_complete` (no threads).
- TTS playback is blocking via `play_audio(...)`; interruption chars are `r`/`q`.
- llama-server is launched as subprocess with `--mlock --no-mmap`, 8k context, port 8080.
- Camera is opened ONCE by `_CameraThread` in vision_server; tools call `_get_latest_frame_b64()`.
- `vision_scan()`/`detect_objects()` parse MCP JSON results (`{"hasil":...}`, `{"detections":[...]}`).
- Gotcha: `analisa_foto`/`cek_sekitar`/`baca_teks` in vision_server call Gemma VLM via llama-server (`GEMMA4_CHAT_URL`). In the `gemini.py` cloud backend there is no llama-server, so `GeminiPipeline.vision_scan()` only works if a local llama-server happens to be running; `vision_scan_direct()` is the full-cloud alternative (needs a frame via a new MCP tool — not yet wired).
- Face subsystem stores per-sample embeddings (not mean) since migration; `find_best_match` uses MAX similarity.
- `.env` still contains legacy keys (`SUPABASE_URL`, `GEMINI_API_KEY`) — unused by current code (see `cleanup.md`).
- Run via `uv` (`uv run ...`); deps in `pyproject.toml` (mcp, onnxruntime, opencv, piper, pygame, pytesseract, ddgs, requests, numpy).
