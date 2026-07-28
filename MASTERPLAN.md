# MASTERPLAN: KAREN → KAREN-Vision (MCP Architecture)

## Vision

Asisten suara + vision lokal untuk kelas, berbasis Gemma 4 E2B, dengan arsitektur
MCP (Model Context Protocol) agar modular, crash-isolated, dan mudah diperluas.

---

## Arsitektur Target

```
┌─────────────────────────────────────────────────────────────────┐
│  main.py (orkestrator, start/stop semua proses)                 │
│                                                                 │
│  ┌──────────┐   ┌──────────────────────┐   ┌──────────────┐    │
│  │ VAD      │   │  llama-server        │   │  Piper TTS   │    │
│  │ record   │──►│  (Gemma 4 E2B)       │──►│  synthesize  │    │
│  └──────────┘   └────────┬─────────────┘   └──────────────┘    │
│                          │                                      │
│                   tool_call (JSON)                              │
│                          │                                      │
│              ┌───────────▼───────────┐                          │
│              │  MCP Host              │ ← Python MCP host       │
│              │  - terima tool_call     │                          │
│              │  - forward ke MCP srv   │                          │
│              │  - balikin hasil        │                          │
│              └───────────┬───────────┘                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     ▼                     ▼                     ▼
┌────────────┐    ┌──────────────┐    ┌──────────────┐
│ MCP Server │    │ MCP Server   │    │ MCP Server   │
│ vision     │    │ ocr          │    │ info/weather │
│ (camera +  │    │ (Tesseract / │    │ (web, JSON)  │
│  describe) │    │  Gemma crop) │    │              │
└────────────┘    └──────────────┘    └──────────────┘
```

---

## Komponen

### Layer 1: Core Pipeline (wajib)

| File | Fungsi | Status |
|---|---|---|
| `main.py` | Entry point, init & loop | 🔴 Belum |
| `core/audio.py` | VAD + record mic | 🔴 Pindah dari gemma4.py |
| `core/gemma_pipeline.py` | Wrapper llama-server (chat + vision) | 🔴 Belum |
| `core/speak_queue.py` | Antrian TTS thread-safe | 🔴 Belum |
| `config.py` | Konstanta terpusat | 🟡 Akan dibuat |

### Layer 2: MCP (Model Context Protocol)

| File | Fungsi | Status |
|---|---|---|
| `mcp/host.py` | MCP host: terima tool_call → forward ke server | 🔴 Belum |
| `mcp/server_base.py` | Base class / util untuk MCP server | 🔴 Belum |

### Layer 3: Tools (MCP Servers)

| Server | Fungsi | Status |
|---|---|---|
| `mcp/servers/vision_server.py` | Camera capture + describe (Gemma 4 vision) | 🔴 Belum |
| `mcp/servers/ocr_server.py` | OCR via Tesseract / Gemma 4 crop | 🔴 Belum |
| `mcp/servers/web_server.py` | DuckDuckGo search + weather | 🟡 Reuse dari tools/web_search.py |
| `mcp/servers/info_server.py` | Query JSON organisasi / DB lokal | 🟡 Reuse dari tools/organisasi_search.py |

### Layer 4: Pipeline (suara)

| File | Fungsi | Status |
|---|---|---|
| `core/pipeline.py` | VoiceAssistantPipeline yg sudah dibersihkan | 🟡 Refactor dari gemma4.py |
| `speak.py` | Piper TTS standalone | 🟢 Reuse |

---

## Phase Plan

### Phase 1: Initial Cleanup (sekarang)

- [x] Fix `pw-record` command (hapus `-a`)
- [x] Hapus VoiceScope dari main loop
- [x] Log ke console (StreamHandler)
- [ ] Debounce spam 'c' key
- [ ] Hapus file obsolete: `greeter.py`, `anime.py`, `visualizer.py`, `database-search.py`, `gemini.py`
- [ ] Hapus `tools/database_search.py`
- [ ] Hapus `models/asr/` (Qwen3 GGUF tidak dipakai)
- [ ] Buat `config.py` berisi konstanta dari gemma4.py
- [ ] Buat struktur direktori `mcp/servers/`

### Phase 2: MCP Infrastructure

- [ ] Buat `mcp/host.py` — implementasi MCP host via stdio JSON-RPC
- [ ] Buat `mcp/server_base.py` — base class untuk MCP server
- [ ] Migrasi `tools/web_search.py` → `mcp/servers/web_server.py`
- [ ] Migrasi `tools/organisasi_search.py` → `mcp/servers/info_server.py`
- [ ] Hapus `tools/registry.py` (tidak diperlukan lagi)
- [ ] Integrasi MCP host ke main pipeline

### Phase 3: Vision

- [ ] Buat `mcp/servers/vision_server.py` — camera capture via OpenCV
- [ ] Integrasi Gemma 4 multimodal (kirim base64 image ke /v1/chat/completions)
- [ ] Tool: `cek_sekitar()` — ambil foto → describe
- [ ] Tool: `baca_teks()` — ambil foto → crop → OCR
- [ ] Test latency end-to-end

### Phase 4: Polish

- [ ] `core/speak_queue.py` — antrian TTS thread-safe
- [ ] `core/pipeline.py` — pipeline tunggal (tidak ada lagi Pipeline A vs B)
- [ ] Hapus `gemma4.py` setelah semua logic pindah
- [ ] Hapus fallback `_detect_tool_intent()` (MCP handle sendiri)
- [ ] Test di Orange Pi / device target

---

## Prinsip Desain

1. **Satu pipeline** — semua request (voice & vision) lewat `core/gemma_pipeline.py`
2. **Tool = MCP Server** — setiap tool jalan di proses sendiri, komunikasi JSON-RPC via stdio
3. **Camera exclusive** — vision server satu-satunya yang akses kamera
4. **No image 저장** — vision cuma simpan teks deskripsi, bukan frame
5. **Config terpusat** — semua path, port, threshold di `config.py`
