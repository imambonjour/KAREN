# MASTERPLAN: KAREN → KAREN-Vision (MCP Architecture)

## Vision

Asisten suara + vision lokal untuk kelas, berbasis Gemma 4 E2B, dengan arsitektur
MCP (Model Context Protocol) agar modular, crash-isolated, dan mudah diperluas.

---

## Arsitektur Final

```
┌────────────────────────────────────────────────────────────────────┐
│  main.py (orkestrator, start/stop semua proses)                   │
│                                                                    │
│  ┌──────────┐   ┌───────────────────────┐   ┌──────────────┐      │
│  │ VAD      │   │  llama-server          │   │  Piper TTS   │      │
│  │ record   │──►│  (Gemma 4 E2B GGUF)   │──►│  synthesize  │      │
│  └──────────┘   └────────┬──────────────┘   └──────────────┘      │
│                          │                                         │
│                   tool_call (JSON)                                 │
│                          │                                         │
│              ┌───────────▼───────────┐                             │
│              │  MCP Host             │ ← karen_mcp/host.py         │
│              │  (stdio JSON-RPC)     │   (Anthropic MCP SDK)       │
│              └───────────┬───────────┘                             │
└──────────────────────────┼─────────────────────────────────────────┘
                           │
     ┌─────────────────────┼──────────────────────────┐
     ▼                     ▼                          ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────────┐
│ karen-web      │  │ karen-info     │  │ karen-vision       │
│ cari_web       │  │ cari_info      │  │ cek_sekitar        │
│ cek_cuaca      │  │ _organisasi    │  │ baca_teks (OCR)    │
│ (DuckDuckGo)   │  │ get_semua_info │  │ (USB Webcam +      │
│                │  │ (JSON lokal)   │  │  Gemma 4 Vision)   │
└────────────────┘  └────────────────┘  └────────────────────┘
```

---

## Komponen (Final)

### Layer 1: Core Pipeline

| File | Fungsi |
|---|---|
| `main.py` | Entry point, init & keyboard loop (`r`/`c`/`q`) |
| `core/audio.py` | Silero VAD + pw-record mic capture |
| `core/gemma_pipeline.py` | Wrapper llama-server (ASR, chat, vision) + MCP integration |
| `core/speak_queue.py` | Piper TTS synthesis + pygame playback |
| `config.py` | Konstanta terpusat (path, port, threshold) |

### Layer 2: MCP Infrastructure

| File | Fungsi |
|---|---|
| `karen_mcp/host.py` | MCP host: launch servers, handshake, aggregate schemas, forward tool calls (Anthropic MCP SDK) |

### Layer 3: MCP Servers

| Server File | Tools |
|---|---|
| `karen_mcp/servers/web_server.py` | `cari_web`, `cek_cuaca` |
| `karen_mcp/servers/info_server.py` | `cari_info_organisasi`, `get_semua_info_organisasi` |
| `karen_mcp/servers/vision_server.py` | `cek_sekitar`, `baca_teks` |

---

## Status Implementasi

### ✅ Phase 1: Initial Cleanup

- [x] Fix `pw-record` command (hapus `-a`)
- [x] Hapus VoiceScope dari main loop
- [x] Log ke console (StreamHandler)
- [x] Debounce spam 'c' key
- [x] Hapus file obsolete: `greeter.py`, `anime.py`, `visualizer.py`, `database-search.py`, `gemini.py`
- [x] Hapus `tools/database_search.py`
- [x] Hapus `models/asr/`
- [x] Buat `config.py`
- [x] Buat struktur direktori `karen_mcp/servers/`

### ✅ Phase 2: MCP Infrastructure

- [x] Buat `karen_mcp/host.py` — MCP host via stdio JSON-RPC (Anthropic MCP SDK)
- [x] Migrasi `tools/web_search.py` → `karen_mcp/servers/web_server.py`
- [x] Migrasi `tools/organisasi_search.py` → `karen_mcp/servers/info_server.py`
- [x] Hapus `tools/registry.py` dan seluruh folder `tools/`
- [x] Integrasi MCP host ke `core/gemma_pipeline.py`

### ✅ Phase 3: Vision

- [x] Buat `karen_mcp/servers/vision_server.py` — camera capture via OpenCV
- [x] Integrasi Gemma 4 multimodal (kirim base64 image ke `/v1/chat/completions`)
- [x] Tool: `cek_sekitar()` — ambil foto → describe
- [x] Tool: `baca_teks()` — ambil foto → OCR

### ✅ Phase 4: Polish

- [x] `core/speak_queue.py` — queue TTS thread-safe
- [x] Hapus `gemma4.py` — semua logic pindah ke `core/` + `main.py`
- [x] Hapus fallback `_detect_tool_intent()` — MCP handle sendiri
- [x] `walkthrough.md` — dokumentasi migrasi
- [x] `KNOWLEDGE_GRAPH.md` — diperbarui sesuai arsitektur MCP

---

## Cara Menjalankan

```bash
python main.py
```

Kontrol: `r` = record, `c` = clear history, `q` = quit.

---

## Prinsip Desain

1. **Satu pipeline** — semua request (voice & vision) lewat `core/gemma_pipeline.py`
2. **Tool = MCP Server** — setiap tool jalan di proses sendiri, komunikasi JSON-RPC via stdio
3. **Camera exclusive** — vision server satu-satunya yang akses kamera
4. **No image storage** — vision cuma simpan teks deskripsi, bukan frame
5. **Config terpusat** — semua path, port, threshold di `config.py`
