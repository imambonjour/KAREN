# Walkthrough: KAREN-Vision MCP Migration

Telah dilakukan refactoring dan migrasi arsitektur KAREN dari monolithic (`gemma4.py`) menjadi arsitektur modular berbasis **Model Context Protocol (MCP)** sesuai dengan `MASTERPLAN.md` dan `KNOWLEDGE_GRAPH.md`.

---

## 🛠️ Ringkasan Perubahan

### 1. MCP Infrastructure (`karen_mcp/`)
- **[karen_mcp/host.py](file:///home/normies/Projects/KAREN/karen_mcp/host.py)**: Built-in MCP Host di Python menggunakan SDK resmi Anthropic MCP. Berfungsi meluncurkan subprocess server via `stdio`, menangani *handshake*, mengumpulkan deskripsi tool ke schema OpenAI function-calling untuk `llama-server`, dan mengeksekusi *tool calls*.
- **[karen_mcp/servers/web_server.py](file:///home/normies/Projects/KAREN/karen_mcp/servers/web_server.py)**: FastMCP server terpisah untuk `cari_web` dan `cek_cuaca` (DuckDuckGo via `ddgs`).
- **[karen_mcp/servers/info_server.py](file:///home/normies/Projects/KAREN/karen_mcp/servers/info_server.py)**: FastMCP server terpisah untuk `cari_info_organisasi` dan `get_semua_info_organisasi` (queries ke `data/organisasi.json`).
- **[karen_mcp/servers/vision_server.py](file:///home/normies/Projects/KAREN/karen_mcp/servers/vision_server.py)**: FastMCP server terpisah untuk `cek_sekitar` (pengambilan gambar & deskripsi suasana) dan `baca_teks` (OCR) menggunakan USB Webcam standar (`/dev/video0`) via OpenCV + Gemma 4 Multimodal Vision.

### 2. Core Modules (`core/`)
- **[core/audio.py](file:///home/normies/Projects/KAREN/core/audio.py)**: Modul khusus perekaman mikrofon (`pw-record`) dan deteksi suara berbasis Silero VAD v5 ONNX.
- **[core/gemma_pipeline.py](file:///home/normies/Projects/KAREN/core/gemma_pipeline.py)**: Wrapper pengelola *lifecycle* `llama-server` (Gemma 4 Q4_K_XL + mmproj), ASR native audio, inferensi LLM, serta integrasi dengan MCP Host.
- **[core/speak_queue.py](file:///home/normies/Projects/KAREN/core/speak_queue.py)**: Queue sintesis suara Piper TTS dan playback audio non-blocking berbasis `pygame.mixer`.

### 3. Entry Point Utama (`main.py`)
- **[main.py](file:///home/normies/Projects/KAREN/main.py)**: Entry point tunggal menggantikan `gemma4.py`. Menghubungkan seluruh komponen audio, Gemma 4, MCP Host, dan kontrol keyboard (`r` = rekam, `c` = hapus history, `q` = keluar).

### 4. Cleanup & Dependensi
- Menghapus folder `tools/` lama (`registry.py`, `web_search.py`, `organisasi_search.py`) dan `gemma4.py`.
- Memperbarui `pyproject.toml` dengan dependensi terpilih (`mcp>=1.28.0,<2.0.0`, `opencv-python`, dll).
- Memperbarui `KNOWLEDGE_GRAPH.md` sesuai arsitektur MCP baru.

---

## 🧪 Verifikasi yang Telah Dilakukan

1. **Uji Impor Modul Core:**
   Semua modul `core/audio.py`, `core/gemma_pipeline.py`, `core/speak_queue.py`, dan `karen_mcp/host.py` terimpor tanpa error.

2. **Uji Handshake & Penemuan Tool MCP:**
   `MCPHost` berhasil meluncurkan ke-3 subprocess MCP Server (`karen-web`, `karen-info`, `karen-vision`), melakukan *list_tools* (6 tools total), dan menutupnya secara aman (*graceful shutdown*).

3. **Uji Eksekusi Tool MCP:**
   - Tool `cari_info_organisasi` berhasil mengambil data JSON dari `data/organisasi.json`.
   - Tool `cek_cuaca` berhasil mengambil data cuaca terkini via `ddgs`.

---

## 🚀 Cara Menjalankan

```bash
# Menjalankan KAREN-Vision
python main.py
```
