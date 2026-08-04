# Rencana: KAREN + YOLO & Tombol Foto di main.py

> **Status:** rencana (belum dieksekusi)
> **Konteks:** menambahkan tombol `f` untuk meng-capture foto & menganalisis dengan
> LLM, dengan YOLO yang berjalan terus-menerus di dalam vision MCP server
> (ONNX + onnxruntime, pemegang kamera tunggal).

## Tujuan

- Tombol keyboard `f` untuk mengambil 1 frame kamera & menganalisis dengan LLM
  (`analisa_foto`) — LLM memutuskan apakah melakukan OCR (baca teks) atau
  deskripsi suasana.
- YOLO (ONNX + onnxruntime) berjalan terus-menerus di background, mendeteksi
  **banyak kelas** secara kontinu tanpa memerlukan LLM.
- Analisis/ucapan LLM dilakukan **on-demand** sesuai kebutuhan (melalui tombol `f`
  atau saat voice/chat LLM memanggil tool).
- Proactive (auto-announce) **opsional** via config.

## Prinsip Arsitektur (MASTERPLAN)

- **Satu pipeline** — semua request lewat `core/gemma_pipeline.py`.
- **Camera exclusive** — `karen_mcp/servers/vision_server.py` satu-satunya yang
  membuka `cv2.VideoCapture` (sekali). YOLO & semua tool membaca frame terbaru dari
  shared state, bukan membuka kamera baru.
- **No image storage** — frame hanya di memori, tidak disimpan ke disk.
- **Config terpusat** — semua path, port, threshold di `config.py`.

## Diagram

```
            main.py (orkestrator)
             ├── r  → voice → pipeline.query_llm (chat + MCP tool calling)
             ├── f  → pipeline.vision_scan()        → MCP analisa_foto → TTS
             ├── c  → reset history
             ├── q  → quit
             └── loop: poll (YOLO_POLL_INTERVAL) pipeline.detect_objects()
                     → jika kelas target (YOLO_TARGET_CLASSES) baru muncul → TTS

karen_mcp/servers/vision_server.py        ← SATU-SATUNYA pemegang kamera
  ├── _CameraThread (daemon, long-lived)
  │     ├── cv2.VideoCapture(CAMERA_INDEX)   # dibuka SEKALI
  │     ├── read frame → _latest_frame        (shared, lock)
  │     └── yolo_loop: tiap YOLO_INFER_INTERVAL
  │            run yolov8n.onnx (onnxruntime, CPU) → _latest_detections
  ├── tools:
  │     ├── cek_sekitar    (deskripsi, reuse frame terbaru)
  │     ├── baca_teks      (OCR, reuse frame terbaru)
  │     ├── analisa_foto   (LLM putuskan OCR vs deskripsi) ← dipakai tombol f
  │     └── deteksi_objek  (tanpa LLM: label+confidence semua kelas)
```

## Perubahan File

### 1. `config.py`

```python
YOLO_MODEL_PATH          = "models/yolo/yolov26n.onnx"
YOLO_INPUT_SIZE          = 640
YOLO_CONF_THRESHOLD      = 0.5
YOLO_IOU_THRESHOLD       = 0.45
YOLO_INFER_INTERVAL      = 0.5        # detik antar inference
YOLO_CLASSES_FILTER      = []         # [] = semua kelas; isi utk filter
YOLO_TARGET_CLASSES      = []         # [] = auto-announce dimatikan
YOLO_POLL_INTERVAL       = 1.0
YOLO_ANNOUNCE_COOLDOWN   = 15.0
CAMERA_INDEX             = 0          # (sudah ada)
```

### 2. `karen_mcp/servers/vision_server.py` (restrukturisasi)

- Tambah `_CameraThread` (daemon, long-lived): buka camera sekali, loop baca frame →
  simpan `_latest_frame` (guarded `threading.Lock`).
- Tambah `_yolo_loop`: tiap `YOLO_INFER_INTERVAL`, jalankan inference `yolov8n.onnx`
  via `onnxruntime` pada frame terbaru; post-process (decode output, NMS, filter
  `YOLO_CLASSES_FILTER`) → simpan `_latest_detections`.
- Ganti `_capture_frame()` di semua tool menjadi `_latest_frame()` (tidak buka lagi).
- Tool baru:
  - `analisa_foto()` → frame terbaru + LLM prompt "LLM putuskan OCR vs deskripsi" →
    `{"hasil": text}`.
  - `deteksi_objek()` → `{"detections": [{"label","confidence","timestamp"}...]}` dari
    `_latest_detections` (bisa dipanggil via tombol/polling maupun LLM tool calling).
- Init: load model onnx & start thread di `if __name__ == "__main__":` sebelum
  `server.run()`.

### 3. `core/gemma_pipeline.py`

- Hapus `vision_query()` (duplikat; logika vision ada di vision server).
- Tambah wrapper tipis ke MCP (pakai `self._loop.run_until_complete` +
  `self._mcp_host.call_tool`):
  - `vision_scan()`     → call MCP `analisa_foto`, parse `{"hasil": ...}` → str.
  - `detect_objects()`  → call MCP `deteksi_objek`, parse → list detections.
- Tambah rule di `query_llm` system prompt: objek/deteksi sekitar → panggil
  `deteksi_objek`.

### 4. `main.py`

- Hapus `capture_frame()` (tidak lagi akses kamera langsung).
- Key `f` → `text = pipeline.vision_scan()`; jika tidak kosong →
  `speaker.synthesize(...)` + `speaker.play_audio(..., get_char_fn=kb.get_char)`
  (pola sama key `r`, tangani aksi `q`/`r`).
- Proactive (jika `YOLO_TARGET_CLASSES` non-empty): di loop, throttle
  `YOLO_POLL_INTERVAL`; skip saat `pygame.mixer.music.get_busy()`; trigger saat
  kelas target transisi tidak-ada → ada; cooldown `YOLO_ANNOUNCE_COOLDOWN`.
- Update log kontrol → `r/f/c/q`.

### 5. `MASTERPLAN.md`

- Update diagram `karen-vision`: tambah `analisa_foto`, `deteksi_objek`, YOLO thread.
- Pertegas prinsip camera-exclusive (YOLO = pemegang kamera tunggal).
- Update baris kontrol `r/f/c/q` & daftar kelas YOLO.

### 6. Model & Dependency

- `onnxruntime` sudah menjadi dependency; tidak perlu torch.
- Unduh `yolov8n.onnx` → `models/yolo/yolov8n.onnx` (Ultralytics / Netron).
- Gunakan label COCO 80 kelas untuk post-process.

## Pemahaman Utama

> "YOLO memberitahu banyak hal, tapi nanti LLM request-nya sesuai kebutuhan."

- Default: YOLO mendeteksi **semua kelas** terus-menerus dan tersedia via
  `deteksi_objek` secara real-time, tanpa LLM.
- LLM/ucapan hanya on-demand (tombol `f`, chat/voice yang memanggil tool).
- Auto-announce **off** secara default (`YOLO_TARGET_CLASSES=[]`). Untuk
  mengaktifkan, isi mis. `["person"]`.

## Verifikasi

1. `uv run python -m py_compile main.py core/gemma_pipeline.py config.py \
   karen_mcp/servers/vision_server.py`
2. Test manual: `uv run main.py`
   - Pastikan `r`, `f`, `c`, `q` berfungsi.
   - Dengan `CAMERA_INDEX` ada, tekan `f` → foto dianalisa LLM → TTS.
   - Dengan `YOLO_TARGET_CLASSES=["person"]`: uji deteksi orang → auto-announce.
</parameter>