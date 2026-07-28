# CLEANUP.md — Branching KAREN → KAREN-Vision

Panduan langkah demi langkah untuk membuat cabang baru dari KAREN, khusus
untuk kasus **asisten kelas guru tunanetra** (fully offline, Gemma 4 lokal
+ modul vision). Ikuti urutan ini supaya tidak merusak KAREN yang lama.

---

## 0. Langkah awal

```bash
cp -r karen karen-vision
cd karen-vision
git checkout -b vision-branch   # kalau pakai git, jangan langsung di main
```

---

## 1. File & folder yang DIHAPUS

| Item | Alasan dibuang |
|---|---|
| `greeter.py` | Fitur sapa pendaftar baru — tidak relevan untuk kelas |
| `anime.py` | Eye-tracker MediaPipe — fitur demo, di luar scope inti |
| `visualizer.py` | VoiceScope Pygame GUI — device target (Orange Pi) belum tentu ada display, dan menambah beban CPU |
| `database-search.py` | CLI testing Supabase — tidak dipakai lagi |
| `tools/database_search.py` | Tool function-calling untuk Supabase |
| `gemini.py` (Pipeline A) | Cloud LLM — bertentangan dengan tujuan fully offline |
| `models/asr/Qwen3-ASR-*.gguf` | Sudah ditandai "belum dipakai"; Gemma 4 E2B/E4B punya native audio encoder sendiri, jadi redundan |
| `.env` entries: `SUPABASE_URL`, `SUPABASE_KEY`, `GEMINI_API_KEY` | Tidak ada lagi cloud/DB call |

```bash
rm -f greeter.py anime.py visualizer.py database-search.py
rm -f tools/database_search.py
rm -f gemini.py
rm -rf models/asr
```

---

## 2. Dependency yang DIHAPUS dari `pyproject.toml`

Cek dan hapus baris-baris berikut (sesuaikan nama persis di file Anda):

- `supabase` (client Python untuk Supabase)
- `mediapipe` (hanya dipakai `anime.py`)
- `pygame` (hanya dipakai `visualizer.py` + playback lama — ganti ke `sounddevice` atau tetap pakai kalau masih perlu playback WAV sederhana)
- `google-genai` (SDK Gemini — tidak dipakai lagi kalau full offline)
- `ddgs` — **opsional**: pertahankan kalau Anda masih mau tool `cari_web`/`cek_cuaca` untuk guru (mis. tanya cuaca sebelum kegiatan luar kelas). Kalau mau benar-benar minimal, hapus juga.

```bash
# contoh kalau pakai uv/pip
uv remove supabase mediapipe google-genai
```

---

## 3. Struktur folder BARU

```
karen-vision/
├── config.py                  # konstanta terpusat (sudah dibuat)
├── main.py                    # entry point, gabung voice loop + vision on-demand
├── speak.py                   # Piper TTS (REUSE dari KAREN, tidak diubah)
├── core/
│   ├── state.py               # SessionState gabungan (sudah dibuat)
│   ├── speak_queue.py         # antrian TTS terpusat (sudah dibuat)
│   ├── gemma_pipeline.py      # wrapper llama-server (sudah dibuat)
│   └── audio.py               # Silero VAD + record mic (REUSE dari KAREN)
├── vision/
│   ├── camera.py              # ambil 1 frame dari webcam
│   └── vision_state.py        # cooldown & mode on_demand/background
├── tools/
│   ├── registry.py            # daftar tool aktif (lihat config.ENABLED_TOOLS)
│   ├── vision_tool.py         # tool baru: cek_sekitar()
│   ├── web_search.py          # REUSE (opsional)
│   └── weather.py             # REUSE (opsional)
├── models/
│   ├── Gemma4/                # gguf + mmproj + mtp (REUSE)
│   ├── piper/                 # voice id_ID (REUSE)
│   └── silero_vad.onnx        # (REUSE)
└── logs/
```

**File yang di-REUSE apa adanya dari KAREN** (tinggal copy, tidak perlu ditulis ulang):
`speak.py`, isi folder `models/Gemma4`, `models/piper`, `silero_vad.onnx`, dan logika VAD dari `gemma4.py` lama (pindahkan ke `core/audio.py`).

---

## 4. Perubahan perilaku penting

1. **Satu pipeline saja** — semua request LLM (voice maupun vision) lewat `core/gemma_pipeline.py`. Tidak ada lagi percabangan Pipeline A vs B.
2. **Vision = tool, bukan loop terpisah** (`VISION_MODE = "on_demand"` di `config.py`) — kamera hanya diaktifkan saat LLM memanggil tool `cek_sekitar()`. Ini menghindari masalah concurrency TTS yang jadi risiko utama di SMART-VISION. Kalau nanti butuh mode monitoring pasif (`"background"`), semua permintaan bicara — baik dari voice maupun vision — **wajib** lewat `core/speak_queue.py`, tidak boleh manggil Piper langsung dari dua tempat.
3. **ASR pakai native audio Gemma 4**, bukan Whisper/Qwen3-ASR — satu model untuk teks, gambar, dan audio sekaligus, lebih hemat resource di Orange Pi.
4. **Tidak ada penyimpanan gambar/video** — `vision_state.py` hanya boleh menyimpan teks hasil deskripsi ke `event_log`, tidak pernah menyimpan frame ke disk (pertimbangan privasi, mengingat subjek adalah murid).

---

## 5. Checklist sebelum jalan pertama kali

- [ ] `llama-server` bisa start dengan `GEMMA_MODEL_PATH` + `GEMMA_MMPROJ_PATH`
- [ ] Test `gemma.chat_with_image()` dengan 1 foto contoh kelas → cek apakah deskripsi masuk akal & dalam Bahasa Indonesia
- [ ] Test `speak_queue` dengan 2 pemanggilan `.say()` beruntun → pastikan tidak tabrakan, keluar berurutan
- [ ] Ukur latensi end-to-end di Orange Pi asli (bukan cuma di laptop dev) — ini yang sering meleset dari asumsi
- [ ] Pastikan tidak ada `.env` tersisa berisi key Supabase/Gemini yang lupa dihapus

---

## 6. File yang BELUM ditulis (masih perlu dibuat)

- `main.py` — entry point
- `core/audio.py` — pindahkan logika VAD dari `gemma4.py` lama
- `vision/camera.py`, `vision/vision_state.py`
- `tools/registry.py`, `tools/vision_tool.py`

Bilang saja kalau mau saya lanjutkan salah satu dari ini.
