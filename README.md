# KAREN (Karya Asisten Registrasi & Edukasi Nirkabel) - Voice Assistant

Proyek ini adalah asisten suara pintar berbasis AI Bahasa Indonesia yang mengintegrasikan deteksi suara (VAD), pengenalan wicara (ASR), pemrosesan bahasa alami (LLM), dan sintesis suara (TTS). Asisten ini dirancang untuk dapat berinteraksi secara interaktif untuk menjawab pertanyaan pendaftaran, cuaca, pencarian web, serta menyapa pendaftar secara otomatis (Welcome Greeter).

---

## 🛠️ Arsitektur & Fitur Utama

Proyek ini memiliki beberapa komponen utama yang bekerja secara bersinergi:

### 1. Sistem Deteksi Suara (Voice Activity Detection - VAD)
*   Menggunakan **Silero VAD** (`models/silero_vad.onnx`) yang dijalankan secara lokal via ONNX Runtime.
*   Secara otomatis memotong audio rekaman ketika pengguna selesai berbicara (berdasarkan ambang kesunyian/silence threshold).

### 2. Sintesis Suara (Text-to-Speech - TTS)
*   Menggunakan **Piper TTS** (`models/piper/id_ID-news_tts-medium.onnx`) untuk menghasilkan suara asisten yang alami dalam Bahasa Indonesia.

### 3. Dua Model Jalur Pipeline Asisten (LLM + ASR)
Proyek menyediakan dua pilihan modul utama untuk berinteraksi:
*   **Pipeline Gemini (`gemini.py`)**:
    *   **ASR**: Menggunakan **pywhispercpp** secara lokal (mengunduh model Whisper `base`).
    *   **LLM**: Menggunakan **Gemini API** via Google GenAI SDK (model `gemini-3.1-flash-lite`).
*   **Pipeline Gemma4 (`main.py`)**:
    *   **ASR & LLM**: Menjalankan server LLM lokal menggunakan **Gemma 4** (`models/Gemma4/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf`) via `llama-server`. Menggunakan kemampuan pemahaman audio bawaan (Native Audio Understanding) untuk mentranskripsikan suara secara langsung.

### 4. Welcome Greeter (`greeter.py`)
*   Script latar belakang (background worker) yang memeriksa tabel pendaftaran (`registrations`) di **Supabase** setiap 5 detik.
*   Jika ada pendaftar baru, sistem akan menyapa mereka secara langsung menggunakan suara (Piper TTS).

### 5. Modul Alat Asisten (Tool Registry / Function Calling)
AI dibekali kemampuan memanggil fungsi eksternal (`tools/`) secara dinamis sesuai konteks:
*   **Pencarian Database (`tools/database_search.py`)**: Terhubung ke database Supabase untuk mencari pendaftar berdasarkan nama, sekolah, atau menghitung total pendaftar secara fleksibel (`cari_pendaftar`, `hitung_total_pendaftar`).
*   **Pencarian Web (`tools/web_search.py`)**: Digunakan ketika pengguna menanyakan informasi umum di luar database lokal.
*   **Registrasi Alat (`tools/registry.py`)**: Menggabungkan skema dan pemetaan fungsi secara dinamis untuk dikonsumsi LLM.

---

## 📁 Struktur Direktori

```bash
KAREN/
├── gemini.py              # Pipeline Asisten menggunakan Gemini API + pywhispercpp
├── main.py                # Pipeline Asisten menggunakan Gemma 4 Lokal (llama-server)
├── greeter.py             # Otomatisasi sapaan pendaftar baru dari database
├── database-search.py     # CLI tool untuk pengujian query Supabase secara manual
├── tools/                 # Modul tool/fungsi eksternal untuk LLM
│   ├── registry.py        # Penggabung skema & fungsi alat
│   ├── database_search.py # Integrasi query database Supabase
│   └── web_search.py      # Pencarian informasi web luar
├── models/                # Folder penyimpanan model local (VAD, Piper, Gemma)
├── .env                   # Variabel lingkungan (Supabase & API Key)
└── pyproject.toml         # Konfigurasi dependensi project (uv / pip)
```

---

## 🚀 Cara Menjalankan

### Persiapan Lingkungan
1. Buat file `.env` dan lengkapi konfigurasi berikut:
   ```env
   SUPABASE_URL="https://your-supabase-url.supabase.co"
   SUPABASE_SERVICE_ROLE_KEY="your-supabase-key"
   GEMINI_API_KEY="your-gemini-api-key"
   ```
2. Pastikan file model ONNX (VAD & Piper) serta GGUF (Gemma4) diletakkan di dalam folder `models/`.

### Menjalankan Asisten Suara
*   **Menggunakan Gemini (Cloud LLM + Local ASR)**:
    ```bash
    python gemini.py
    ```
*   **Menggunakan Gemma 4 (Full Local)**:
    ```bash
    python main.py
    ```
*   *Penggunaan:* Tekan tombol `r` pada keyboard untuk mulai berbicara, dan asisten akan otomatis memproses setelah Anda selesai berbicara. Tekan `q` untuk keluar.

### Menjalankan Welcome Greeter
Untuk menyapa pendaftar baru yang masuk ke database secara otomatis:
```bash
python greeter.py
```
