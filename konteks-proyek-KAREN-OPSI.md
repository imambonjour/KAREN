# Konteks Proyek: KAREN & Proposal OPSI 2026 "Smart-Vision Pedagogy"

> File ini adalah ringkasan percakapan sebelumnya, dibuat untuk dijadikan konteks di chat baru.

## 1. Ringkasan Proyek

**Nama software:** KAREN (Karya Asisten Registrasi & Edukasi Nirkabel)
Asisten suara & visi berbasis AI Bahasa Indonesia untuk SBC (Raspberry Pi/Orange Pi), menggabungkan:
- Gemma 4 VLM lokal (via `llama-server`) — ASR native audio, chat, vision
- YOLO ONNX (`yolov26n.onnx`, varian **COCO object detection biasa**, bukan pose) — deteksi objek kontinu
- Face recognition terpisah (YOLOv8-Face + ArcFace) — saat ini fitur standalone (`face.py`), belum terintegrasi ke `main.py`
- Silero VAD, Piper TTS
- Arsitektur MCP (Model Context Protocol): server `karen-web`, `karen-info`, `karen-vision`

**Proposal OPSI 2026 (bidang FTR):**
**Judul:** *Smart-Vision Pedagogy: Implementasi Edge AI pada Kacamata Pintar untuk Meningkatkan Aksesibilitas dan Interaksi Guru Tunanetra dalam Pembelajaran Inklusif*
**Peneliti:** Imam Ahmad Sujiwo & Zahirah Deges Raeli Siregar, MAN 2 Kota Bogor
**Status:** Proposal sudah diunggah dan **sudah di-ACC / lolos seleksi proposal**.
**Metode:** R&D dengan ADDIE (Analysis, Design, Development, Implementation, Evaluation)
**Hardware:** Raspberry Pi 5 (16GB), baterai Li-Po 10.000mAh, webcam USB, earpiece open-ear
**Anggaran:** ± Rp 3.925.000 (di bawah batas maksimal OPSI Rp15.000.000)

## 2. Masalah Teknis yang Dibahas

- YOLO (baik YOLOv11 di proposal maupun YOLOS/paper akademis arXiv:2106.00666 yang sempat dibahas — **dua hal berbeda meski nama mirip**) bersifat **single-frame detector**, tidak bisa menyimpulkan pola dari sequence/multi-frame secara native.
- Untuk deteksi "tangan terangkat" (isyarat non-verbal siswa) dibutuhkan **pose estimation** (mis. YOLOv8-Pose, keypoint bahu/pergelangan tangan), bukan deteksi objek COCO biasa. Proyek KAREN sudah punya model pose terpisah (belum diintegrasikan ke pipeline utama).
- Solusi yang didiskusikan: heuristik geometris di atas keypoint pose (bandingkan posisi y wrist vs shoulder) + `DetectionHistory` (buffer temporal sederhana, `deque`) untuk syarat stabilitas (bukan 1 frame noise) sebelum dianggap valid — ini **zero-shot, tidak butuh training model baru**.

## 3. Keputusan Scoping (Penting)

Karena hardware terbatas (Raspberry Pi 5 tanpa GPU besar) dan waktu penelitian terbatas (~4 bulan), penelitian **sebaiknya fokus**, tidak menjalankan semua model AI sekaligus (COCO + Pose + Face Recognition + VLM berat semua kontinu).

**Hasil wawancara dengan guru tunanetra (transkrip di `wawancara.md`)** mengonfirmasi ulang prioritas:
- 🟢 **Prioritas tinggi:** OCR papan tulis/catatan (hands-free, tanpa buka HP); deteksi zona keluar-masuk pintu kelas & area meja guru (line-crossing/presence detection — secara teknis lebih murah dari pose estimation, cukup pakai model COCO yang sudah ada)
- 🟡 **Prioritas sedang:** deteksi tangan terangkat (relevan ke rumusan masalah proposal, tapi tidak disebut eksplisit oleh narasumber)
- 🔴 **Turunkan prioritas/drop:** face recognition per-siswa (bukan concern utama guru, mahal secara komputasi); deteksi menyontek (guru sendiri meragukan urgensinya — "niat nyontek mah nyontek aja... mau gurunya ngeliat")

## 4. Kepatuhan terhadap Panduan OPSI 2026 (Penting!)

Berdasarkan `Panduan_OPSI_SMA_Sederajat_2026.pdf`:

- **Pergeseran fokus fitur setelah proposal ACC itu SAH/wajar** dalam metode ADDIE — didokumentasikan lewat:
  - **Logbook penelitian** (Lampiran 9)
  - **Form respons terhadap masukan reviewer** (Lampiran 10)
  - Tidak perlu approval administratif tambahan selama substansi rumusan masalah & judul besar tidak berubah drastis.

- **Klirens Etik / Informed Consent WAJIB** untuk penelitian yang melibatkan manusia sebagai subjek (wawancara dengan guru sudah dilakukan):
  - Perlu **informed consent tertulis** (format di Lampiran 7 & 8 panduan)
  - Laporan akhir wajib mencantumkan **data responden (nama & kontak)** yang diwawancara (poin 2.7.f) — dokumen ini terpisah dari ACC proposal, disiapkan sebelum unggah laporan akhir.
  - Deadline pengunggahan laporan hasil penelitian: **31 Agustus 2026**

- **Aturan Penggunaan AI (poin 2.12.3, ketat):**
  - AI **hanya boleh** untuk mencari referensi, memahami topik, memantik ide.
  - **Substansi analisis, hasil, dan penulisan proposal/laporan harus murni karya peserta** — TIDAK boleh dibuatkan AI.
  - Wajib mengisi **Surat Pernyataan Penggunaan AI** (Lampiran 2) — sebutkan aplikasi AI yang dipakai & untuk keperluan apa.
  - Panitia akan **menguji penggunaan AI** pada naskah hasil penelitian.
  - Implikasi: draft narasi/paragraf yang dihasilkan dari sesi chat dengan AI (termasuk chat ini) **tidak boleh langsung disalin** ke proposal/laporan resmi — harus ditulis ulang dengan kata-kata sendiri oleh peserta.

- **Struktur Laporan Akhir** (Lampiran 5): Halaman Sampul, Abstrak, Daftar Isi, BAB 1 Pendahuluan, BAB 2 Landasan Teori & Studi Pustaka, BAB 3 Metode Penelitian, BAB 4 Hasil dan Pembahasan, BAB 5 Anggaran Biaya dan Kegiatan, BAB 5 (harusnya 6) Kesimpulan dan Saran, Ucapan Terima Kasih, Pernyataan Penggunaan AI, Daftar Pustaka, Lampiran (termasuk realisasi RAB).

- Uji similaritas maksimal 30% (plagiarisme), naskah > 20 halaman di luar lampiran, format PDF maks 8MB.

## 5. Isu yang Perlu Ditindaklanjuti

1. **Segera siapkan informed consent** untuk guru narasumber yang sudah diwawancara (Lampiran 7/8 template ada di panduan) — retroaktif karena wawancara sudah terjadi sebelum dokumen ini dibuat.
2. **Isi Surat Pernyataan Penggunaan AI** secara jujur (Lampiran 2) — sebutkan Claude/AI dipakai untuk: mencari ide, memahami konsep teknis (pose estimation, temporal tracking), evaluasi metode, TIDAK untuk menulis substansi laporan akhir.
3. Belum diketahui detail kasus **teman yang "ditegur"** panitia — user menyebut ini sebagai alasan merasa proyeknya "aman", tapi konteks kasusnya belum dijelaskan lebih lanjut. Perlu klarifikasi lebih lanjut jika relevan untuk mitigasi risiko serupa.
4. Implementasi teknis yang masih perlu digarap: `DetectionHistory` buffer + integrasi model pose (jika lanjut prioritas tangan terangkat), atau logic zona/line-crossing untuk deteksi pintu & meja guru (prioritas lebih tinggi berdasarkan wawancara).

## 6. File Referensi yang Sudah Diupload User

- `README.md`, `AGENTS.md`, `main.py` — dokumentasi/kode proyek KAREN
- `Opsi_2026.pdf` — draft proposal (di /mnt/project/, hanya bisa dibaca bukan diedit)
- `2106_00666v3.pdf` — paper YOLOS (referensi akademis, dikonfirmasi BEDA dari YOLOv11 yang dipakai proyek)
- `wawancara.md` — transkrip wawancara dengan guru tunanetra (narasumber, MAN 2 Kota Bogor)
- `Panduan_OPSI_SMA_Sederajat_2026.pdf` — panduan resmi OPSI dari Puspresnas Kemendikdasmen
