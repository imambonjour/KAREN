import subprocess
import sys
import re

# --- KONFIGURASI ---
MODEL_PATH = "../models/piper/id_ID-news_tts-medium.onnx"

def text_to_speech():
    print("=== PIPER TTS GENERATOR ===")
    print("Ketik 'q' atau 'exit' untuk keluar.")
    
    while True:
        try:
            # 1. Minta input teks dari user
            user_text = input("\nMasukkan teks yang ingin diubah menjadi suara: ").strip()
            
            if user_text.lower() in ["q", "exit"]:
                print("Sampai jumpa!")
                break

            if not user_text:
                print("Teks tidak boleh kosong!")
                continue

            # 2. Buat nama file dari input user (bersihkan karakter ilegal agar aman untuk nama file)
            safe_filename = re.sub(r'[\\/*?:"<>|]', "", user_text) # Hapus karakter ilegal file
            safe_filename = safe_filename.replace(" ", "_")[:50]   # Ganti spasi & batasi max 50 karakter
            output_file = f"{safe_filename}.wav"

            print(f"Sedang memproses suara ({output_file})...")

            # 3. Jalankan perintah piper via subprocess
            process = subprocess.Popen(
                [
                    "piper", 
                    "--model", MODEL_PATH, 
                    "--output_file", output_file
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )

            stdout, stderr = process.communicate(input=user_text)

            if process.returncode == 0:
                print(f"Berhasil! File suara disimpan sebagai: {output_file}")
            else:
                print("Terjadi kesalahan saat membuat suara:")
                print(stderr)

        except FileNotFoundError:
            print("Error: Perintah 'piper' atau file model tidak ditemukan.")
            print("Pastikan Piper sudah terinstall / masuk ke PATH dan path MODEL_PATH sudah benar.")
            break
        except (KeyboardInterrupt, EOFError):
            print("\nKeluar dari program.")
            break

if __name__ == "__main__":
    text_to_speech()