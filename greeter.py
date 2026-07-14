import os
os.environ["SDL_AUDIODRIVER"] = "pulseaudio"

import pygame
import time
import io
import wave
from supabase import create_client
from piper import PiperVoice

from dotenv import load_dotenv
load_dotenv()


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

PIPER_MODEL = "./models/piper/id_ID-news_tts-medium.onnx"
TEMP_WAV_PATH = "greet_temp.wav"

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
voice = PiperVoice.load(PIPER_MODEL)

pygame.mixer.init()

def speak(text: str):
    # Sintesis langsung ke file (bukan BytesIO), sesuai pola yang terbukti jalan
    with wave.open(TEMP_WAV_PATH, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    pygame.mixer.music.load(TEMP_WAV_PATH)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.05)

def main():
    last_check = None

    while True:
        query = supabase.table("registrations").select("*").order("created_at", desc=False)
        if last_check:
            query = query.gt("created_at", last_check)

        response = query.execute()
        rows = response.data

        for row in rows:
            nama = row["full_name"]
            sekolah = row["school"]
            greeting = f"Halo {nama} dari {sekolah}, selamat datang di kir!"
            print(f"[GREET] {greeting}")
            speak(greeting)
            last_check = row["created_at"]

        time.sleep(5)

if __name__ == "__main__":
    main()
