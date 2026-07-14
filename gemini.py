import base64
import json
import logging
import os
import re
import select
import subprocess
import sys
import termios
import time
import tty
import wave

import numpy as np
import onnxruntime as ort
import pygame
import requests
from piper.voice import PiperVoice
from tools.registry import get_schemas, get_function_map

from visualizer import VoiceScope

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("voice_assistant.log", mode="a"),
    ],
)
log = logging.getLogger("voice_assistant_gemini")

TTS_MODEL_PATH = "models/piper/id_ID-news_tts-medium.onnx"
VAD_MODEL_PATH = "models/silero_vad.onnx"
SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 500
VAD_SPEECH_PAD_MS = 30
VAD_MIN_SPEECH_MS = 250

MAX_HISTORY_TURNS = 10  # jumlah pasang (user+assistant) yang disimpan dalam satu sesi

TOOLS_SCHEMA = get_schemas()
FUNCTION_MAP = get_function_map()

# Pass python functions directly to Gemini tools
gemini_tools = list(FUNCTION_MAP.values())

class SileroVAD:
    CONTEXT_SIZE = 64

    def __init__(self, model_path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.reset_states()

    def reset_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SIZE), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def __call__(self, chunk):
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        x = np.concatenate([self._context, chunk], axis=1)
        out, state = self.session.run(
            None,
            {"input": x, "state": self._state, "sr": self._sr},
        )
        self._state = state
        self._context = x[:, -self.CONTEXT_SIZE :]
        return float(out[0, 0])


class StreamingVADIterator:
    def __init__(
        self,
        model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    ):
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_samples = round(sampling_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = round(sampling_rate * speech_pad_ms / 1000)
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def process(self, chunk):
        window_size_samples = len(chunk)
        self.current_sample += window_size_samples
        speech_prob = self.model(chunk)

        if speech_prob >= self.threshold and self.temp_end:
            self.temp_end = 0

        if speech_prob >= self.threshold and not self.triggered:
            self.triggered = True
            speech_start = max(
                0,
                self.current_sample - self.speech_pad_samples - window_size_samples,
            )
            return {"start": int(speech_start)}

        if speech_prob < self.threshold - 0.15 and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
            self.temp_end = 0
            self.triggered = False
            return {"end": int(speech_end)}

        return None


class KeyboardInput:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def enable_raw(self):
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def disable_raw(self):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            self.old_settings = None

    def get_char(self):
        if sys.stdin.isatty():
            if select.select([sys.stdin], [], [], 0.02)[0]:
                return sys.stdin.read(1)
        return None


class VoiceAssistantPipeline:
    def __init__(self):
        self.tts_voice = None
        self.vad = None
        self.vad_iterator = None
        self.asr_model = None
        self.client = genai.Client()
        self.chat = None  # persistent chat session untuk conversation history
        pygame.mixer.init()

    def _ensure_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    def load_asr_model(self):
        log.info("Loading pywhispercpp ASR model (base)...")
        from pywhispercpp.model import Model
        # This will automatically download the ggml 'base' model if not present.
        self.asr_model = Model('base', print_realtime=False, print_progress=False)

    def load_tts_model(self):
        log.info("Loading Piper TTS voice model...")
        self.tts_voice = PiperVoice.load(TTS_MODEL_PATH)
        log.info("Piper TTS loaded successfully.")

    def load_vad(self):
        log.info("Loading Silero VAD model...")
        self.vad = SileroVAD(VAD_MODEL_PATH)
        self.vad_iterator = StreamingVADIterator(self.vad)
        log.info("Silero VAD loaded successfully.")

    def _clean_response(self, text):
        text = re.sub(r"<\|[^>]+?\|>", "", text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"</?think>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.strip('"').strip("'").strip()

    def speech_to_text(self, audio_path):
        """Transcribe audio using pywhispercpp."""
        log.info(f"Running ASR using pywhispercpp on: {audio_path}")
        start_time = time.time()

        try:
            segments = self.asr_model.transcribe(audio_path, language="id")
            text = " ".join([segment.text for segment in segments]).strip()
        except Exception as e:
            log.error(f"ASR error: {e}")
            text = ""
            
        text = self._clean_response(text)
        duration = time.time() - start_time
        log.info(f'ASR transcription [{duration:.2f}s]: "{text}"')
        return text

    def _get_or_create_chat(self):
        """Mengembalikan chat session yang ada, atau membuat yang baru jika belum ada."""
        if self.chat is None:
            system_prompt = (
                "Kamu adalah asisten suara AI pintar Bahasa Indonesia. bernama KAREN.\n"
                "WAJIB: Semua jawaban dalam Bahasa Indonesia.\n"
                "JANGAN PERNAH mencampurkan bahasa lain.\n"
                "\n"
                "SANGAT PENTING - ATURAN TOOL CALL:\n"
                "Kamu memiliki tools berikut, WAJIB digunakan sesuai konteks:\n"
                "- mencari informasi pendaftar (berdasarkan nama atau sekolah atau keduanya) → panggil cari_pendaftar\n"
                "- berapa jumlah/total pendaftar → panggil hitung_total_pendaftar\n"
                "- cuaca, suhu, hujan di suatu kota → panggil cek_cuaca\n"
                "- pertanyaan umum, berita, fakta, pengetahuan → panggil cari_web\n"
                "- pertanyaan terkait organisasi KIR → panggil cari_info_organisasi\n"
                "JANGAN jawab sendiri tanpa tool jika informasi bisa dicari.\n"
                "JANGAN bilang 'mohon berikan konteks' jika sudah ada kata kunci.\n"
                "\n"
                "Jawab singkat dan padat dalam 1-2 kalimat setelah mendapat hasil tool."
            )
            self.chat = self.client.chats.create(
                model="gemini-3.1-flash-lite",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    max_output_tokens=200,
                    tools=gemini_tools
                )
            )
            log.info("New Gemini chat session created.")
        return self.chat

    def reset_history(self):
        """Menghapus seluruh riwayat percakapan dan memulai sesi chat baru."""
        self.chat = None
        log.info("Chat history cleared. Sesi baru akan dimulai pada pesan berikutnya.")

    def query_llm(self, user_text):
        log.info("Sending prompt to Gemini...")
        start_time = time.time()

        chat = self._get_or_create_chat()
        history_len = len(chat.get_history()) // 2  # pasang turn
        log.info(f"Chat history: {history_len}/{MAX_HISTORY_TURNS} turns")

        # Jika history sudah penuh, hapus turn paling lama dengan membuat sesi baru
        # yang membawa max_history_turns terakhir sebagai seed
        if history_len >= MAX_HISTORY_TURNS:
            log.info("History limit reached, trimming oldest turns...")
            old_history = chat.get_history()
            # Ambil N pasang turn terakhir (tiap pasang = 2 entry: user + model)
            keep = MAX_HISTORY_TURNS * 2
            trimmed_history = old_history[-keep:] if len(old_history) > keep else old_history
            system_prompt = (
                "Kamu adalah asisten suara AI pintar Bahasa Indonesia. bernama KAREN.\n"
                "WAJIB: Semua jawaban dalam Bahasa Indonesia.\n"
                "JANGAN PERNAH mencampurkan bahasa lain.\n"
                "\n"
                "SANGAT PENTING - ATURAN TOOL CALL:\n"
                "Kamu memiliki tools berikut, WAJIB digunakan sesuai konteks:\n"
                "- mencari informasi pendaftar (berdasarkan nama atau sekolah atau keduanya) → panggil cari_pendaftar\n"
                "- berapa jumlah/total pendaftar → panggil hitung_total_pendaftar\n"
                "- cuaca, suhu, hujan di suatu kota → panggil cek_cuaca\n"
                "- pertanyaan umum, berita, fakta, pengetahuan → panggil cari_web\n"
                "- pertanyaan terkait organisasi KIR → panggil cari_info_organisasi\n"
                "- jika gak ada data yg ketemu dengan filter pakai get_semua_info_organisasi lalu simpulkan sesuai dengan pertanyaan\n"
                "JANGAN jawab sendiri tanpa tool jika informasi bisa dicari.\n"
                "JANGAN bilang 'mohon berikan konteks' jika sudah ada kata kunci.\n"
                "\n"
                "Jawab singkat dan padat dalam 1-2 kalimat setelah mendapat hasil tool."
            )
            self.chat = self.client.chats.create(
                model="gemini-3.1-flash-lite",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    max_output_tokens=200,
                    tools=gemini_tools
                ),
                history=trimmed_history,
            )
            chat = self.chat

        response = chat.send_message(user_text)

        # Check if a tool was called
        if response.function_calls:
            log.info(f"Gemini requested {len(response.function_calls)} tool call(s).")
            tool_responses = []
            for function_call in response.function_calls:
                func_name = function_call.name
                args = {}
                if function_call.args:
                    args = {k: v for k, v in function_call.args.items()}

                log.info(f"[TOOL CALL] {func_name}({args})")
                func = FUNCTION_MAP.get(func_name)
                if func:
                    try:
                        hasil = func(**args)
                    except Exception as e:
                        log.error(f"Tool {func_name} error: {e}")
                        hasil = {"error": str(e)}
                else:
                    hasil = {"error": f"Fungsi {func_name} tidak ditemukan"}

                tool_responses.append(
                    types.Part.from_function_response(
                        name=func_name,
                        response={"result": hasil}
                    )
                )

            # Send the tool responses back to Gemini (tetap dalam sesi yang sama)
            follow_up = chat.send_message(tool_responses)
            response_text = follow_up.text.strip()
        else:
            if response.text:
                response_text = response.text.strip()
            else:
                response_text = "Maaf, saya tidak mengerti."

        duration = time.time() - start_time
        response_text = self._clean_response(response_text)
        log.info(f'LLM response [{duration:.2f}s]: "{response_text}"')
        return response_text

    def text_to_speech(self, text, output_path):
        log.info(f"Synthesizing speech via Piper TTS to: {output_path}")
        start_time = time.time()
        with wave.open(output_path, "wb") as wav_file:
            self.tts_voice.synthesize_wav(text, wav_file)
        duration = time.time() - start_time
        log.info(f"TTS synthesis complete [{duration:.2f}s].")

    def run_pipeline(self, input_audio_path, output_audio_path):
        pipeline_start = time.time()

        asr_start = time.time()
        transcribed_text = self.speech_to_text(input_audio_path)
        asr_time = time.time() - asr_start

        if not transcribed_text.strip():
            log.warning("Transcribed text is empty.")
            return False

        llm_start = time.time()
        response_text = self.query_llm(transcribed_text)
        llm_time = time.time() - llm_start

        tts_start = time.time()
        self.text_to_speech(response_text, output_audio_path)
        tts_time = time.time() - tts_start

        total_time = time.time() - pipeline_start

        log.info(
            f"Pipeline executed successfully. Total: {total_time:.2f}s "
            f"(ASR: {asr_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s)"
        )
        return True

    def record_audio(self, output_path, kb_input):
        log.info("Recording started, listening for speech...")

        cmd = ["pw-record", "--channels=1", "--rate", str(SAMPLE_RATE), "--format=s16", "-a", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.vad_iterator.reset_states()

        bytes_per_window = VAD_WINDOW_SAMPLES * 2
        audio_chunks = []
        speech_start_sample = None
        speech_end_sample = None
        quit_requested = False
        cancelled = False

        try:
            while True:
                char = kb_input.get_char()
                if char in ("\n", "\r"):
                    cancelled = True
                    break
                if char and char.lower() == "q":
                    quit_requested = True
                    break

                chunk_bytes = proc.stdout.read(bytes_per_window)
                if not chunk_bytes:
                    break
                if len(chunk_bytes) < bytes_per_window:
                    chunk_bytes += b"\x00" * (bytes_per_window - len(chunk_bytes))

                samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                audio_chunks.append(samples)

                result = self.vad_iterator.process(samples)
                if result and "start" in result:
                    speech_start_sample = result["start"]
                    log.info("Speech detected.")
                if result and "end" in result:
                    speech_end_sample = result["end"]
                    log.info("Speech ended. Submitting...")
                    break
        finally:
            proc.terminate()
            proc.wait()

        if quit_requested:
            return "quit"
        if cancelled:
            log.info("Recording cancelled by user.")
            return None
        if speech_start_sample is None or speech_end_sample is None:
            log.info("No speech detected.")
            time.sleep(1)
            return None

        full_audio = np.concatenate(audio_chunks)
        trimmed = full_audio[speech_start_sample:speech_end_sample]
        min_samples = round(SAMPLE_RATE * VAD_MIN_SPEECH_MS / 1000)
        if len(trimmed) < min_samples:
            log.info("Speech too short, ignored.")
            time.sleep(1)
            return None

        pcm = np.clip(trimmed * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())

        log.info("Captured speech successfully.")
        return "ok"

    def play_audio_and_listen(self, audio_path, kb_input):
        if not os.path.exists(audio_path):
            return None

        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        log.info("Playing response audio...")

        interrupted_by = None
        while pygame.mixer.music.get_busy():
            char = kb_input.get_char()
            if char:
                char_lower = char.lower()
                if char_lower in ("r", "q"):
                    interrupted_by = char_lower
                    pygame.mixer.music.stop()
                    break
            time.sleep(0.05)

        while kb_input.get_char():
            pass
        return interrupted_by

    def cleanup(self):
        pass


def _print_loading_log(step, total, message):
    log.info(f"Loading [{step}/{total}] {message}")


def main():
    pipeline = VoiceAssistantPipeline()
    kb = KeyboardInput()
    scope = VoiceScope()
    scope.start()

    try:
        _print_loading_log(1, 3, "Loading ASR model...")
        pipeline.load_asr_model()

        _print_loading_log(2, 3, "Loading TTS model...")
        pipeline.load_tts_model()

        _print_loading_log(3, 3, "Loading VAD model...")
        pipeline.load_vad()

        input_audio = "input_record.wav"
        output_audio = "output_response.wav"

        kb.enable_raw()
        scope.set_flat()

        trigger_record = False
        while True:
            if trigger_record:
                char_lower = "r"
                trigger_record = False
            else:
                char = kb.get_char()
                char_lower = char.lower() if char else None

            if char_lower == "q":
                break
            if char_lower == "c":
                pipeline.reset_history()
                continue
            if char_lower == "r":
                scope.set_listening()
                record_result = pipeline.record_audio(input_audio, kb)
                if record_result == "quit":
                    break
                if record_result != "ok":
                    scope.set_flat()
                    continue

                scope.set_thinking()
                success = pipeline.run_pipeline(input_audio, output_audio)
                if not success:
                    scope.set_flat()
                    time.sleep(1)
                    continue

                scope.set_speaking(output_audio)
                action = pipeline.play_audio_and_listen(output_audio, kb)
                scope.set_flat()
                if action == "q":
                    break
                if action == "r":
                    trigger_record = True
                    continue
            time.sleep(0.05)

    except Exception as e:
        log.error(f"An error occurred: {e}", exc_info=True)
    finally:
        scope.stop()
        kb.disable_raw()
        pipeline.cleanup()


if __name__ == "__main__":
    main()
