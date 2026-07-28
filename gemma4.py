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

import config
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("voice_assistant.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("voice_assistant_gemma4")

LLAMA_SERVER_PATH = config.LLAMA_SERVER_PATH
GEMMA4_MODEL_PATH = config.GEMMA4_MODEL_PATH
GEMMA4_MMPROJ_PATH = config.GEMMA4_MMPROJ_PATH
GEMMA4_MTP_PATH = config.GEMMA4_MTP_PATH
TTS_MODEL_PATH = config.TTS_MODEL_PATH
VAD_MODEL_PATH = config.VAD_MODEL_PATH
GEMMA4_PORT = config.GEMMA4_PORT
GEMMA4_CHAT_URL = config.GEMMA4_CHAT_URL
SAMPLE_RATE = config.SAMPLE_RATE
VAD_WINDOW_SAMPLES = config.VAD_WINDOW_SAMPLES
VAD_THRESHOLD = config.VAD_THRESHOLD
VAD_MIN_SILENCE_MS = config.VAD_MIN_SILENCE_MS
VAD_SPEECH_PAD_MS = config.VAD_SPEECH_PAD_MS
VAD_MIN_SPEECH_MS = config.VAD_MIN_SPEECH_MS
MAX_HISTORY_TURNS = config.MAX_HISTORY_TURNS

# ---------------------------------------------------------------------------
# Tools (diload dari folder tools/ via registry)
# ---------------------------------------------------------------------------

TOOLS_SCHEMA = get_schemas()
FUNCTION_MAP = get_function_map()


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
        self.gemma4_proc = None
        self.tts_voice = None
        self.vad = None
        self.vad_iterator = None
        self.conversation_history = []  # list of {role, content} untuk conversation memory
        pygame.mixer.init()

    def _ensure_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    def _wait_for_server(self, proc, port, name, timeout_seconds=120):
        log.info(f"Waiting for {name} on port {port}...")
        for _ in range(timeout_seconds):
            try:
                resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                if resp.status_code == 200:
                    log.info(f"{name} is ready.")
                    return
            except requests.RequestException:
                pass
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                log.error(f"{name} failed to start (exit code {proc.returncode})")
                log.error(f"STDOUT: {stdout}")
                log.error(f"STDERR: {stderr}")
                raise RuntimeError(f"{name} failed to start.")
            time.sleep(1)
        raise RuntimeError(f"{name} did not become ready within {timeout_seconds} seconds.")

    def _start_server(self, cmd, port, name):
        log.info(f"Starting {name}: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._wait_for_server(proc, port, name)
        return proc

    def start_gemma4_server(self):
        self._ensure_file(LLAMA_SERVER_PATH)
        if not os.path.exists(GEMMA4_MODEL_PATH):
            log.info(f"{GEMMA4_MODEL_PATH} not found. Downloading from Hugging Face...")
            os.makedirs(os.path.dirname(GEMMA4_MODEL_PATH), exist_ok=True)
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="unsloth/gemma-4-E2B-it-GGUF",
                filename="gemma-4-E2B-it-UD-Q4_K_XL.gguf",
                local_dir=os.path.dirname(GEMMA4_MODEL_PATH),
                local_dir_use_symlinks=False
            )
        self._ensure_file(GEMMA4_MODEL_PATH)
        self._ensure_file(GEMMA4_MMPROJ_PATH)
        threads = max(1, (os.cpu_count() or 4) - 1)
        cmd = [
            LLAMA_SERVER_PATH,
            "-m",
            GEMMA4_MODEL_PATH,
            "--mmproj",
            GEMMA4_MMPROJ_PATH,
            "--port",
            str(GEMMA4_PORT),
            "--media-path",
            ".",
            "--mlock",
            "--no-mmap",
            "-t",
            str(threads),
            "-c",
            "8192",
            "--jinja",
            "--no-webui",
        ]
        self.gemma4_proc = self._start_server(cmd, GEMMA4_PORT, "Gemma4 llama-server")

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
        """Transcribe audio using Gemma4's native audio understanding via chat endpoint."""
        log.info(f"Running Gemma4 native ASR on: {audio_path}")
        start_time = time.time()

        with open(audio_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode("ascii")

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah mesin transkripsi audio otomatis. "
                        "Tugasmu hanya menuliskan teks dari apa yang terdengar di dalam audio secara verbatim (kata per kata). "
                        "JANGAN menjawab pertanyaan, JANGAN memberikan salam, JANGAN memberikan penjelasan tambahan. "
                        "Hanya tulis teks dari suara yang ada di audio tersebut."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data,
                                "format": "wav",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Tuliskan apa yang diucapkan dalam audio ini secara verbatim.",
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        response = requests.post(GEMMA4_CHAT_URL, json=payload, timeout=120)
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"]
        duration = time.time() - start_time
        text = self._clean_response(raw_text)
        log.info(f'ASR transcription [{duration:.2f}s]: "{text}"')
        return text

    def _detect_tool_intent(self, text: str):
        """
        Fallback: deteksi intent tool dari teks jika model tidak memanggil tool sendiri.
        Mengembalikan list tool_calls dalam format yang kompatibel, atau None.
        """
        import re as _re
        t = text.lower().strip()

        def make_tool(name, args=None):
            return [{
                "id": f"fallback_{name}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args or {})
                }
            }]

        # ---------------------------------------------------------------
        # 1. HITUNG TOTAL — "berapa orang/pendaftar", "siapa saja yang daftar"
        # ---------------------------------------------------------------
        total_keywords = ["berapa", "total", "jumlah", "banyak", "siapa saja"]
        registration_keywords = ["daftar", "pendaftar", "peserta", "mendaftar", "terdaftar"]

        if any(kw in t for kw in total_keywords) and \
           any(kw in t for kw in registration_keywords) and \
           not _re.search(r"dari\s+\S|sekolah\s+\S|dari\s+(man|sma|smk|smp|sd|mts|ma)", t):
            return make_tool("hitung_total_pendaftar")

        # ---------------------------------------------------------------
        # 2. CARI BY NAMA — "apakah X sudah mendaftar", "cek X", dll.
        # ---------------------------------------------------------------
        patterns_nama = [
            r"apakah\s+(.+?)\s+sudah(?:\s+mendaftar)?",
            r"sudahkah\s+(.+?)\s+mendaftar",
            r"apakah\s+ada\s+(.+?)\s*(?:yang\s+(?:sudah\s+)?mendaftar|mendaftar|$)",
            r"cari\s+(?:nama\s+)?(.+?)\s*(?:sudah|mendaftar|$)",
            r"cek\s+(?:nama\s+)?(.+?)\s*(?:sudah|mendaftar|daftar|$)",
        ]
        for pat in patterns_nama:
            m = _re.search(pat, t)
            if m:
                nama = m.group(1).strip().title()
                # Hindari false positive: kata generik bukan nama orang
                skip_words = {"dia", "mereka", "siapa", "itu", "ini", "seseorang", "orang"}
                if nama and len(nama) > 1 and nama.lower() not in skip_words:
                    return make_tool("cari_by_nama", {"nama": nama})

        # ---------------------------------------------------------------
        # 3. CARI BY SEKOLAH/KOTA
        # Contoh: "dari Bandung", "dari MAN 2", "sekolah X siapa", "ada berapa dari X"
        # ---------------------------------------------------------------
        patterns_sekolah = [
            # "siapa saja dari X" / "dari X siapa"
            r"dari\s+(.+?)\s+(?:siapa|yang)",
            r"siapa.*?dari\s+(.+?)(?:\s*\?|$)",
            # "berapa orang/peserta dari X" / "ada berapa dari X"
            r"(?:berapa|ada\s+berapa)\s+(?:orang|peserta|pendaftar)?\s*(?:yang\s+)?dari\s+(.+?)(?:\s*[\?\.]|$)",
            # "sekolah X siapa"
            r"sekolah\s+(.+?)\s+(?:siapa|yang|ada)",
            # Nama sekolah langsung: MAN/SMA/SMK/SMP/SD/MTs/MA + nomor/nama
            r"((?:man|sma|smk|smp|sd|mts|ma)\s+\d*\s*\w+(?:\s+\w+)?)",
        ]
        for pat in patterns_sekolah:
            m = _re.search(pat, t)
            if m:
                sekolah = m.group(1).strip().title()
                if sekolah and len(sekolah) > 2:
                    return make_tool("cari_by_sekolah", {"sekolah": sekolah})

        # ---------------------------------------------------------------
        # 4. CEK CUACA — "cuaca di X", "bagaimana cuaca X", "suhu di X"
        # ---------------------------------------------------------------
        patterns_cuaca = [
            r"cuaca\s+(?:di\s+|di\s+kota\s+)?(.+?)(?:\s*[\?\.]|$)",
            r"bagaimana\s+cuaca\s+(?:di\s+)?(.+?)(?:\s*[\?\.]|$)",
            r"suhu\s+(?:di\s+)?(.+?)(?:\s*[\?\.]|$)",
            r"hujan\s+(?:tidak\s+|enggak\s+|gak\s+)?(?:di\s+)?(.+?)(?:\s*[\?\.]|$)",
            r"panas\s+(?:tidak\s+|enggak\s+)?(?:di\s+)?(.+?)(?:\s*[\?\.]|$)",
        ]
        for pat in patterns_cuaca:
            m = _re.search(pat, t)
            if m:
                kota = m.group(1).strip().title()
                skip = {"sini", "sana", "situ", "hari", "ini", "sekarang", "besok"}
                if kota and len(kota) > 1 and kota.lower() not in skip:
                    return make_tool("cek_cuaca", {"kota": kota})

        # ---------------------------------------------------------------
        # 5. CARI WEB — pertanyaan umum yang tidak ada di database lokal
        # ---------------------------------------------------------------
        web_triggers = [
            r"apa\s+(?:itu\s+|yang\s+dimaksud\s+)?(.+?)(?:\s*[\?\.]|$)",
            r"siapa\s+(?:itu\s+)?(.+?)(?:\s*[\?\.]|$)",
            r"kapan\s+(.+?)(?:\s*[\?\.]|$)",
            r"di\s+mana\s+(.+?)(?:\s*[\?\.]|$)",
            r"berapa\s+(?:harga\s+|biaya\s+)?(.+?)(?:\s*[\?\.]|$)",
            r"cara\s+(.+?)(?:\s*[\?\.]|$)",
            r"kenapa\s+(.+?)(?:\s*[\?\.]|$)",
            r"cari\s+(?:info\s+|informasi\s+)?(.+?)(?:\s*[\?\.]|$)",
        ]
        # Hindari trigger web search untuk pertanyaan database yang sudah tertangkap
        db_keywords = {"pendaftar", "daftar", "mendaftar", "peserta", "registrasi"}
        if not any(kw in t for kw in db_keywords):
            for pat in web_triggers:
                m = _re.search(pat, t)
                if m:
                    query = m.group(1).strip()
                    if query and len(query) > 2:
                        return make_tool("cari_web", {"query": text.strip()})

        return None

    def _append_to_history(self, user_text: str, assistant_text: str):
        """Menambahkan satu pasang giliran ke history dan menerapkan rolling window."""
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": assistant_text})
        # Rolling window: batasi jumlah pasang yang disimpan
        max_entries = MAX_HISTORY_TURNS * 2  # user + assistant per pasang
        if len(self.conversation_history) > max_entries:
            self.conversation_history = self.conversation_history[-max_entries:]
        log.info(f"History updated: {len(self.conversation_history) // 2}/{MAX_HISTORY_TURNS} turns stored.")

    def reset_history(self):
        """Menghapus seluruh riwayat percakapan."""
        self.conversation_history.clear()
        log.info("Conversation history cleared.")

    def query_llm(self, user_text):
        log.info("Sending prompt to Gemma4 via llama-server...")
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
        # Bangun messages: system + history + pesan user saat ini
        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
            {"role": "user", "content": user_text},
        ]
        log.info(f"Chat history: {len(self.conversation_history) // 2}/{MAX_HISTORY_TURNS} turns")

        start_time = time.time()
        response = requests.post(
            GEMMA4_CHAT_URL,
            json={
                "messages": messages,
                "tools": TOOLS_SCHEMA,
                "tool_choice": "auto",
                "temperature": 0.1,
                "max_tokens": 200,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        tool_calls = message.get("tool_calls")

        # Fallback: jika model tidak memanggil tool padahal seharusnya,
        # deteksi intent dari teks dan panggil tool secara manual.
        if not tool_calls:
            tool_calls = self._detect_tool_intent(user_text)
            if tool_calls:
                log.info(f"[FALLBACK] Detected {len(tool_calls)} tool intent(s) from text.")
                message = {"role": "assistant", "content": None, "tool_calls": tool_calls}

        if tool_calls:
            log.info(f"LLM requested {len(tool_calls)} tool call(s).")
            messages.append(message)

            for call in tool_calls:
                func_name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {}

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

                tool_call_id = call.get("id", f"call_{func_name}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(hasil, default=str, ensure_ascii=False),
                    }
                )

            follow_up = requests.post(
                GEMMA4_CHAT_URL,
                json={
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 200,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=30,
            )
            follow_up.raise_for_status()
            raw_response_text = follow_up.json()["choices"][0]["message"]["content"].strip()
        else:
            raw_response_text = message["content"].strip()

        duration = time.time() - start_time
        response_text = self._clean_response(raw_response_text)
        log.info(f'LLM response (Raw)  [{duration:.2f}s]: "{raw_response_text}"')
        log.info(f'LLM response (Clean) [{duration:.2f}s]: "{response_text}"')
        # Simpan giliran ini ke history
        self._append_to_history(user_text, response_text)
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

        cmd = ["pw-record", "--channels=1", "--rate", str(SAMPLE_RATE), "--format=s16", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
            stderr_output = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            if stderr_output:
                log.warning(f"pw-record stderr: {stderr_output.strip()}")

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
        if self.gemma4_proc:
            log.info("Terminating Gemma4 llama-server...")
            self.gemma4_proc.terminate()
            try:
                self.gemma4_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.gemma4_proc.kill()
            log.info("Gemma4 llama-server stopped.")


def _print_loading_log(step, total, message):
    log.info(f"Loading [{step}/{total}] {message}")


def main():
    pipeline = VoiceAssistantPipeline()
    kb = KeyboardInput()

    try:
        _print_loading_log(1, 3, "Starting Gemma4 server (LLM + ASR)...")
        pipeline.start_gemma4_server()

        _print_loading_log(2, 3, "Loading TTS model...")
        pipeline.load_tts_model()

        _print_loading_log(3, 3, "Loading VAD model...")
        pipeline.load_vad()

        input_audio = "input_record.wav"
        output_audio = "output_response.wav"

        kb.enable_raw()

        trigger_record = False
        last_c_time = 0.0
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
                now = time.monotonic()
                if now - last_c_time > 0.5:
                    last_c_time = now
                    pipeline.reset_history()
                continue
            if char_lower == "r":
                record_result = pipeline.record_audio(input_audio, kb)
                if record_result == "quit":
                    break
                if record_result != "ok":
                    continue

                success = pipeline.run_pipeline(input_audio, output_audio)
                if not success:
                    time.sleep(1)
                    continue

                action = pipeline.play_audio_and_listen(output_audio, kb)
                if action == "q":
                    break
                if action == "r":
                    trigger_record = True
                    continue
            time.sleep(0.05)

    except Exception as e:
        log.error(f"An error occurred: {e}", exc_info=True)
    finally:
        kb.disable_raw()
        pipeline.cleanup()


if __name__ == "__main__":
    main()
