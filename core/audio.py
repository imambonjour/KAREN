"""
core/audio.py — Voice Activity Detection + Microphone Recording.
Extracted from gemma4.py. Uses Silero VAD + pw-record.
"""

import logging
import subprocess
import time
import wave

import numpy as np
import onnxruntime as ort

import config

log = logging.getLogger("karen.audio")


class SileroVAD:
    """Silero VAD v5 ONNX wrapper."""
    CONTEXT_SIZE = 64

    def __init__(self, model_path: str):
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
        self._sr = np.array(config.SAMPLE_RATE, dtype=np.int64)

    def __call__(self, chunk: np.ndarray) -> float:
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        x = np.concatenate([self._context, chunk], axis=1)
        out, state = self.session.run(
            None,
            {"input": x, "state": self._state, "sr": self._sr},
        )
        self._state = state
        self._context = x[:, -self.CONTEXT_SIZE:]
        return float(out[0, 0])


class StreamingVADIterator:
    """Streaming iterator that emits start/end events for speech segments."""

    def __init__(self, model: SileroVAD):
        self.model = model
        self.threshold = config.VAD_THRESHOLD
        self.sampling_rate = config.SAMPLE_RATE
        self.min_silence_samples = round(
            config.SAMPLE_RATE * config.VAD_MIN_SILENCE_MS / 1000
        )
        self.speech_pad_samples = round(
            config.SAMPLE_RATE * config.VAD_SPEECH_PAD_MS / 1000
        )
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def process(self, chunk: np.ndarray) -> dict | None:
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


def load_vad() -> tuple[SileroVAD, StreamingVADIterator]:
    """Load VAD model and return (model, iterator)."""
    log.info("Loading Silero VAD model...")
    vad = SileroVAD(config.VAD_MODEL_PATH)
    iterator = StreamingVADIterator(vad)
    log.info("Silero VAD loaded successfully.")
    return vad, iterator


def record_speech(vad_iterator: StreamingVADIterator, output_path: str, get_char_fn) -> str | None:
    """Record audio from mic via pw-record, using VAD to detect speech boundaries.

    Args:
        vad_iterator: Initialized StreamingVADIterator.
        output_path: Path to write the captured WAV file.
        get_char_fn: Callable returning a char or None (for keyboard input).

    Returns:
        "ok" if speech captured, "quit" if user pressed q, None if cancelled/no speech.
    """
    log.info("Recording started, listening for speech...")

    cmd = ["pw-record", "--channels=1", "--rate", str(config.SAMPLE_RATE), "--format=s16", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    vad_iterator.reset_states()

    bytes_per_window = config.VAD_WINDOW_SAMPLES * 2
    audio_chunks = []
    speech_start_sample = None
    speech_end_sample = None
    quit_requested = False
    cancelled = False

    try:
        while True:
            char = get_char_fn()
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

            result = vad_iterator.process(samples)
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
    min_samples = round(config.SAMPLE_RATE * config.VAD_MIN_SPEECH_MS / 1000)
    if len(trimmed) < min_samples:
        log.info("Speech too short, ignored.")
        time.sleep(1)
        return None

    pcm = np.clip(trimmed * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(config.SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())

    log.info("Captured speech successfully.")
    return "ok"
