"""
core/speak_queue.py — Thread-safe TTS queue using Piper.
Synthesizes speech from a queue of text items and plays them via pygame.
"""

import logging
import os
import queue
import threading
import time
import wave

import pygame

from piper.voice import PiperVoice

import config

log = logging.getLogger("karen.speak")


class SpeakQueue:
    """Thread-safe queue for TTS synthesis and playback."""

    def __init__(self):
        self._voice: PiperVoice | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        pygame.mixer.init()

    def load_model(self):
        """Load the Piper TTS voice model."""
        log.info("Loading Piper TTS voice model...")
        self._voice = PiperVoice.load(config.TTS_MODEL_PATH)
        log.info("Piper TTS loaded successfully.")

    def start(self):
        """Start the background TTS worker thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the worker thread."""
        self._running = False
        self._queue.put(None)  # sentinel
        if self._thread:
            self._thread.join(timeout=5)

    def speak(self, text: str):
        """Add text to the synthesis queue."""
        self._queue.put(text)

    def synthesize(self, text: str, output_path: str):
        """Synthesize text to a WAV file (blocking)."""
        log.info(f"Synthesizing speech to: {output_path}")
        start_time = time.time()
        with wave.open(output_path, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)
        log.info(f"TTS synthesis complete [{time.time() - start_time:.2f}s].")

    def play_audio(self, audio_path: str, get_char_fn=None) -> str | None:
        """Play a WAV file. Returns interrupt char ('r', 'q') or None."""
        if not os.path.exists(audio_path):
            return None

        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        log.info("Playing response audio...")

        interrupted_by = None
        while pygame.mixer.music.get_busy():
            if get_char_fn:
                char = get_char_fn()
                if char:
                    char_lower = char.lower()
                    if char_lower in ("r", "q"):
                        interrupted_by = char_lower
                        pygame.mixer.music.stop()
                        break
            time.sleep(0.05)

        # Drain leftover key presses
        if get_char_fn:
            while get_char_fn():
                pass
        return interrupted_by

    def _worker(self):
        """Background worker: dequeue text → synthesize → play."""
        while self._running:
            try:
                text = self._queue.get(timeout=1)
            except queue.Empty:
                continue
            if text is None:
                break
            try:
                output = "output_response.wav"
                self.synthesize(text, output)
                self.play_audio(output)
            except Exception:
                log.exception("TTS worker error")
