#!/usr/bin/env python3
"""
main.py — KAREN-Vision entry point.
Orchestrates: llama-server, MCP host, VAD recording, TTS playback.

Controls:
  r  → record & submit voice
  c  → clear conversation history
  q  → quit
"""

import logging
import select
import sys
import termios
import time
import tty

import config
from core.audio import load_vad, record_speech
from core.gemma_pipeline import GemmaPipeline
from core.speak_queue import SpeakQueue

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("voice_assistant.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("karen.main")


class KeyboardInput:
    """Raw-mode keyboard reader for non-blocking single-char input."""

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

    def get_char(self) -> str | None:
        if sys.stdin.isatty():
            if select.select([sys.stdin], [], [], 0.02)[0]:
                return sys.stdin.read(1)
        return None


def _log_step(step: int, total: int, message: str):
    log.info(f"Loading [{step}/{total}] {message}")


def main():
    pipeline = GemmaPipeline()
    speaker = SpeakQueue()
    kb = KeyboardInput()

    try:
        _log_step(1, 3, "Starting Gemma4 server + MCP tools...")
        pipeline.start()

        _log_step(2, 3, "Loading TTS model...")
        speaker.load_model()

        _log_step(3, 3, "Loading VAD model...")
        _vad, vad_iterator = load_vad()

        input_audio = "input_record.wav"
        output_audio = "output_response.wav"

        kb.enable_raw()
        log.info("KAREN ready. Press 'r' to record, 'c' to clear history, 'q' to quit.")

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
                record_result = record_speech(vad_iterator, input_audio, kb.get_char)
                if record_result == "quit":
                    break
                if record_result != "ok":
                    continue

                # ASR
                transcribed = pipeline.speech_to_text(input_audio)
                if not transcribed.strip():
                    log.warning("Transcribed text is empty.")
                    time.sleep(1)
                    continue

                # LLM + tool calling
                response_text = pipeline.query_llm(transcribed)

                # TTS
                speaker.synthesize(response_text, output_audio)
                action = speaker.play_audio(output_audio, get_char_fn=kb.get_char)

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
        pipeline.stop()
        log.info("KAREN stopped.")


if __name__ == "__main__":
    main()
