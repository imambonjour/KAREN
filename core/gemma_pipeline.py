"""
core/gemma_pipeline.py — Wrapper for llama-server (Gemma 4 E2B).
Handles: server lifecycle, ASR (native audio), chat + tool calling, vision.
"""

import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import time

import requests

import config
from karen_mcp.host import MCPHost

log = logging.getLogger("karen.gemma_pipeline")


class GemmaPipeline:
    """Manages llama-server and provides ASR, chat, and vision capabilities."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._mcp_host = MCPHost()
        self._conversation_history: list[dict] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- Server lifecycle ----------------------------------------------------

    def start(self):
        """Start llama-server and MCP host."""
        self._start_llama_server()
        # Start MCP host in an event loop
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._mcp_host.start())
        log.info(f"Tools available: {[s['function']['name'] for s in self._mcp_host.tool_schemas]}")

    def stop(self):
        """Stop llama-server and MCP host."""
        if self._loop:
            try:
                self._loop.run_until_complete(self._mcp_host.stop())
            except Exception:
                log.warning("MCP host cleanup raised an error (ignored during shutdown).", exc_info=True)
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None
        if self._proc:
            log.info("Terminating llama-server...")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            log.info("llama-server stopped.")

    def _start_llama_server(self):
        self._ensure_file(config.LLAMA_SERVER_PATH)
        if not os.path.exists(config.GEMMA4_MODEL_PATH):
            log.info(f"{config.GEMMA4_MODEL_PATH} not found. Downloading from Hugging Face...")
            os.makedirs(os.path.dirname(config.GEMMA4_MODEL_PATH), exist_ok=True)
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="unsloth/gemma-4-E2B-it-GGUF",
                filename="gemma-4-E2B-it-UD-Q4_K_XL.gguf",
                local_dir=os.path.dirname(config.GEMMA4_MODEL_PATH),
                local_dir_use_symlinks=False,
            )
        self._ensure_file(config.GEMMA4_MODEL_PATH)
        self._ensure_file(config.GEMMA4_MMPROJ_PATH)

        threads = max(1, (os.cpu_count() or 4) - 1)
        cmd = [
            config.LLAMA_SERVER_PATH,
            "-m", config.GEMMA4_MODEL_PATH,
            "--mmproj", config.GEMMA4_MMPROJ_PATH,
            "--port", str(config.GEMMA4_PORT),
            "--media-path", ".",
            "--mlock", "--no-mmap",
            "-t", str(threads),
            "-c", "8192",
            "--jinja", "--no-webui",
        ]
        log.info(f"Starting llama-server: {' '.join(cmd)}")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self._wait_for_server()

    def _wait_for_server(self, timeout_seconds: int = 120):
        log.info(f"Waiting for llama-server on port {config.GEMMA4_PORT}...")
        for _ in range(timeout_seconds):
            try:
                resp = requests.get(f"http://localhost:{config.GEMMA4_PORT}/health", timeout=1)
                if resp.status_code == 200:
                    log.info("llama-server is ready.")
                    return
            except requests.RequestException:
                pass
            if self._proc.poll() is not None:
                stdout, stderr = self._proc.communicate()
                log.error(f"llama-server failed (exit {self._proc.returncode})")
                log.error(f"STDOUT: {stdout}")
                log.error(f"STDERR: {stderr}")
                raise RuntimeError("llama-server failed to start.")
            time.sleep(1)
        raise RuntimeError(f"llama-server not ready within {timeout_seconds}s")

    @staticmethod
    def _ensure_file(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # -- ASR -----------------------------------------------------------------

    def speech_to_text(self, audio_path: str) -> str:
        """Transcribe audio using Gemma 4's native audio understanding."""
        log.info(f"Running Gemma4 native ASR on: {audio_path}")
        start_time = time.time()

        with open(audio_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("ascii")

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
                        {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}},
                        {"type": "text", "text": "Tuliskan apa yang diucapkan dalam audio ini secara verbatim."},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        response = requests.post(config.GEMMA4_CHAT_URL, json=payload, timeout=120)
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"]
        text = self._clean_response(raw_text)
        log.info(f'ASR [{time.time() - start_time:.2f}s]: "{text}"')
        return text

    # -- Vision (via MCP camera-exclusive thread) ----------------------------

    def vision_scan(self) -> str:
        """Capture latest frame and analyse with LLM (OCR or description).
        Calls MCP tool 'analisa_foto' in vision_server."""
        start_time = time.time()
        try:
            result_str = self._loop.run_until_complete(
                self._mcp_host.call_tool("analisa_foto", {})
            )
            import json as _json
            data = _json.loads(result_str)
            text = data.get("hasil") or data.get("error", "")
        except Exception as e:
            log.exception("vision_scan failed")
            text = ""
        text = self._clean_response(text)
        log.info(f'vision_scan [{time.time() - start_time:.2f}s]: "{text[:80]}"')
        return text

    def detect_objects(self) -> list[dict]:
        """Return latest YOLO detections from vision_server (no LLM).
        Calls MCP tool 'deteksi_objek'."""
        try:
            result_str = self._loop.run_until_complete(
                self._mcp_host.call_tool("deteksi_objek", {})
            )
            import json as _json
            data = _json.loads(result_str)
            return data.get("detections", [])
        except Exception as e:
            log.warning(f"detect_objects failed: {e}")
            return []

    def simpan_frame(self) -> str:
        """Save the latest camera frame to temp/ (debug feature).
        Calls MCP tool 'simpan_frame'."""
        try:
            result_str = self._loop.run_until_complete(
                self._mcp_host.call_tool("simpan_frame", {})
            )
            return result_str
        except Exception as e:
            log.warning(f"simpan_frame failed: {e}")
            return ""

    # -- Chat + Tool Calling -------------------------------------------------

    def query_llm(self, user_text: str) -> str:
        """Send user text to Gemma 4 with MCP tools, handle tool calls, return final text."""
        log.info("Sending prompt to Gemma4...")
        system_prompt = (
            "Kamu adalah asisten suara AI pintar Bahasa Indonesia. bernama KAREN.\n"
            "WAJIB: Semua jawaban dalam Bahasa Indonesia.\n"
            "JANGAN PERNAH mencampurkan bahasa lain.\n"
            "\n"
            "SANGAT PENTING - ATURAN TOOL CALL:\n"
            "Kamu memiliki tools berikut, WAJIB digunakan sesuai konteks:\n"
            "- cuaca, suhu, hujan di suatu kota → panggil cek_cuaca\n"
            "- pertanyaan umum, berita, fakta, pengetahuan → panggil cari_web\n"
            "- pertanyaan terkait organisasi KIR → panggil cari_info_organisasi\n"
            "- informasi lengkap organisasi → panggil get_semua_info_organisasi\n"
            "- lihat sekitar, apa yang di depan, deskripsikan pemandangan → panggil cek_sekitar\n"
            "- baca tulisan, baca teks, apa yang tertulis → panggil baca_teks\n"
            "- objek apa di sekitar, ada apa di ruangan, deteksi benda → panggil deteksi_objek\n"
            "JANGAN jawab sendiri tanpa tool jika informasi bisa dicari.\n"
            "\n"
            "Jawab singkat dan padat dalam 1-2 kalimat setelah mendapat hasil tool."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *self._conversation_history,
            {"role": "user", "content": user_text},
        ]
        log.info(f"Chat history: {len(self._conversation_history) // 2}/{config.MAX_HISTORY_TURNS} turns")

        start_time = time.time()
        response = requests.post(
            config.GEMMA4_CHAT_URL,
            json={
                "messages": messages,
                "tools": self._mcp_host.tool_schemas,
                "tool_choice": "auto",
                "temperature": 0.1,
                "max_tokens": 200,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=180,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        tool_calls = message.get("tool_calls")

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

                # Execute via MCP host
                result_str = self._loop.run_until_complete(
                    self._mcp_host.call_tool(func_name, args)
                )

                tool_call_id = call.get("id", f"call_{func_name}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_str,
                })

            follow_up = requests.post(
                config.GEMMA4_CHAT_URL,
                json={
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 200,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=180,
            )
            follow_up.raise_for_status()
            raw_response_text = follow_up.json()["choices"][0]["message"]["content"].strip()
        else:
            raw_response_text = message["content"].strip()

        duration = time.time() - start_time
        response_text = self._clean_response(raw_response_text)
        log.info(f'LLM response [{duration:.2f}s]: "{response_text}"')

        self._append_to_history(user_text, response_text)
        return response_text

    # -- History management --------------------------------------------------

    def _append_to_history(self, user_text: str, assistant_text: str):
        self._conversation_history.append({"role": "user", "content": user_text})
        self._conversation_history.append({"role": "assistant", "content": assistant_text})
        max_entries = config.MAX_HISTORY_TURNS * 2
        if len(self._conversation_history) > max_entries:
            self._conversation_history = self._conversation_history[-max_entries:]

    def reset_history(self):
        self._conversation_history.clear()
        log.info("Conversation history cleared.")

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _clean_response(text: str) -> str:
        text = re.sub(r"<\|[^>]+?\|>", "", text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"</?think>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.strip('"').strip("'").strip()
        