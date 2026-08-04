"""
core/gemini_pipeline.py — Gemini API cloud backend untuk KAREN.
Handles: ASR (multimodal audio), Chat + MCP tool calling, Vision (multimodal image).

Sama seperti GemmaPipeline tetapi menggunakan Google Gemini API
(google-genai SDK) sebagai ganti llama-server lokal.
"""

import asyncio
import base64
import json
import logging
import re
import time

import config
from karen_mcp.host import MCPHost

log = logging.getLogger("karen.gemini_pipeline")

# System prompt — identik dengan GemmaPipeline agar perilaku konsisten
_SYSTEM_PROMPT = (
    "Kamu adalah asisten suara AI pintar Bahasa Indonesia bernama KAREN.\n"
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


class GeminiPipeline:
    """Gemini API cloud backend — ASR, Chat + MCP tool calling, Vision."""

    def __init__(self):
        self._client = None
        self._mcp_host = MCPHost()
        self._conversation_history: list[dict] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._gemini_tools: list[dict] = []  # Skema tools dalam format Gemini

    # -- Lifecycle -----------------------------------------------------------

    def start(self):
        """Inisialisasi Gemini client dan MCP host."""
        import os
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY tidak ditemukan di environment. "
                "Tambahkan ke file .env atau export ke shell."
            )

        self._client = genai.Client(api_key=api_key)
        log.info("Gemini client initialized.")

        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._mcp_host.start())

        # Konversi OpenAI-schema dari MCP host ke format Gemini function declaration
        self._gemini_tools = self._convert_tools_to_gemini(
            self._mcp_host.tool_schemas
        )
        log.info(
            f"Tools available: {[t['name'] for t in self._gemini_tools]}"
        )

    def stop(self):
        """Shutdown MCP host dan event loop."""
        if self._loop:
            self._loop.run_until_complete(self._mcp_host.stop())
            self._loop.close()
            self._loop = None
        log.info("GeminiPipeline stopped.")

    # -- Tool schema conversion ----------------------------------------------

    @staticmethod
    def _convert_tools_to_gemini(openai_schemas: list[dict]) -> list[dict]:
        """Convert OpenAI-compatible function schemas to Gemini tool format."""
        tools = []
        for schema in openai_schemas:
            fn = schema.get("function", {})
            tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return tools

    # -- ASR -----------------------------------------------------------------

    def speech_to_text(self, audio_path: str) -> str:
        """Transcribe audio file menggunakan Gemini multimodal (native audio input)."""
        log.info(f"Running Gemini ASR on: {audio_path}")
        start_time = time.time()

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        from google.genai import types

        response = self._client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                (
                    "Tuliskan secara verbatim (kata per kata) apa yang diucapkan "
                    "dalam audio ini dalam Bahasa Indonesia. "
                    "HANYA tulis teks ucapan, tanpa pengantar atau penjelasan tambahan."
                ),
            ],
        )

        raw_text = response.text or ""
        text = self._clean_response(raw_text)
        log.info(f'ASR [{time.time() - start_time:.2f}s]: "{text}"')
        return text

    # -- Vision (via MCP camera-exclusive thread) ----------------------------

    def vision_scan(self) -> str:
        """Ambil frame terbaru dari kamera via MCP, kirim ke Gemini Vision."""
        log.info("vision_scan: fetching latest frame via MCP analisa_foto...")
        start_time = time.time()

        try:
            # Ambil frame dari vision_server via MCP tool analisa_foto
            result_str = self._loop.run_until_complete(
                self._mcp_host.call_tool("analisa_foto", {})
            )
            data = json.loads(result_str)
        except Exception:
            log.exception("vision_scan: failed to get frame from MCP")
            return ""

        # Jika MCP vision server sudah mengembalikan analisis teks dari LLM,
        # gunakan hasilnya langsung (hemat satu round-trip ke Gemini).
        # Kalau error atau tidak ada gambar, return empty.
        if "error" in data:
            log.warning(f"vision_scan: MCP error: {data['error']}")
            return ""

        # MCP analisa_foto sudah memanggil Gemma 4 Vision di vision_server.
        # Di sini kita ambil hasilnya dan kembalikan teks.
        # (Untuk alternatif: intercept frame b64 dan kirim ke Gemini Vision
        # langsung — lihat vision_scan_direct() di bawah)
        text = data.get("hasil", "")
        text = self._clean_response(text)
        log.info(f'vision_scan [{time.time() - start_time:.2f}s]: "{text[:80]}"')
        return text

    def vision_scan_direct(self, frame_b64: str) -> str:
        """Kirim gambar base64 JPEG langsung ke Gemini Vision (bypass MCP VLM).
        Gunakan ini jika llama-server tidak berjalan dan ingin full-cloud."""
        from google.genai import types

        jpeg_bytes = base64.b64decode(frame_b64)
        response = self._client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                (
                    "Lihat gambar ini dengan seksama. "
                    "Jika terdapat teks/tulisan yang signifikan, bacakan dan transkripsikan "
                    "seluruh isi teks secara verbatim dalam Bahasa Indonesia. "
                    "Jika tidak ada teks yang signifikan, deskripsikan objek dan suasana "
                    "yang terlihat secara ringkas. "
                    "Jawab hanya dengan hasil analisis, tanpa pengantar."
                ),
            ],
        )
        return self._clean_response(response.text or "")

    # -- Chat + MCP Tool Calling ---------------------------------------------

    def query_llm(self, user_text: str) -> str:
        """Kirim teks ke Gemini dengan MCP tools, tangani tool calls, return respons akhir."""
        log.info("Sending prompt to Gemini...")
        start_time = time.time()

        # Bangun contents: system prompt + history + user turn
        contents = [{"role": "user", "parts": [_SYSTEM_PROMPT]}]

        for msg in self._conversation_history:
            role = msg["role"]
            content = msg["content"]
            if role in ("user", "model"):
                contents.append({"role": role, "parts": [content]})

        contents.append({"role": "user", "parts": [user_text]})

        log.info(
            f"Chat history: {len(self._conversation_history) // 2}/{config.MAX_HISTORY_TURNS} turns"
        )

        from google.genai import types

        # Konversi tool list ke format yang diharapkan Gemini SDK
        gemini_tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=t.get("parameters"),
                    )
                    for t in self._gemini_tools
                ]
            )
        ] if self._gemini_tools else []

        response = self._client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                temperature=0.1,
                max_output_tokens=256,
            ),
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts if candidate.content else []

        # Periksa apakah ada function calls
        function_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]

        if function_calls:
            log.info(f"Gemini requested {len(function_calls)} tool call(s).")

            # Tambahkan respons model (yang berisi function_call) ke contents
            contents.append({"role": "model", "parts": [{"function_call": {
                "name": fc.function_call.name,
                "args": dict(fc.function_call.args or {}),
            }} for fc in function_calls]})

            # Eksekusi setiap tool via MCP
            tool_results = []
            for part in function_calls:
                fc = part.function_call
                func_name = fc.name
                args = dict(fc.args or {})
                log.info(f"[TOOL CALL] {func_name}({args})")

                result_str = self._loop.run_until_complete(
                    self._mcp_host.call_tool(func_name, args)
                )
                tool_results.append({
                    "function_response": {
                        "name": func_name,
                        "response": {"result": result_str},
                    }
                })

            # Kirim balik tool results ke Gemini untuk respons akhir
            contents.append({"role": "user", "parts": tool_results})

            follow_up = self._client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=256,
                ),
            )
            raw_response = follow_up.text or ""
        else:
            # Respons langsung tanpa tool call
            raw_response = "".join(
                p.text for p in parts if hasattr(p, "text") and p.text
            )

        duration = time.time() - start_time
        response_text = self._clean_response(raw_response)
        log.info(f'LLM response [{duration:.2f}s]: "{response_text}"')

        self._append_to_history(user_text, response_text)
        return response_text

    # -- Detect Objects (YOLO via MCP, no LLM) -------------------------------

    def detect_objects(self) -> list[dict]:
        """Return latest YOLO detections from vision_server (no LLM).
        Calls MCP tool 'deteksi_objek'."""
        try:
            result_str = self._loop.run_until_complete(
                self._mcp_host.call_tool("deteksi_objek", {})
            )
            data = json.loads(result_str)
            return data.get("detections", [])
        except Exception as e:
            log.warning(f"detect_objects failed: {e}")
            return []

    # -- History Management --------------------------------------------------

    def _append_to_history(self, user_text: str, assistant_text: str):
        self._conversation_history.append({"role": "user", "content": user_text})
        self._conversation_history.append({"role": "model", "content": assistant_text})
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
