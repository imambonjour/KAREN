#!/usr/bin/env python3
"""
MCP Host — manages MCP server subprocesses and provides a unified tool interface.

Usage from pipeline:
    host = MCPHost()
    await host.start()           # launches all servers, discovers tools
    schemas = host.tool_schemas  # OpenAI-compatible tool schemas for LLM
    result = await host.call_tool("cari_web", {"query": "..."})
    await host.stop()
"""

import asyncio
import json
import logging
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger("karen.mcp_host")

# Server definitions: name → command to launch via stdio
SERVER_DEFS: list[dict] = [
    {
        "name": "karen-web",
        "command": sys.executable,
        "args": ["-m", "karen_mcp.servers.web_server"],
    },
    {
        "name": "karen-info",
        "command": sys.executable,
        "args": ["-m", "karen_mcp.servers.info_server"],
    },
    {
        "name": "karen-vision",
        "command": sys.executable,
        "args": ["-m", "karen_mcp.servers.vision_server"],
    },
]


class MCPHost:
    """Launches MCP server subprocesses and exposes a unified tool interface."""

    def __init__(self, server_defs: list[dict] | None = None):
        self._defs = server_defs or SERVER_DEFS
        self._sessions: dict[str, ClientSession] = {}
        self._tool_map: dict[str, str] = {}  # tool_name → server_name
        self._tool_schemas: list[dict] = []
        self._exit_stack = AsyncExitStack()

    # -- public properties ---------------------------------------------------

    @property
    def tool_schemas(self) -> list[dict]:
        """OpenAI-compatible tool schemas (type=function) for the LLM."""
        return self._tool_schemas

    # -- lifecycle -----------------------------------------------------------

    async def start(self):
        """Launch all servers, perform handshake, and discover tools."""
        await self._exit_stack.__aenter__()

        for sdef in self._defs:
            name = sdef["name"]
            params = StdioServerParameters(
                command=sdef["command"],
                args=sdef.get("args", []),
            )
            try:
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
                session: ClientSession = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                self._sessions[name] = session

                # Discover tools
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    self._tool_map[tool.name] = name
                    self._tool_schemas.append(
                        _mcp_tool_to_openai_schema(tool)
                    )
                log.info(
                    f"MCP server '{name}' started — "
                    f"{len(tools_result.tools)} tool(s): "
                    f"{[t.name for t in tools_result.tools]}"
                )
            except Exception:
                log.exception(f"Failed to start MCP server '{name}'")

        log.info(
            f"MCPHost ready — {len(self._tool_schemas)} tools across "
            f"{len(self._sessions)} server(s)"
        )

    async def stop(self):
        """Shut down all server subprocesses."""
        await self._exit_stack.aclose()
        self._sessions.clear()
        self._tool_map.clear()
        self._tool_schemas.clear()
        log.info("MCPHost stopped.")

    # -- tool execution ------------------------------------------------------

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool by name. Returns JSON string result."""
        server_name = self._tool_map.get(tool_name)
        if not server_name:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})

        session = self._sessions.get(server_name)
        if not session:
            return json.dumps({"error": f"Server '{server_name}' not connected"})

        try:
            result = await session.call_tool(tool_name, arguments)
            # MCP returns content as a list of content blocks
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            text = "\n".join(parts) if parts else ""
            # Try to parse as JSON for cleaner downstream handling
            try:
                return json.dumps(json.loads(text), ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                return text
        except Exception as e:
            log.error(f"Tool call '{tool_name}' failed: {e}")
            return json.dumps({"error": str(e)})


def _mcp_tool_to_openai_schema(tool) -> dict:
    """Convert an MCP Tool descriptor to OpenAI function-calling schema."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
        },
    }
