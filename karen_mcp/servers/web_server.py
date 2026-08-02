#!/usr/bin/env python3
"""
MCP Server: Web Search & Weather
Tools: cari_web, cek_cuaca — via DuckDuckGo.
Run standalone:  python -m karen_mcp.servers.web_server
"""

from mcp.server.fastmcp import FastMCP

server = FastMCP(
    "karen-web",
    instructions="Menyediakan pencarian web dan info cuaca via DuckDuckGo.",
)


@server.tool()
def cari_web(query: str, max_results: int = 3) -> list:
    """Mencari informasi terkini dari internet tentang berita, fakta,
    pengetahuan umum, atau pertanyaan yang tidak ada di database lokal."""
    from ddgs import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "judul": r.get("title", ""),
                "ringkasan": r.get("body", ""),
                "url": r.get("href", ""),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


@server.tool()
def cek_cuaca(kota: str) -> list | dict:
    """Mengecek cuaca terkini untuk suatu kota atau daerah."""
    from ddgs import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"cuaca terkini {kota}", max_results=2))
        if results:
            return [
                {"judul": r.get("title", ""), "ringkasan": r.get("body", "")}
                for r in results
            ]
        return {"error": "Data cuaca tidak ditemukan"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    server.run(transport="stdio")
