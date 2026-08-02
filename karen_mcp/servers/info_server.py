#!/usr/bin/env python3
"""
MCP Server: Organisasi KIR Info
Tools: cari_info_organisasi, get_semua_info_organisasi — query data/organisasi.json.
Run standalone:  python -m karen_mcp.servers.info_server
"""

import json
import os
import re

from mcp.server.fastmcp import FastMCP

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "organisasi.json")

server = FastMCP(
    "karen-info",
    instructions="Menyediakan informasi organisasi KIR MAN 2 Kota Bogor.",
)


def _load_data():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def _flatten(data, prefix=""):
    chunks = []
    if isinstance(data, dict):
        for k, v in data.items():
            label = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                chunks.extend(_flatten(v, label))
            else:
                chunks.append({"key": label, "value": str(v)})
    elif isinstance(data, list):
        for i, item in enumerate(data):
            label = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                chunks.extend(_flatten(item, label))
            else:
                chunks.append({"key": label, "value": str(item)})
    return chunks


@server.tool()
def cari_info_organisasi(query: str) -> dict:
    """Mencari informasi tentang organisasi KIR MAN 2 Kota Bogor.
    Gunakan ini jika ditanya tentang sejarah, visi misi, struktur pengurus,
    prestasi, riset berjalan, kegiatan, atau apapun tentang KIR MAN 2 Kota Bogor."""
    data = _load_data()
    chunks = _flatten(data)
    q = query.lower().strip()

    keywords = re.split(r"[\s,]+", q)
    matched = []
    for chunk in chunks:
        text = f"{chunk['key']} {chunk['value']}".lower()
        score = sum(1 for kw in keywords if kw and kw in text)
        if score > 0:
            matched.append((score, chunk))

    matched.sort(key=lambda x: -x[0])
    results = [{"topik": m[1]["key"], "informasi": m[1]["value"]} for m in matched[:10]]

    if not results:
        return {"hasil": [], "pesan": f"Tidak menemukan informasi tentang '{query}' di data organisasi."}

    return {"hasil": results, "pesan": f"Ditemukan {len(results)} informasi terkait '{query}'."}


@server.tool()
def get_semua_info_organisasi() -> dict:
    """Mengambil seluruh data organisasi KIR MAN 2 Kota Bogor secara lengkap."""
    return _load_data()


if __name__ == "__main__":
    server.run(transport="stdio")
