import json
import os
import re

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "organisasi.json")


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


def cari_info_organisasi(query: str):
    data = _load_data()
    chunks = _flatten(data)
    q = query.lower().strip()

    # Cari chunks yang cocok dengan query
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


def get_semua_info_organisasi():
    data = _load_data()
    return data


TOOLS = [
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "cari_info_organisasi",
                "description": (
                    "Mencari informasi tentang organisasi KIR MAN 2 Kota Bogor. "
                    "Gunakan ini jika ditanya tentang sejarah, visi misi, struktur pengurus, "
                    "prestasi, riset berjalan, kegiatan, atau apapun tentang KIR MAN 2 Kota Bogor."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Kata kunci atau topik yang dicari, misal 'sejarah', 'visi misi', 'pengurus', 'prestasi', 'riset opsi', 'pembina'"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        "function": cari_info_organisasi
    },
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "get_semua_info_organisasi",
                "description": "Mengambil seluruh data organisasi KIR MAN 2 Kota Bogor secara lengkap",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        "function": get_semua_info_organisasi
    }
]
