from ddgs import DDGS

ddgs = DDGS(api_url="http://localhost:4479", spawn_api=True)

def cari_web(query: str, max_results: int = 3) -> list:
    """Cari informasi umum dari internet menggunakan DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"judul": r.get("title", ""), "ringkasan": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


def cek_cuaca(kota: str) -> dict:
    """Cek cuaca terkini untuk sebuah kota."""
    try:
        with DDGS() as ddgs:
            result = ddgs.weather(kota)
        if result:
            w = result
            return {
                "kota": w.get("location", kota),
                "suhu_celsius": w.get("temperature", "?"),
                "kondisi": w.get("condition", "?"),
                "kelembaban_persen": w.get("humidity", "?"),
                "angin_kmh": w.get("wind_speed", "?"),
                "terasa_seperti": w.get("feels_like", "?"),
            }
        return {"error": "Data cuaca tidak ditemukan"}
    except Exception as e:
        return {"error": str(e)}


TOOLS = [
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "cari_web",
                "description": (
                    "Mencari informasi terkini dari internet tentang berita, fakta, "
                    "pengetahuan umum, atau pertanyaan yang tidak ada di database lokal"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Kata kunci atau pertanyaan yang ingin dicari"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Jumlah hasil pencarian (default 3)",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        "function": cari_web
    },
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "cek_cuaca",
                "description": "Mengecek cuaca terkini untuk suatu kota atau daerah",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kota": {
                            "type": "string",
                            "description": "Nama kota yang ingin dicek cuacanya, misal 'Bogor', 'Jakarta'"
                        }
                    },
                    "required": ["kota"]
                }
            }
        },
        "function": cek_cuaca
    }
]
