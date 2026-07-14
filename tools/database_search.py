import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def cari_pendaftar(nama: str = None, sekolah: str = None):
    query = supabase.table("registrations").select("full_name, school, whatsapp, created_at")
    if nama:
        query = query.ilike("full_name", f"%{nama}%")
    if sekolah:
        query = query.ilike("school", f"%{sekolah}%")
    
    response = query.execute()
    return response.data


def hitung_total_pendaftar():
    response = supabase.table("registrations").select("id", count="exact").execute()
    return {"total": response.count}


# Setiap tool didaftarkan di sini: schema (untuk LLM) + fungsi asli (untuk eksekusi)
TOOLS = [
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "cari_pendaftar",
                "description": "Mencari data pendaftar. Bisa mencari berdasarkan nama, sekolah, atau keduanya. Gunakan ini jika ditanya 'siapa saja dari sekolah X', 'apakah A sudah daftar', atau 'A dari sekolah mana'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nama": {
                            "type": "string",
                            "description": "Nama pendaftar yang dicari (opsional)"
                        },
                        "sekolah": {
                            "type": "string",
                            "description": "Nama sekolah yang dicari (opsional)"
                        }
                    }
                }
            }
        },
        "function": cari_pendaftar
    },
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "hitung_total_pendaftar",
                "description": "Menghitung total jumlah pendaftar yang sudah masuk",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        "function": hitung_total_pendaftar
    }
]