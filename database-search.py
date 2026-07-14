import os
import sys
import json
import argparse
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def cari_by_sekolah(sekolah: str):
    response = supabase.table("registrations") \
        .select("full_name, school, whatsapp, created_at") \
        .ilike("school", f"%{sekolah}%") \
        .execute()
    return response.data


def cari_by_nama(nama: str):
    response = supabase.table("registrations") \
        .select("full_name, school, whatsapp, created_at") \
        .ilike("full_name", f"%{nama}%") \
        .execute()
    return response.data


def hitung_total_pendaftar():
    response = supabase.table("registrations").select("id", count="exact").execute()
    return {"total": response.count}


# Registry biar gampang dipanggil by nama fungsi (dipakai nanti pas connect ke LLM)
AVAILABLE_FUNCTIONS = {
    "cari_by_sekolah": cari_by_sekolah,
    "cari_by_nama": cari_by_nama,
    "hitung_total_pendaftar": hitung_total_pendaftar,
}


def main():
    parser = argparse.ArgumentParser(description="Test pencarian data pendaftar")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_sekolah = subparsers.add_parser("cari_by_sekolah")
    p_sekolah.add_argument("sekolah", type=str)

    p_nama = subparsers.add_parser("cari_by_nama")
    p_nama.add_argument("nama", type=str)

    subparsers.add_parser("hitung_total_pendaftar")

    args = parser.parse_args()

    if args.command == "cari_by_sekolah":
        hasil = cari_by_sekolah(args.sekolah)
    elif args.command == "cari_by_nama":
        hasil = cari_by_nama(args.nama)
    elif args.command == "hitung_total_pendaftar":
        hasil = hitung_total_pendaftar()

    print(json.dumps(hasil, indent=2, default=str))


if __name__ == "__main__":
    main()