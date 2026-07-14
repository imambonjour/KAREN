from tools import database_search, web_search, organisasi_search

ALL_MODULES = [database_search, web_search, organisasi_search]


def get_all_tools():
    """Gabungkan semua TOOLS dari tiap modul jadi 1 list."""
    combined = []
    for module in ALL_MODULES:
        combined.extend(module.TOOLS)
    return combined


def get_schemas():
    """Ambil cuma bagian schema (dikirim ke LLM)."""
    return [t["schema"] for t in get_all_tools()]


def get_function_map():
    """Ambil mapping nama fungsi -> fungsi asli (buat eksekusi)."""
    return {t["schema"]["function"]["name"]: t["function"] for t in get_all_tools()}