import sqlite3
import numpy as np
from pathlib import Path
from app.config import DB_PATH


def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Tabel utama: satu row per orang
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabel embeddings: satu row per sample
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS person_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            captured_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    _migrate_legacy(conn)
    conn.close()


def _migrate_legacy(conn):
    """
    Migrasi dari schema lama (persons.embedding) ke schema baru (person_embeddings).
    Aman dijalankan berulang — cek dulu apakah kolom lama masih ada.
    """
    cursor = conn.cursor()

    # Cek apakah kolom 'embedding' masih ada di persons
    cursor.execute("PRAGMA table_info(persons)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'embedding' not in columns:
        return  # sudah dimigasi sebelumnya

    print("[DB] Migrating legacy single-embedding schema to multi-embedding...")
    cursor.execute("SELECT id, name, embedding FROM persons WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()

    for person_id, name, emb_bytes in rows:
        if emb_bytes:
            # Cek apakah sudah ada embedding untuk orang ini
            cursor.execute(
                "SELECT COUNT(*) FROM person_embeddings WHERE person_id = ?", (person_id,)
            )
            count = cursor.fetchone()[0]
            if count == 0:
                cursor.execute(
                    "INSERT INTO person_embeddings (person_id, embedding) VALUES (?, ?)",
                    (person_id, emb_bytes)
                )
                print(f"  Migrated: '{name}'")

    conn.commit()

    # Hapus kolom lama (SQLite <3.35 tidak support DROP COLUMN, jadi kita rebuild)
    try:
        cursor.execute("ALTER TABLE persons DROP COLUMN embedding")
        conn.commit()
        print("  Dropped legacy 'embedding' column.")
    except sqlite3.OperationalError:
        # SQLite versi lama: biarkan kolom lama ada, tidak masalah
        pass

    try:
        cursor.execute("ALTER TABLE persons DROP COLUMN samples_count")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    print("[DB] Migration complete.")


def add_person(name: str, embedding: np.ndarray, num_samples: int = 1) -> bool:
    """
    Menambah person baru atau menambahkan embedding baru ke person yang sudah ada.
    Setiap embedding disimpan sebagai row terpisah di person_embeddings.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Pastikan persons row ada
        cursor.execute("INSERT OR IGNORE INTO persons (name) VALUES (?)", (name,))
        conn.commit()

        cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
        person_id = cursor.fetchone()[0]

        # Simpan embedding baru
        emb_bytes = embedding.astype(np.float32).tobytes()
        cursor.execute(
            "INSERT INTO person_embeddings (person_id, embedding) VALUES (?, ?)",
            (person_id, emb_bytes)
        )

        # Hitung total untuk log
        cursor.execute(
            "SELECT COUNT(*) FROM person_embeddings WHERE person_id = ?", (person_id,)
        )
        total = cursor.fetchone()[0]

        conn.commit()
        conn.close()
        print(f"Updated '{name}' in database. Added {num_samples} new samples. New total: {total} samples.")
        return True
    except Exception as e:
        print(f"Database error in add_person: {e}")
        return False


def add_person_batch(name: str, embeddings: list[np.ndarray]) -> bool:
    """
    Menambahkan banyak embeddings sekaligus (lebih efisien untuk registrasi awal).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("INSERT OR IGNORE INTO persons (name) VALUES (?)", (name,))
        conn.commit()

        cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
        person_id = cursor.fetchone()[0]

        rows = [
            (person_id, emb.astype(np.float32).tobytes())
            for emb in embeddings
        ]
        cursor.executemany(
            "INSERT INTO person_embeddings (person_id, embedding) VALUES (?, ?)", rows
        )

        cursor.execute(
            "SELECT COUNT(*) FROM person_embeddings WHERE person_id = ?", (person_id,)
        )
        total = cursor.fetchone()[0]

        conn.commit()
        conn.close()
        print(f"Registered '{name}' with {len(embeddings)} samples. Total: {total}.")
        return True
    except Exception as e:
        print(f"Database error in add_person_batch: {e}")
        return False


def get_all_persons():
    """
    Retrieves all registered persons with ALL their embeddings.
    Returns list of dict:
        {'name': str, 'embeddings': np.ndarray shape (N, 512)}
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.name, pe.embedding
            FROM persons p
            JOIN person_embeddings pe ON pe.person_id = p.id
            ORDER BY p.name, pe.id
        """)
        rows = cursor.fetchall()
        conn.close()

        persons_map: dict[str, list] = {}
        for name, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            if name not in persons_map:
                persons_map[name] = []
            persons_map[name].append(emb)

        persons = []
        for name, embs in persons_map.items():
            persons.append({
                'name': name,
                'embeddings': np.stack(embs),  # shape (N, 512)
            })
        return persons
    except Exception as e:
        print(f"Database error in get_all_persons: {e}")
        return []


def delete_person(name: str) -> bool:
    """Deletes a person and all their embeddings from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM persons WHERE name = ?", (name,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted
    except Exception as e:
        print(f"Database error in delete_person: {e}")
        return False
