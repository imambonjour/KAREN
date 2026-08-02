import numpy as np


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D embedding vectors.
    If the vectors are already L2-normalized, this is equivalent to the dot product.
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def find_best_match(query_embedding: np.ndarray, database_persons, threshold: float = 0.45):
    """
    Finds the best matching person from the database.

    database_persons: list of dict, each containing:
        - 'name': str
        - 'embeddings': np.ndarray shape (N, 512)  ← multi-embedding (new schema)
          OR 'embedding': np.ndarray shape (512,)   ← single embedding (legacy)

    Strategy: MAX-similarity — untuk tiap orang, ambil similarity tertinggi
    dari semua sample yang disimpan. Jauh lebih robust dari mean-vector matching
    karena kondisi pencahayaan/pose berbeda akan punya sample representatifnya.

    Returns:
        tuple: (matched_name, best_similarity_score)
    """
    if not database_persons:
        return "Unknown", 0.0

    best_name = "Unknown"
    global_best_sim = -1.0

    for person in database_persons:
        name = person['name']

        # Support both new multi-embedding and legacy single-embedding schema
        if 'embeddings' in person:
            embs = person['embeddings']  # shape (N, 512)
        elif 'embedding' in person:
            embs = person['embedding'].reshape(1, -1)  # shape (1, 512)
        else:
            continue

        # Vectorized: dot product dengan semua sample sekaligus
        # embs: (N, 512), query: (512,) → similarities: (N,)
        similarities = embs @ query_embedding  # assumes both are L2-normalized

        # Ambil similarity TERTINGGI dari semua sample orang ini
        person_best_sim = float(np.max(similarities))

        if person_best_sim > global_best_sim:
            global_best_sim = person_best_sim
            if person_best_sim >= threshold:
                best_name = name

    return best_name, global_best_sim
