import os
import json
import faiss
import numpy as np
from embeddings.embedding_utils import EmbeddingModel

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "data", "shl_catalog_raw.json")

# -------------------------------
# Globals (cached in memory)
# -------------------------------
_faiss_index = None
_metadata = None
_embedder = None


def _build_index():
    global _faiss_index, _metadata, _embedder

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    texts = [
        f"{item['name']} {item.get('description', '')}"
        for item in _metadata
    ]

    if _embedder is None:
        _embedder = EmbeddingModel()

    embeddings = _embedder.embed_texts(texts)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    _faiss_index = index


def _load_index():
    global _faiss_index, _metadata

    if _faiss_index is None or _metadata is None:
        _build_index()

    return _faiss_index, _metadata


def retrieve(query, top_n=30):
    index, metadata = _load_index()

    embedder = EmbeddingModel()
    query_vec = embedder.embed_texts([query]).astype("float32")

    scores, indices = index.search(query_vec, top_n)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        item = metadata[idx]
        results.append({
            "name": item["name"],
            "url": item["url"],
            "score": float(score)
        })

    return results


def recommend(query, k=10):
    candidates = retrieve(query, top_n=30)

    # Simple re-ranking (semantic score only for stability)
    ranked = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked[:k]
