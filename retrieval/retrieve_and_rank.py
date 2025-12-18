import os
import faiss
import json
from embeddings.embedding_utils import EmbeddingModel

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "embeddings", "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")
CATALOG_PATH = os.path.join(BASE_DIR, "data", "shl_catalog_raw.json")

_embedding_model = None


def build_index():
    global _embedding_model

    os.makedirs(INDEX_DIR, exist_ok=True)

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    texts = [
        f"{item['name']} {item.get('description', '')}"
        for item in catalog
    ]

    if _embedding_model is None:
        _embedding_model = EmbeddingModel()

    embeddings = _embedding_model.embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    return index, catalog


def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata

    # Build index if missing (Streamlit-safe)
    return build_index()
