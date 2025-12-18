"""
DEPRECATED / EXPERIMENTAL

Gemini embeddings were explored but NOT used in the final system
due to free-tier quota and bulk embedding limitations.

Kept only for experimentation / future work reference.
"""

import sys
import os

# --------------------------------------------------
# Ensure project root is on PYTHONPATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import json
import faiss
import numpy as np

from embeddings.gemini_embedding_utils import GeminiEmbeddingModel


DATA_PATH = "data/shl_catalog_raw.json"
INDEX_DIR = "embeddings/faiss_index_gemini"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")


def build_document_text(a):
    return f"""
    Assessment Name: {a.get('name', '')}
    Description: {a.get('description', '')}
    Test Type: {a.get('test_type', '')}
    """


def main():
    print("📥 Loading SHL catalogue...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    print(f"✅ Assessments loaded: {len(assessments)}")

    documents = [build_document_text(a) for a in assessments]

    print("🧠 Generating Gemini embeddings...")
    embedder = GeminiEmbeddingModel()
    embeddings = embedder.embed_texts(documents)

    dim = embeddings.shape[1]
    print(f"📐 Embedding dimension: {dim}")

    print("⚡ Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    print("✅ Gemini FAISS index built successfully")
    print(f"📁 Index saved at: {INDEX_PATH}")


if __name__ == "__main__":
    main()
