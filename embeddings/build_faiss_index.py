import json
import os
import faiss
import numpy as np

from embedding_utils import EmbeddingModel


DATA_PATH = "data/shl_catalog_raw.json"
INDEX_DIR = "embeddings/faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")


def build_document_text(assessment):
    """
    Combine assessment fields into a single semantic document
    """
    parts = [
        f"Assessment Name: {assessment.get('name', '')}",
        f"Description: {assessment.get('description', '')}",
        f"Test Type: {assessment.get('test_type', '')}",
        f"Duration: {assessment.get('duration', '')}",
        f"Remote Support: {assessment.get('remote_support', '')}",
        f"Adaptive Support: {assessment.get('adaptive_support', '')}",
    ]
    return "\n".join(parts)


def main():
    print("📥 Loading SHL catalogue...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    print(f"✅ Assessments loaded: {len(assessments)}")

    documents = []
    metadata = []

    for a in assessments:
        documents.append(build_document_text(a))
        metadata.append({
            "assessment_id": a.get("assessment_id"),
            "name": a.get("name"),
            "url": a.get("url"),
            "test_type": a.get("test_type"),
            "duration": a.get("duration"),
            "remote_support": a.get("remote_support"),
            "adaptive_support": a.get("adaptive_support")
        })

    print("🧠 Generating embeddings (SentenceTransformers)...")
    embedder = EmbeddingModel()
    embeddings = embedder.embed_texts(documents)

    dim = embeddings.shape[1]
    print(f"📐 Embedding dimension: {dim}")

    print("⚡ Building FAISS index...")
    index = faiss.IndexFlatIP(dim)  # cosine similarity (with normalized vectors)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("✅ FAISS index built successfully")
    print(f"📁 Index saved at: {INDEX_PATH}")
    print(f"📁 Metadata saved at: {META_PATH}")


if __name__ == "__main__":
    main()
