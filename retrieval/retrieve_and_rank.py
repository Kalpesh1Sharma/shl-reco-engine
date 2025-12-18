import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------
# Ensure project root is on PYTHONPATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import faiss
import json
import re

from retrieval.rank_utils import balanced_rerank
from llm.query_rewriter import rewrite_query

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
ALPHA = 0.2  # keyword boost weight

INDEX_PATH = "embeddings/faiss_index/index.faiss"
META_PATH = "embeddings/faiss_index/metadata.json"

# --------------------------------------------------
# Skill synonym map
# --------------------------------------------------
SKILL_SYNONYMS = {
    "java": ["core java", "java ee", "j2ee"],
    "python": ["python programming"],
    "sql": ["database", "relational database"],
    "excel": ["spreadsheets"],
    "communication": ["verbal", "written"],
    "teamwork": ["collaboration"],
    "problem": ["analytical", "reasoning"],
}

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def expand_query(query: str) -> str:
    tokens = query.lower().split()
    expanded = set(tokens)

    for t in tokens:
        if t in SKILL_SYNONYMS:
            expanded.update(SKILL_SYNONYMS[t])

    return " ".join(expanded)


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return set(text.split())


def keyword_overlap_score(query: str, assessment: dict) -> float:
    q_tokens = tokenize(query)
    a_tokens = tokenize(assessment.get("name", ""))

    if not q_tokens:
        return 0.0

    return len(q_tokens & a_tokens) / len(q_tokens)


def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# --------------------------------------------------
# Core retrieval
# --------------------------------------------------
def retrieve(query: str, top_n: int = 30):
    # ---- Silent optional LLM rewrite ----
    rewritten = rewrite_query(query)
    final_query = rewritten if rewritten else query

    # Expand with skill synonyms
    expanded_query = expand_query(final_query)

    from embeddings.embedding_utils import EmbeddingModel
    embedder = EmbeddingModel()
    query_vec = embedder.embed_query(expanded_query)

    index, metadata = load_index()
    scores, indices = index.search(query_vec, top_n)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = metadata[idx]

        item["semantic_score"] = float(score)
        item["keyword_score"] = keyword_overlap_score(expanded_query, item)
        item["final_score"] = (
            item["semantic_score"] + ALPHA * item["keyword_score"]
        )

        results.append(item)

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


def recommend(query: str, k: int = 10):
    candidates = retrieve(query, top_n=30)
    return balanced_rerank(candidates, top_k=k)

# --------------------------------------------------
# Manual test
# --------------------------------------------------
if __name__ == "__main__":
    query = (
        "I have a JD Job Description People Science. People Answers. "
        "Looking for HR analytics and behavioral assessment tools."
    )

    recs = recommend(query)

    print("\nTop Recommendations:\n")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r['name']}")
        print(f"   Score: {round(r['final_score'], 3)}")
        print(f"   {r['url']}")
