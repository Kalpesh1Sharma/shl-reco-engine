import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
from retrieval.retrieve_and_rank import recommend
from evaluation.utils import recall_at_k, extract_slug


TRAIN_DATA_PATH = "Gen_AI Dataset.xlsx"
TOP_K = 10


def main():
    print("📥 Loading labeled train dataset...")
    df = pd.read_excel(TRAIN_DATA_PATH)

    if "Query" not in df.columns or "Assessment_url" not in df.columns:
        raise ValueError("❌ Train dataset must contain Query & Assessment_url")

    recalls = []

    print("\n🧪 Evaluating Recall@10 (slug-based)...\n")

    for i, (query, group) in enumerate(df.groupby("Query"), 1):
        relevant_slugs = {
            extract_slug(u) for u in group["Assessment_url"].dropna().tolist()
        }

        recommendations = recommend(query, k=TOP_K)
        predicted_slugs = [
            extract_slug(r["url"]) for r in recommendations
        ]

        r_at_10 = recall_at_k(predicted_slugs, relevant_slugs, k=TOP_K)
        recalls.append(r_at_10)

        print(f"Query {i}: Recall@10 = {r_at_10:.2f}")

    avg_recall = sum(recalls) / len(recalls)
    print("\n📊 Average Recall@10:", round(avg_recall, 3))


if __name__ == "__main__":
    main()
