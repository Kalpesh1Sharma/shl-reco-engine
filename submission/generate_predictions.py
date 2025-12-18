import sys
import os

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
from retrieval.retrieve_and_rank import recommend


INPUT_DATA_PATH = "Gen_AI Dataset.xlsx"
OUTPUT_CSV_PATH = "submission/shl_test_predictions.csv"
TOP_K = 10


def is_test_row(val):
    """
    Decide whether Assessment_url indicates a TEST row
    """
    if pd.isna(val):
        return True

    val = str(val).strip().lower()

    if val in ["", "na", "n/a", "none", "?", "test"]:
        return True

    return False


def main():
    print("📥 Loading dataset...")

    # ---- Load ALL sheets ----
    excel = pd.ExcelFile(INPUT_DATA_PATH)
    print(f"📄 Sheets found: {excel.sheet_names}")

    all_test_queries = []

    for sheet in excel.sheet_names:
        df = excel.parse(sheet)

        if "Query" not in df.columns:
            continue

        if "Assessment_url" not in df.columns:
            # Entire sheet is TEST
            test_df = df[["Query"]].dropna()
            all_test_queries.extend(test_df["Query"].tolist())
            continue

        # Mixed train + test
        test_df = df[df["Assessment_url"].apply(is_test_row)]
        all_test_queries.extend(test_df["Query"].dropna().tolist())

    # Deduplicate queries
    test_queries = list(dict.fromkeys(all_test_queries))

    print(f"🧪 Test queries found: {len(test_queries)}")

    if len(test_queries) == 0:
        raise ValueError(
            "❌ No test queries detected. "
            "Check dataset format manually."
        )

    predictions = []

    for q in test_queries:
        recs = recommend(q, k=TOP_K)
        urls = [r["url"] for r in recs]

        predictions.append({
            "Query": q,
            "Recommended_Assessments": ";".join(urls)
        })

        print(f"✅ Processed query: {q[:60]}...")

    os.makedirs("submission", exist_ok=True)
    output_df = pd.DataFrame(predictions)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\n📁 Predictions saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
