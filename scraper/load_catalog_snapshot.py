import pandas as pd
import json
import uuid
import os

CSV_INPUT = "Catalogue.csv"
OUTPUT_PATH = "data/shl_catalog_raw.json"
MIN_REQUIRED = 377


def find_column(columns, keywords):
    """Find a column whose name contains any keyword"""
    for col in columns:
        col_lower = col.lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None


def main():
    if not os.path.exists(CSV_INPUT):
        raise FileNotFoundError(
            f"❌ Cannot find {CSV_INPUT}. Place it in project root."
        )

    print(f"📥 Loading catalogue from {CSV_INPUT} ...")
    df = pd.read_csv(CSV_INPUT)

    print("🧾 Columns found:")
    for c in df.columns:
        print("  -", c)

    # Detect required columns
    name_col = find_column(df.columns, ["assessment", "name", "title"])
    url_col = find_column(df.columns, ["url", "link"])

    desc_col = find_column(df.columns, ["description"])
    type_col = find_column(df.columns, ["test type", "type"])
    duration_col = find_column(df.columns, ["duration", "time"])
    remote_col = find_column(df.columns, ["remote"])
    adaptive_col = find_column(df.columns, ["adaptive"])

    if not name_col or not url_col:
        raise ValueError("❌ Required columns (name/url) not found")

    print("\n🔎 Column mapping:")
    print("  Name       →", name_col)
    print("  URL        →", url_col)
    print("  Description→", desc_col)
    print("  Test Type  →", type_col)
    print("  Duration   →", duration_col)
    print("  Remote     →", remote_col)
    print("  Adaptive   →", adaptive_col)

    records = []

    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        url = str(row.get(url_col, "")).strip()

        if not name or not url:
            continue

        # Normalize URL
        if url.startswith("/"):
            url = "https://www.shl.com" + url

        records.append({
            "assessment_id": str(uuid.uuid4()),
            "name": name,
            "url": url,
            "description": str(row.get(desc_col, "")).strip() if desc_col else "",
            "test_type": str(row.get(type_col, "")).strip() if type_col else "",
            "duration": str(row.get(duration_col, "")).strip() if duration_col else "",
            "remote_support": str(row.get(remote_col, "")).strip() if remote_col else "",
            "adaptive_support": str(row.get(adaptive_col, "")).strip() if adaptive_col else ""
        })

    # Deduplicate by URL
    unique = {r["url"]: r for r in records}
    final_records = list(unique.values())

    print(f"\n✅ Total unique assessments collected: {len(final_records)}")

    if len(final_records) < MIN_REQUIRED:
        raise ValueError(
            f"❌ Only {len(final_records)} assessments found. "
            f"Minimum required is {MIN_REQUIRED}."
        )

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)

    print(f"📁 Final catalogue saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
