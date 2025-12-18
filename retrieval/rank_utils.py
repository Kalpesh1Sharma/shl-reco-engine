from collections import defaultdict


def categorize_assessment(test_type: str):
    """
    Map raw test type text into high-level categories
    """
    if not test_type:
        return "other"

    t = test_type.lower()

    if "cognitive" in t or "ability" in t or "aptitude" in t:
        return "cognitive"
    if "personality" in t or "behavior" in t:
        return "personality"
    if "skill" in t or "knowledge" in t or "technical" in t:
        return "skills"

    return "other"


def balanced_rerank(candidates, top_k=10):
    """
    Ensure diversity across assessment categories
    """
    buckets = defaultdict(list)

    for item in candidates:
        category = categorize_assessment(item.get("test_type", ""))
        buckets[category].append(item)

    final = []

    # Priority order preferred by SHL
    priority = ["skills", "cognitive", "personality", "other"]

    while len(final) < top_k:
        added = False
        for p in priority:
            if buckets[p]:
                final.append(buckets[p].pop(0))
                added = True
                if len(final) == top_k:
                    break
        if not added:
            break

    return final
