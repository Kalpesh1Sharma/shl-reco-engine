import re


def extract_slug(url: str) -> str:
    """
    Extract and normalize assessment slug from SHL URL
    """
    if not url:
        return ""

    url = url.lower()

    # Remove protocol + domain
    url = re.sub(r"https?://[^/]+", "", url)

    # Remove locale
    url = re.sub(r"^/en/", "/", url)

    # Extract slug
    parts = url.strip("/").split("/")
    if not parts:
        return ""

    slug = parts[-1]

    # Normalize slug
    slug = slug.replace("-new", "")
    slug = re.sub(r"\d+", "", slug)  # remove numbers
    slug = slug.replace("-", "")

    return slug


def recall_at_k(predicted_slugs, relevant_slugs, k=10):
    if not relevant_slugs:
        return 0.0

    top_k = predicted_slugs[:k]
    hits = sum(1 for s in top_k if s in relevant_slugs)

    return hits / len(relevant_slugs)
