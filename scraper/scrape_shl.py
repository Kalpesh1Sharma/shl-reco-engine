import requests
import json
import uuid
import time
from tqdm import tqdm
from bs4 import BeautifulSoup

OUTPUT_PATH = "data/shl_catalog_raw.json"

SITEMAP_INDEX = "https://www.shl.com/sitemap_index.xml"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/xml,text/html"
}


# -------------------------------------------------
# STEP 1: Fetch sitemap index
# -------------------------------------------------
def get_sitemap_urls():
    print("📡 Fetching sitemap index...")
    r = requests.get(SITEMAP_INDEX, headers=HEADERS, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "xml")

    sitemap_urls = [loc.text for loc in soup.find_all("loc")]
    return sitemap_urls


# -------------------------------------------------
# STEP 2: Extract product URLs from product sitemaps
# -------------------------------------------------
def get_product_urls():
    sitemap_urls = get_sitemap_urls()
    product_urls = set()

    for sitemap in sitemap_urls:
        # Only product-related sitemaps
        if "product" not in sitemap.lower():
            continue

        print(f"🔍 Parsing sitemap: {sitemap}")
        r = requests.get(sitemap, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            continue

        soup = BeautifulSoup(r.text, "xml")

        for loc in soup.find_all("loc"):
            url = loc.text.lower()

            if "/solutions/products/" in url and "job" not in url:
                product_urls.add(loc.text)

    return list(product_urls)


# -------------------------------------------------
# STEP 3: Scrape individual assessment page
# -------------------------------------------------
def scrape_assessment_page(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return None

    try:
        name = soup.find("h1").get_text(strip=True)
    except Exception:
        return None

    description = ""
    desc = soup.find("div", class_="product-description")
    if desc:
        description = desc.get_text(" ", strip=True)

    def extract(label):
        tag = soup.find(text=lambda x: x and label.lower() in x.lower())
        if tag:
            nxt = tag.find_next()
            if nxt:
                return nxt.get_text(strip=True)
        return ""

    return {
        "assessment_id": str(uuid.uuid4()),
        "name": name,
        "url": url,
        "description": description,
        "test_type": extract("Test Type"),
        "duration": extract("Duration"),
        "remote_support": extract("Remote"),
        "adaptive_support": extract("Adaptive")
    }


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def main():
    product_links = get_product_urls()
    print(f"🔗 Product URLs discovered: {len(product_links)}")

    assessments = []

    for url in tqdm(product_links, desc="Scraping assessments"):
        data = scrape_assessment_page(url)
        if data:
            assessments.append(data)
        time.sleep(0.3)

    print(f"✅ Successfully scraped {len(assessments)} assessments")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    print(f"📁 Data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
