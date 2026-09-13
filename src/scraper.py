import os
import io
import re
import time
import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image
import imagehash

warnings.filterwarnings("ignore")

HERO_QUERIES = {
    "soekarno": [
        "Soekarno",
        "Bung Karno",
        "Presiden Soekarno",
        "Lukisan Bung Karno museum",
        "Soekarno peci arsip"
    ],
    "dewantara": [
        "Ki Hajar Dewantara",
        "Ki Hadjar Dewantara",
        "Soewardi Soerjaningrat",
        "Lukisan Ki Hajar Dewantara",
        "Taman Siswa Ki Hadjar Dewantara"
    ],
    "kartini": [
        "Raden Ajeng Kartini",
        "RA Kartini kebaya",
        "Lukisan RA Kartini museum",
        "Kartini Jepara arsip",
        "Pahlawan RA Kartini"
    ],
    "soedirman": [
        "Jenderal Soedirman",
        "Panglima Besar Soedirman",
        "Jenderal Sudirman gerilya",
        "Lukisan Jenderal Soedirman",
        "Patung Jenderal Soedirman"
    ],
    "background_negatives": [
        "Museum gallery empty wall",
        "Museum exhibition room interior",
        "Classic empty picture frame wall",
        "Museum visitor gallery blurred"
    ]
}

WIKIMEDIA_SEARCH = {
    "soekarno": ["Sukarno", "Bung Karno", "Category:Sukarno"],
    "dewantara": ["Ki Hajar Dewantara", "Ki Hadjar Dewantara", "Soewardi Soerjaningrat"],
    "kartini": ["Kartini", "Raden Ayu Kartini", "Category:Kartini"],
    "soedirman": ["Sudirman (general)", "Jenderal Sudirman", "Category:Sudirman"]
}

def fetch_wikimedia(term, max_limit=80):
    url = "https://commons.wikimedia.org/w/api.php"
    urls = []
    headers = {"User-Agent": "NationalHeroesAR/1.0 (academic research)"}

    if term.startswith("Category:"):
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": term,
            "gcmlimit": max_limit,
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": 800,
            "format": "json"
        }
    else:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": term,
            "gsrnamespace": 6,
            "gsrlimit": max_limit,
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": 800,
            "format": "json"
        }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=12)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for _, p in pages.items():
                info = p.get("imageinfo", [])
                if info:
                    u = info[0].get("thumburl") or info[0].get("url")
                    m = info[0].get("mime", "")
                    if u and ("image" in m or u.lower().endswith((".jpg", ".jpeg", ".png"))):
                        urls.append(u)
    except Exception as e:
        print(f"Wikimedia error for {term}: {e}", flush=True)

    return urls

def fetch_bing(query, max_pages=3):
    urls = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }

    for page_idx in range(max_pages):
        first = 1 + (page_idx * 50)
        url = f"https://www.bing.com/images/async?q={query}&first={first}&count=50&mmasync=1"
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                matches = re.findall(r'murl&quot;:&quot;(http[^&]+)&quot;', r.text)
                if not matches:
                    matches = re.findall(r'"murl":"(http[^"]+)"', r.text)
                urls.extend(matches)
            time.sleep(0.3)
        except Exception:
            break

    return urls

def download_image(url, min_dim=180):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        img = Image.open(io.BytesIO(r.content))
        w, h = img.size
        if w < min_dim or h < min_dim:
            return None
        img_rgb = img.convert("RGB")
        hsh = imagehash.phash(img_rgb)
        return (img_rgb, hsh, url)
    except Exception:
        return None

def curate_and_save(urls, save_dir, max_workers=10):
    os.makedirs(save_dir, exist_ok=True)
    existing_hashes = []
    saved_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image, u): u for u in urls}
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            img_rgb, hsh, u = res
            is_dup = False
            for eh in existing_hashes:
                if hsh - eh < 6:
                    is_dup = True
                    break
            if is_dup:
                continue

            existing_hashes.append(hsh)
            saved_count += 1
            filename = f"hero_{saved_count:04d}_{hashlib.md5(u.encode()).hexdigest()[:6]}.jpg"
            img_rgb.save(os.path.join(save_dir, filename), "JPEG", quality=92)

    return saved_count

def run_scraper(output_base="data/raw"):
    print("Initiating Multi-Provider Harvesting Engine (Wikimedia + Bing)...", flush=True)
    total_saved = 0

    for category, queries in HERO_QUERIES.items():
        category_dir = os.path.join(output_base, category)
        urls = set()

        if category in WIKIMEDIA_SEARCH:
            for term in WIKIMEDIA_SEARCH[category]:
                wiki_urls = fetch_wikimedia(term, max_limit=80)
                urls.update(wiki_urls)

        for q in queries:
            bing_urls = fetch_bing(q, max_pages=2)
            urls.update(bing_urls)

        print(f"[{category}] Collected {len(urls)} candidates. Downloading...", flush=True)
        saved = curate_and_save(list(urls), category_dir, max_workers=10)
        print(f"[{category}] Successfully curated {saved} unique high-quality images.", flush=True)
        total_saved += saved

    print(f"Scraping completed. Total curated raw samples: {total_saved}", flush=True)
    return total_saved

if __name__ == "__main__":
    run_scraper()
