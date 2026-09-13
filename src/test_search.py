import requests
import re
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

def test_bing(query, count=60):
    url = f"https://www.bing.com/images/async?q={query}&first=1&count={count}&mmasync=1"
    r = requests.get(url, headers=headers, timeout=10)
    matches = re.findall(r'murl&quot;:&quot;(http[^&]+)&quot;', r.text)
    if not matches:
        matches = re.findall(r'"murl":"(http[^"]+)"', r.text)
    return matches

def test_wikimedia_deep(query, max_limit=100):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": max_limit,
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "format": "json"
    }
    r = requests.get(url, params=params, headers={"User-Agent": "NationalHeroesAR/1.0"}, timeout=10)
    urls = []
    if r.status_code == 200:
        pages = r.json().get("query", {}).get("pages", {})
        for _, p in pages.items():
            info = p.get("imageinfo", [])
            if info:
                u = info[0].get("url")
                if u:
                    urls.append(u)
    return urls

for q in ["Soekarno", "Ki Hajar Dewantara", "Kartini", "Jenderal Sudirman", "Museum exhibition empty wall"]:
    bing_urls = test_bing(q)
    wiki_urls = test_wikimedia_deep(q)
    print(f"Query '{q}': Bing={len(bing_urls)}, Wikimedia={len(wiki_urls)}")
