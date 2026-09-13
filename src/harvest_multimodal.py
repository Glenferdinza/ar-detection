import os
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

os.makedirs("data/multimodal", exist_ok=True)

MULTIMODAL_QUERIES = {
    "soekarno": [
        "Lukisan Ir Soekarno museum",
        "Patung Ir Soekarno",
        "Patung Bung Karno Monas",
        "Monumen Soekarno",
        "Bust of Sukarno",
        "Presiden Sukarno lukisan resmi",
        "Lukisan Basoeki Abdullah Soekarno",
        "Patung Soekarno Blitar",
        "Lukisan dinding museum Soekarno"
    ],
    "dewantara": [
        "Lukisan Ki Hajar Dewantara museum",
        "Patung Ki Hajar Dewantara",
        "Patung Ki Hadjar Dewantara Kemdikbud",
        "Monumen Ki Hajar Dewantara",
        "Bust of Ki Hajar Dewantara",
        "Lukisan resmi Ki Hadjar Dewantara",
        "Patung Ki Hajar Dewantara Tamansiswa",
        "Dewantara Kirti Griya lukisan",
        "Patung dada Ki Hajar Dewantara"
    ],
    "kartini": [
        "Lukisan RA Kartini museum",
        "Patung RA Kartini",
        "Patung Kartini Monas",
        "Monumen RA Kartini",
        "Bust of Kartini",
        "Lukisan resmi Raden Ajeng Kartini",
        "Patung Kartini Jepara",
        "Museum Kartini Rembang patung",
        "Lukisan Kartini dinding museum"
    ],
    "soedirman": [
        "Lukisan Jenderal Soedirman museum",
        "Patung Jenderal Sudirman Jakarta",
        "Patung Sudirman Bintaran",
        "Monumen Jenderal Sudirman",
        "Bust of General Sudirman",
        "Lukisan resmi Panglima Besar Soedirman",
        "Patung Jenderal Soedirman kuda",
        "Museum Sasmitaloka Sudirman patung",
        "Lukisan perang gerilya Soedirman"
    ]
}

detector = YOLO("yolov8n.pt")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

def fetch_bing_image_urls(query, max_count=35):
    url = f"https://www.bing.com/images/async?q={query}&first=1&count={max_count}&mmasync=1"
    urls = []
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            matches = re.findall(r'murl&quot;:&quot;(http[^&]+)&quot;', r.text)
            if not matches:
                matches = re.findall(r'"murl":"(http[^"]+)"', r.text)
            urls.extend(matches)
    except Exception:
        pass
    return urls

def fetch_wikimedia_commons(query, limit=35):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 800,
        "format": "json"
    }
    urls = []
    try:
        r = requests.get(url, params=params, headers={"User-Agent": "NationalHeroesAR/2.0"}, timeout=12).json()
        pages = r.get("query", {}).get("pages", {})
        for pid, p in pages.items():
            info = p.get("imageinfo", [])
            if info:
                u = info[0].get("thumburl") or info[0].get("url")
                if u:
                    urls.append(u)
    except Exception:
        pass
    return urls

def validate_and_crop(pil_img):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    aspect = w / float(h)
    if aspect > 1.35 or aspect < 0.45:
        return None, None

    res = detector.predict(cv_img, conf=0.18, verbose=False)[0]
    person_boxes = [b for b in res.boxes if int(b.cls[0]) == 0]

    # Reject if crowded (boybands/crowds)
    if len(person_boxes) > 2:
        return None, None

    if len(person_boxes) >= 1:
        best_b = max(person_boxes, key=lambda b: float(b.xywhn[0][2] * b.xywhn[0][3]))
        norm_box = best_b.xywhn[0].cpu().numpy().tolist()
        return pil_img, norm_box

    # For close-up statue busts that YOLO person class misses:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    center_crop = gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    if center_crop.std() > 30 and 40 < np.mean(center_crop) < 220:
        return pil_img, [0.5, 0.5, 0.75, 0.85]

    return None, None

import imagehash

def download_single(url):
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200 or len(r.content) < 5000:
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        w, h = img.size
        aspect = w / float(h)
        if aspect > 1.4 or aspect < 0.45 or w < 120 or h < 120:
            return None
        return img
    except Exception:
        pass
    return None

def harvest_multimodal_dataset(target_per_hero=35):
    print("Harvesting Multi-Modal Data: Photos, Museum Paintings, Statues, Busts...", flush=True)

    for hero, queries in MULTIMODAL_QUERIES.items():
        save_dir = os.path.join("data", "multimodal", hero)
        os.makedirs(save_dir, exist_ok=True)

        existing_count = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
        print(f"\n[{hero}] Multi-modal harvest. Current: {existing_count}. Target: {target_per_hero}...", flush=True)
        if existing_count >= target_per_hero:
            print(f"[{hero}] Already satisfied ({existing_count} >= {target_per_hero}).", flush=True)
            continue

        candidate_urls = set()
        for q in queries:
            wiki_urls = fetch_wikimedia_commons(q, limit=30)
            bing_urls = fetch_bing_image_urls(q, max_count=30)
            candidate_urls.update(wiki_urls)
            candidate_urls.update(bing_urls)
            time.sleep(0.1)

        print(f"[{hero}] Collected {len(candidate_urls)} candidate URLs. Downloading images in parallel...", flush=True)

        downloaded_images = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(download_single, u): u for u in candidate_urls}
            for fut in as_completed(future_to_url):
                img = fut.result()
                if img is not None:
                    downloaded_images.append(img)

        print(f"[{hero}] Downloaded {len(downloaded_images)} raw images. Validating with YOLO detector...", flush=True)

        saved = existing_count
        seen_hashes = set()
        # Seed existing hashes to avoid duplicates
        for existing_img_name in os.listdir(save_dir):
            if existing_img_name.endswith(".jpg"):
                try:
                    ex_img = Image.open(os.path.join(save_dir, existing_img_name))
                    seen_hashes.add(str(imagehash.phash(ex_img)))
                except Exception:
                    pass

        for img in downloaded_images:
            if saved >= target_per_hero:
                break
            h_val = str(imagehash.phash(img))
            if h_val in seen_hashes:
                continue
            seen_hashes.add(h_val)

            valid_img, box = validate_and_crop(img)
            if valid_img is not None:
                saved += 1
                img_path = os.path.join(save_dir, f"{hero}_multi_{saved:03d}.jpg")
                box_path = os.path.join(save_dir, f"{hero}_multi_{saved:03d}.box")
                valid_img.save(img_path, "JPEG", quality=92)
                with open(box_path, "w") as f:
                    f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")

        print(f"[{hero}] Multi-modal collection reached {saved} authentic samples.", flush=True)

    print("\nMulti-modal harvesting complete!", flush=True)

if __name__ == "__main__":
    harvest_multimodal_dataset(target_per_hero=35)

