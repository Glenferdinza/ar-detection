import os
import io
import re
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image
import cv2
import numpy as np
import imagehash
from ultralytics import YOLO

os.makedirs("data/multimodal", exist_ok=True)

detector = YOLO("yolov8n.pt")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

TARGETED_QUERIES = {
    "soekarno": [
        "Soekarno portrait",
        "Bung Karno foto resmi",
        "Ir Soekarno lukisan museum",
        "Presiden Sukarno berpeci",
        "Lukisan Basoeki Abdullah Bung Karno",
        "Foto Sukarno close up",
        "Soekarno pidato portrait",
        "Patung dada Bung Karno",
        "Lukisan wajah Soekarno",
        "Sukarno Indonesia president photo",
        "Soekarno peci hitam",
        "Bung Karno vintage photo"
    ],
    "dewantara": [
        "Ki Hajar Dewantara portrait",
        "Ki Hadjar Dewantara foto resmi",
        "Lukisan Ki Hajar Dewantara museum",
        "Ki Hajar Dewantara kacamata blangkon",
        "Ki Hadjar Dewantara Tamansiswa",
        "Patung dada Ki Hajar Dewantara",
        "Bust Ki Hadjar Dewantara",
        "Lukisan wajah Ki Hajar Dewantara",
        "Foto Ki Hajar Dewantara tempo doeloe",
        "Tokoh pendidikan Ki Hadjar Dewantara portrait",
        "Ki Hajar Dewantara close up",
        "Ki Hadjar Dewantara official portrait"
    ],
    "kartini": [
        "RA Kartini portrait",
        "Raden Ajeng Kartini foto resmi",
        "Lukisan RA Kartini museum",
        "Kartini sanggul kebaya foto",
        "Lukisan wajah Raden Adjeng Kartini",
        "Patung dada RA Kartini",
        "Bust of Raden Ajeng Kartini",
        "Foto Kartini Tropenmuseum",
        "Kartini Jepara photo vintage",
        "Raden Ayu Kartini portrait",
        "Lukisan Kartini Basoeki Abdullah",
        "Kartini official portrait photo"
    ],
    "soedirman": [
        "Jenderal Sudirman portrait",
        "Jenderal Soedirman foto resmi",
        "Lukisan Jenderal Soedirman museum",
        "Panglima Besar Sudirman mantel ikat kepala",
        "Patung dada Jenderal Sudirman",
        "Bust of General Sudirman",
        "Foto Jenderal Sudirman gerilya",
        "Lukisan wajah Panglima Sudirman",
        "Soedirman blangkon mantel foto",
        "General Sudirman portrait photo",
        "Jenderal Soedirman tempo doeloe",
        "Panglima Besar Soedirman lukisan"
    ]
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
        r = requests.get(url, params=params, headers={"User-Agent": "NationalHeroesAR/3.0"}, timeout=12).json()
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

def download_single(url):
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200 or len(r.content) < 6000:
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        w, h = img.size
        aspect = w / float(h)
        if aspect > 1.35 or aspect < 0.50 or w < 150 or h < 150:
            return None
        return img
    except Exception:
        pass
    return None

def validate_person_strictly(pil_img):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    res = detector.predict(cv_img, conf=0.25, verbose=False)[0]
    person_boxes = [b for b in res.boxes if int(b.cls[0]) == 0]

    if len(person_boxes) == 0 or len(person_boxes) > 2:
        return None, None

    best_b = max(person_boxes, key=lambda b: float(b.xywhn[0][2] * b.xywhn[0][3]))
    bw = float(best_b.xywhn[0][2])
    bh = float(best_b.xywhn[0][3])
    area = bw * bh

    if area < 0.12 or area > 0.96:
        return None, None

    norm_box = best_b.xywhn[0].cpu().numpy().tolist()
    return pil_img, norm_box

def clean_and_harvest_pure(target_per_hero=35):
    print("Starting Pure Multi-Modal Dataset Harvesting (Strict Single-Person Verification)...")

    for hero, queries in TARGETED_QUERIES.items():
        save_dir = os.path.join("data", "multimodal", hero)
        os.makedirs(save_dir, exist_ok=True)

        existing_files = glob.glob(os.path.join(save_dir, "*.jpg"))
        valid_existing = []
        seen_hashes = set()

        for f in existing_files:
            box_file = f.replace(".jpg", ".box")
            if os.path.exists(box_file):
                with open(box_file) as bf:
                    content = bf.read()
                    if "0.5 0.5 0.75 0.85" in content:
                        try:
                            os.remove(f)
                            os.remove(box_file)
                        except Exception:
                            pass
                        continue
            try:
                im = Image.open(f)
                seen_hashes.add(str(imagehash.phash(im)))
                valid_existing.append(f)
            except Exception:
                pass

        print(f"\n[{hero}] Retained pure verified images: {len(valid_existing)}. Target: {target_per_hero}...")
        if len(valid_existing) >= target_per_hero:
            continue

        candidate_urls = set()
        for q in queries:
            wiki_urls = fetch_wikimedia_commons(q, limit=35)
            bing_urls = fetch_bing_image_urls(q, max_count=35)
            candidate_urls.update(wiki_urls)
            candidate_urls.update(bing_urls)
            time.sleep(0.1)

        print(f"[{hero}] Collected {len(candidate_urls)} candidate URLs. Downloading images in parallel...")

        downloaded = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(download_single, u): u for u in candidate_urls}
            for fut in as_completed(future_to_url):
                res = fut.result()
                if res is not None:
                    downloaded.append(res)

        print(f"[{hero}] Downloaded {len(downloaded)} images. Validating with YOLO person detector...")

        saved = len(valid_existing)
        for img in downloaded:
            if saved >= target_per_hero:
                break
            h_val = str(imagehash.phash(img))
            if h_val in seen_hashes:
                continue
            seen_hashes.add(h_val)

            valid_img, box = validate_person_strictly(img)
            if valid_img is not None:
                saved += 1
                img_path = os.path.join(save_dir, f"{hero}_pure_{saved:03d}.jpg")
                box_path = os.path.join(save_dir, f"{hero}_pure_{saved:03d}.box")
                valid_img.save(img_path, "JPEG", quality=92)
                with open(box_path, "w") as f:
                    f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")

        print(f"[{hero}] Pure collection now has {saved} verified authentic samples.")

    print("\nPure multi-modal dataset harvesting complete!")

if __name__ == "__main__":
    clean_and_harvest_pure(target_per_hero=35)
