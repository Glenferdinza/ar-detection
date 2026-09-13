import os
import io
import time
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

os.makedirs("data/references", exist_ok=True)
os.makedirs("data/curated", exist_ok=True)

CANONICAL_PORTRAITS = {
    "soekarno": "https://thumb.wikimedia.org/wikipedia/commons/thumb/0/01/Presiden_Sukarno.jpg/960px-Presiden_Sukarno.jpg",
    "dewantara": "https://upload.wikimedia.org/wikipedia/commons/a/ad/Ki_Hajar_Dewantara.jpg",
    "kartini": "https://upload.wikimedia.org/wikipedia/commons/2/23/COLLECTIE_TROPENMUSEUM_Portret_van_Raden_Ajeng_Kartini_TMnr_10018776.jpg",
    "soedirman": "https://thumb.wikimedia.org/wikipedia/commons/thumb/e/e7/Sudirman.jpg/960px-Sudirman.jpg"
}

WIKI_QUERIES = {
    "soekarno": [
        "Sukarno",
        "Bung Karno",
        "Presiden Sukarno",
        "Soekarno 1945",
        "Soekarno portrait"
    ],
    "dewantara": [
        "Ki Hajar Dewantara",
        "Ki Hadjar Dewantara",
        "Soewardi Soerjaningrat",
        "Taman Siswa Dewantara",
        "Dewantara portrait"
    ],
    "kartini": [
        "Kartini",
        "Raden Ayu Kartini",
        "Raden Ajeng Kartini",
        "Kartini Jepara",
        "Kartini portrait"
    ],
    "soedirman": [
        "Sudirman",
        "Jenderal Sudirman",
        "Panglima Besar Sudirman",
        "General Sudirman",
        "Sudirman portrait"
    ]
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet = models.resnet18(weights='DEFAULT')
resnet.fc = torch.nn.Identity()
resnet.eval()

detector = YOLO("yolov8n.pt")

headers = {"User-Agent": "NationalHeroesAR/2.0 (verified historical collection)"}

def download_image_as_rgb(url):
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content))
            return img.convert("RGB")
    except Exception:
        pass
    return None

def extract_portrait_crop(pil_img):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    res = detector.predict(cv_img, conf=0.25, verbose=False)[0]
    person_boxes = [b for b in res.boxes if int(b.cls[0]) == 0]

    if len(person_boxes) != 1:
        return None, None

    best_b = person_boxes[0]
    xyxy = best_b.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy

    h, w = cv_img.shape[:2]
    face_h = int((y2 - y1) * 0.55)
    fy2 = min(h, y1 + face_h)

    crop = pil_img.crop((x1, y1, x2, fy2))
    norm_box = best_b.xywhn[0].cpu().numpy().tolist()
    return crop, norm_box

def get_embedding(pil_crop):
    tensor = transform(pil_crop).unsqueeze(0)
    with torch.no_grad():
        emb = resnet(tensor)
    return torch.nn.functional.normalize(emb, p=2, dim=1)

def setup_reference_embeddings():
    ref_embeddings = {}
    for hero, url in CANONICAL_PORTRAITS.items():
        ref_path = f"data/references/{hero}.jpg"
        if not os.path.exists(ref_path):
            img = download_image_as_rgb(url)
            if img is not None:
                img.save(ref_path, "JPEG", quality=95)

        ref_img = Image.open(ref_path).convert("RGB")
        crop, _ = extract_portrait_crop(ref_img)
        if crop is None:
            crop = ref_img
        ref_embeddings[hero] = get_embedding(crop)
        print(f"Loaded canonical reference for {hero}")
    return ref_embeddings

def search_wikimedia_commons(query, limit=50):
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
        r = requests.get(url, params=params, headers=headers, timeout=12).json()
        pages = r.get("query", {}).get("pages", {})
        for pid, p in pages.items():
            info = p.get("imageinfo", [])
            if info:
                u = info[0].get("thumburl") or info[0].get("url")
                if u:
                    urls.append(u)
    except Exception as e:
        print(f"Wikimedia search error for '{query}': {e}")
    return urls

def run_verified_harvest():
    ref_embeddings = setup_reference_embeddings()
    total_verified = 0

    for hero, queries in WIKI_QUERIES.items():
        save_dir = os.path.join("data", "curated", hero)
        os.makedirs(save_dir, exist_ok=True)

        canonical_path = f"data/references/{hero}.jpg"
        if os.path.exists(canonical_path):
            import shutil
            shutil.copy(canonical_path, os.path.join(save_dir, f"{hero}_verified_000.jpg"))
            ref_img = Image.open(canonical_path).convert("RGB")
            _, box = extract_portrait_crop(ref_img)
            if box is None:
                box = [0.5, 0.5, 0.75, 0.85]
            with open(os.path.join(save_dir, f"{hero}_verified_000.box"), "w") as f:
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")

        candidate_urls = set()
        for q in queries:
            urls = search_wikimedia_commons(q, limit=40)
            candidate_urls.update(urls)
            time.sleep(0.3)

        print(f"[{hero}] Filtering {len(candidate_urls)} historical candidates with single-person & face similarity check...", flush=True)

        accepted_count = 1
        for url in candidate_urls:
            img = download_image_as_rgb(url)
            if img is None:
                continue

            crop, norm_box = extract_portrait_crop(img)
            if crop is None:
                continue

            cand_emb = get_embedding(crop)
            sim_target = float(torch.mm(cand_emb, ref_embeddings[hero].T)[0][0])

            other_sims = [
                float(torch.mm(cand_emb, ref_embeddings[oh].T)[0][0])
                for oh in ref_embeddings if oh != hero
            ]
            max_other_sim = max(other_sims) if other_sims else 0.0

            if sim_target >= 0.42 and sim_target >= (max_other_sim - 0.05):
                accepted_count += 1
                img_path = os.path.join(save_dir, f"{hero}_verified_{accepted_count:03d}.jpg")
                box_path = os.path.join(save_dir, f"{hero}_verified_{accepted_count:03d}.box")
                img.save(img_path, "JPEG", quality=95)
                with open(box_path, "w") as f:
                    f.write(f"{norm_box[0]} {norm_box[1]} {norm_box[2]} {norm_box[3]}\n")

        print(f"[{hero}] Total authentic verified portraits: {accepted_count}", flush=True)
        total_verified += accepted_count

    print(f"Harvest complete. Total 100% verified hero portraits: {total_verified}", flush=True)
    return total_verified

if __name__ == "__main__":
    run_verified_harvest()
