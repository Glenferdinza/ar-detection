import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

HERO_MAP = {
    "soekarno": 0,
    "dewantara": 1,
    "kartini": 2,
    "soedirman": 3
}

def get_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(cascade_path):
        return cv2.CascadeClassifier(cascade_path)
    return None

def detect_hero_roi(img_bgr, yolo_model, face_cascade):
    h, w = img_bgr.shape[:2]
    total_area = h * w

    # 1. First pass: YOLO proposal network
    res = yolo_model.predict(img_bgr, conf=0.10, imgsz=416, verbose=False)[0]
    valid_boxes = []

    for b in res.boxes:
        conf = float(b.conf[0])
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
        bw, bh = x2 - x1, y2 - y1
        if bw < 25 or bh < 25:
            continue
        # Discard full-body vertical human bodies (> 2.15)
        if (bh / float(max(1, bw))) > 2.20:
            continue
        # Discard whole-frame canvas boxes
        if (bw * bh) > 0.95 * total_area:
            continue
        valid_boxes.append((conf, x1, y1, x2, y2))

    if valid_boxes:
        valid_boxes.sort(key=lambda x: x[0], reverse=True)
        _, bx1, by1, bx2, by2 = valid_boxes[0]
        return max(0, bx1), max(0, by1), min(w, bx2), min(h, by2)

    # 2. Second pass fallback: Haar Cascade Face Detector
    if face_cascade is not None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            # Select largest face
            fx, fy, fw, fh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            # Expand face box slightly to include accessories (peci/sanggul/blangkon)
            pad_y = int(fh * 0.25)
            pad_x = int(fw * 0.15)
            x1 = max(0, fx - pad_x)
            y1 = max(0, fy - pad_y)
            x2 = min(w, fx + fw + pad_x)
            y2 = min(h, fy + fh + int(pad_y * 0.5))
            return x1, y1, x2, y2

    # 3. Third pass fallback: Central portrait heuristic
    cx, cy = w // 2, h // 2
    rw, rh = int(w * 0.65), int(h * 0.65)
    return max(0, cx - rw // 2), max(0, cy - rh // 2), min(w, cx + rw // 2), min(h, cy + rh // 2)

def process_all_incoming(incoming_root="data/incoming", curated_root="data/curated"):
    print("Processing all incoming photos from museum...")
    yolo_model = YOLO("weights/best.pt")
    face_cascade = get_face_cascade()

    stats = {}

    for hero, cls_id in HERO_MAP.items():
        src_dir = os.path.join(incoming_root, hero)
        dst_dir = os.path.join(curated_root, hero)
        os.makedirs(dst_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(src_dir, "*.*")))
        valid_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        processed_count = 0

        for idx, f in enumerate(valid_files):
            img = cv2.imread(f)
            if img is None:
                continue

            h, w = img.shape[:2]
            x1, y1, x2, y2 = detect_hero_roi(img, yolo_model, face_cascade)

            # Ensure valid bounds
            bw, bh = x2 - x1, y2 - y1
            if bw < 25 or bh < 25:
                continue

            # Target filenames
            base_name = f"incoming_{hero}_{idx:03d}"
            target_img = os.path.join(dst_dir, f"{base_name}.jpg")
            target_box = os.path.join(dst_dir, f"{base_name}.box")

            # Save normalized image
            cv2.imwrite(target_img, img)

            # Write .box file [x1, y1, x2, y2]
            with open(target_box, "w") as bf:
                bf.write(f"{x1},{y1},{x2},{y2}\n")

            processed_count += 1

        stats[hero] = processed_count
        print(f"  [{hero.upper()}] Processed & integrated {processed_count} / {len(valid_files)} incoming images.")

    print(f"Auto-annotation and integration complete: {stats}")
    return stats

if __name__ == "__main__":
    process_all_incoming()
