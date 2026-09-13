import os
import glob
import random
import cv2
import numpy as np
from ultralytics import YOLO
from preprocessor import ArtefactPreprocessor
from augmentor import MuseumAugmentor

CLASSES = {
    "soekarno": 0,
    "dewantara": 1,
    "kartini": 2,
    "soedirman": 3
}

detector = YOLO("yolov8n.pt")

def detect_face_or_bust(img_bgr):
    res = detector.predict(img_bgr, conf=0.18, verbose=False)[0]
    person_boxes = [b for b in res.boxes if int(b.cls[0]) == 0]

    if len(person_boxes) > 0:
        best_b = max(person_boxes, key=lambda b: float(b.xywhn[0][2] * b.xywhn[0][3]))
        px, py, pw, ph = best_b.xywhn[0].cpu().numpy().tolist()
        # Tighten box to isolate the head/face and immediate accessories (peci/sanggul/blangkon)
        # Shift center upward towards head, reduce width & height to eliminate torso/phone bias
        head_y = max(0.08, py - 0.20 * ph)
        head_w = max(0.12, min(0.92, pw * 0.72))
        head_h = max(0.15, min(0.92, ph * 0.58))
        return [px, head_y, head_w, head_h]

    return [0.5, 0.45, 0.65, 0.70]

def clean_directory(dir_path):
    if os.path.exists(dir_path):
        for f in glob.glob(os.path.join(dir_path, "*")):
            try:
                os.remove(f)
            except Exception:
                pass

def build_dataset_from_reviewed(review_dir="data/review", output_dir="dataset", variants_per_image=25, val_ratio=0.20):
    print("Building YOLO Dataset from User-Verified Clean Images...")
    random.seed(42)
    np.random.seed(42)

    preprocessor = ArtefactPreprocessor(beta_passing_grade=50.0, target_size=(640, 640))
    augmentor = MuseumAugmentor(target_size=(640, 640))

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        clean_directory(d)

    sample_counter = 0
    total_train = 0
    total_val = 0

    for category, class_id in CLASSES.items():
        cat_dir = os.path.join(review_dir, category)
        image_paths = sorted(glob.glob(os.path.join(cat_dir, "*.jpg")) + glob.glob(os.path.join(cat_dir, "*.png")) + glob.glob(os.path.join(cat_dir, "*.jpeg")))

        if not image_paths:
            print(f"Warning: No reviewed images found in {cat_dir}")
            continue

        random.shuffle(image_paths)
        total_base = len(image_paths)
        num_val = max(3, int(total_base * val_ratio))
        val_bases = set(image_paths[:num_val])
        train_bases = [p for p in image_paths if p not in val_bases]

        target_train_per_class = 650
        num_train = len(train_bases)
        class_variants_per_image = max(5, int(target_train_per_class / num_train) - 1) if num_train > 0 else 25

        print(f"[{category}] Clean verified images: {total_base} (Train: {len(train_bases)}, Hold-out Val: {len(val_bases)}) | Adaptive variants/img: {class_variants_per_image}")

        # 1. Training images with museum augmentations (dynamically balanced)
        for img_path in train_bases:
            img = cv2.imread(img_path)
            if img is None:
                continue

            box = detect_face_or_bust(img)
            variants = [(img, box)]
            variants.extend(augmentor.generate_variants(img, box, num_variants=class_variants_per_image))

            for var_img, var_box in variants:
                sample_counter += 1
                total_train += 1

                std_img, _ = preprocessor.standardize(var_img)
                img_name = f"{category}_tr_{sample_counter:06d}.jpg"
                lbl_name = f"{category}_tr_{sample_counter:06d}.txt"

                cv2.imwrite(os.path.join(train_img_dir, img_name), std_img)

                if var_box is not None:
                    bx, by, bw, bh = var_box
                    bx = max(0.01, min(0.99, bx))
                    by = max(0.01, min(0.99, by))
                    bw = max(0.02, min(0.98, bw))
                    bh = max(0.02, min(0.98, bh))
                    with open(os.path.join(train_lbl_dir, lbl_name), "w") as f:
                        f.write(f"{class_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

        # 2. Hold-out Validation images (100% clean, letterbox only, zero leakage)
        for img_path in val_bases:
            img = cv2.imread(img_path)
            if img is None:
                continue

            sample_counter += 1
            total_val += 1

            box = detect_face_or_bust(img)
            std_img, _ = preprocessor.standardize(img)
            img_name = f"{category}_val_{sample_counter:06d}.jpg"
            lbl_name = f"{category}_val_{sample_counter:06d}.txt"

            cv2.imwrite(os.path.join(val_img_dir, img_name), std_img)

            if box is not None:
                bx, by, bw, bh = box
                bx = max(0.01, min(0.99, bx))
                by = max(0.01, min(0.99, by))
                bw = max(0.02, min(0.98, bw))
                bh = max(0.02, min(0.98, bh))
                with open(os.path.join(val_lbl_dir, lbl_name), "w") as f:
                    f.write(f"{class_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

    # 3. Modern Human Negative Faces (Teaching model to completely ignore ordinary people & visitors)
    print("Injecting modern human negative face samples (non-hero visitors)...")
    neg_face_paths = sorted(glob.glob("data/negative_faces/*.jpg"))
    random.shuffle(neg_face_paths)
    val_neg_faces = set(neg_face_paths[:10])
    train_neg_faces = [p for p in neg_face_paths if p not in val_neg_faces]

    for p in train_neg_faces:
        im = cv2.imread(p)
        if im is None:
            continue
        sample_counter += 1
        total_train += 1
        std_im, _ = preprocessor.standardize(im)
        cv2.imwrite(os.path.join(train_img_dir, f"neg_face_tr_{sample_counter:06d}.jpg"), std_im)
        with open(os.path.join(train_lbl_dir, f"neg_face_tr_{sample_counter:06d}.txt"), "w") as f:
            pass

    for p in val_neg_faces:
        im = cv2.imread(p)
        if im is None:
            continue
        sample_counter += 1
        total_val += 1
        std_im, _ = preprocessor.standardize(im)
        cv2.imwrite(os.path.join(val_img_dir, f"neg_face_val_{sample_counter:06d}.jpg"), std_im)
        with open(os.path.join(val_lbl_dir, f"neg_face_val_{sample_counter:06d}.txt"), "w") as f:
            pass

    # 4. Museum negative backgrounds
    print("Adding museum negative background frames...")
    for _ in range(30):
        sample_counter += 1
        total_train += 1
        bg = np.full((640, 640, 3), random.randint(30, 180), dtype=np.uint8)
        frame_c = (random.randint(20, 60), random.randint(20, 60), random.randint(20, 60))
        cv2.rectangle(bg, (50, 50), (590, 590), frame_c, thickness=random.randint(10, 30))
        cv2.imwrite(os.path.join(train_img_dir, f"bg_tr_{sample_counter:06d}.jpg"), bg)
        with open(os.path.join(train_lbl_dir, f"bg_tr_{sample_counter:06d}.txt"), "w") as f:
            pass

    for _ in range(10):
        sample_counter += 1
        total_val += 1
        bg = np.full((640, 640, 3), random.randint(40, 190), dtype=np.uint8)
        cv2.imwrite(os.path.join(val_img_dir, f"bg_val_{sample_counter:06d}.jpg"), bg)
        with open(os.path.join(val_lbl_dir, f"bg_val_{sample_counter:06d}.txt"), "w") as f:
            pass

    abs_path = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""path: {abs_path}
train: images/train
val: images/val

names:
  0: soekarno
  1: dewantara
  2: kartini
  3: soedirman
"""
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"\nDataset successfully built from reviewed images!")
    print(f"Total training samples: {total_train}")
    print(f"Total validation samples: {total_val}")

if __name__ == "__main__":
    build_dataset_from_reviewed()
