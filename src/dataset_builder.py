import os
import glob
import random
import cv2
import numpy as np
from preprocessor import ArtefactPreprocessor
from augmentor import MuseumAugmentor

CLASSES = {
    "soekarno": 0,
    "dewantara": 1,
    "kartini": 2,
    "soedirman": 3
}

def load_verified_box(box_path):
    if os.path.exists(box_path):
        with open(box_path, "r") as f:
            line = f.readline().strip()
            if line:
                parts = [float(p) for p in line.split()]
                if len(parts) == 4:
                    return tuple(parts)
    return (0.5, 0.5, 0.75, 0.85)

def clean_output_dir(output_dir):
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        full_sub = os.path.join(output_dir, sub)
        if os.path.exists(full_sub):
            for f in glob.glob(os.path.join(full_sub, "*")):
                try:
                    os.remove(f)
                except Exception:
                    pass

def build_curated_yolo_dataset(output_dir="dataset", variants_per_image=8, val_ratio=0.20):
    print("Building Leak-Free Multi-Modal Dataset with Base-Image Group Splitting and Handheld Augmentation...")
    random.seed(42)
    np.random.seed(42)

    preprocessor = ArtefactPreprocessor(beta_passing_grade=50.0, target_size=(640, 640))
    augmentor = MuseumAugmentor(target_size=(640, 640))

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    clean_output_dir(output_dir)

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    sample_counter = 0
    total_train_samples = 0
    total_val_samples = 0

    for category, class_id in CLASSES.items():
        base_paths = []
        for src_dir in ["data/curated", "data/multimodal"]:
            cat_dir = os.path.join(src_dir, category)
            if os.path.exists(cat_dir):
                base_paths.extend(glob.glob(os.path.join(cat_dir, "*.jpg")))

        base_paths = sorted(list(set(base_paths)))
        random.shuffle(base_paths)

        total_base = len(base_paths)
        num_val = max(3, int(total_base * val_ratio))
        val_bases = set(base_paths[:num_val])
        train_bases = [p for p in base_paths if p not in val_bases]

        print(f"[{category}] Total base images: {total_base}. Train bases: {len(train_bases)}. Hold-out Val bases: {len(val_bases)}")

        for img_path in train_bases:
            box_path = img_path.replace(".jpg", ".box")
            base_box = load_verified_box(box_path)
            base_img = cv2.imread(img_path)
            if base_img is None:
                continue

            variants = [(base_img, base_box)]
            if variants_per_image > 0:
                variants.extend(augmentor.generate_variants(base_img, base_box, num_variants=variants_per_image))

            for var_img, var_box in variants:
                sample_counter += 1
                total_train_samples += 1

                std_var, _ = preprocessor.standardize(var_img)
                img_filename = f"{category}_tr_{sample_counter:06d}.jpg"
                lbl_filename = f"{category}_tr_{sample_counter:06d}.txt"

                cv2.imwrite(os.path.join(train_img_dir, img_filename), std_var)

                if var_box is not None:
                    bx, by, bw, bh = var_box
                    bx = max(0.01, min(0.99, bx))
                    by = max(0.01, min(0.99, by))
                    bw = max(0.02, min(0.98, bw))
                    bh = max(0.02, min(0.98, bh))
                    lbl_content = f"{class_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n"
                    with open(os.path.join(train_lbl_dir, lbl_filename), "w") as f:
                        f.write(lbl_content)

        for img_path in val_bases:
            box_path = img_path.replace(".jpg", ".box")
            base_box = load_verified_box(box_path)
            base_img = cv2.imread(img_path)
            if base_img is None:
                continue

            # Pure base image
            sample_counter += 1
            total_val_samples += 1

            std_img, _ = preprocessor.standardize(base_img)
            img_filename = f"{category}_val_{sample_counter:06d}.jpg"
            lbl_filename = f"{category}_val_{sample_counter:06d}.txt"

            cv2.imwrite(os.path.join(val_img_dir, img_filename), std_img)

            if base_box is not None:
                bx, by, bw, bh = base_box
                bx = max(0.01, min(0.99, bx))
                by = max(0.01, min(0.99, by))
                bw = max(0.02, min(0.98, bw))
                bh = max(0.02, min(0.98, bh))
                lbl_content = f"{class_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n"
                with open(os.path.join(val_lbl_dir, lbl_filename), "w") as f:
                    f.write(lbl_content)

            # Also 1 handheld phone validation sample per hold-out base
            p_img, p_box = augmentor.generate_handheld_variant(base_img, box=base_box)
            sample_counter += 1
            total_val_samples += 1

            p_std, _ = preprocessor.standardize(p_img)
            img_filename = f"{category}_pval_{sample_counter:06d}.jpg"
            lbl_filename = f"{category}_pval_{sample_counter:06d}.txt"
            cv2.imwrite(os.path.join(val_img_dir, img_filename), p_std)
            if p_box is not None:
                bx, by, bw, bh = p_box
                lbl_content = f"{class_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n"
                with open(os.path.join(val_lbl_dir, lbl_filename), "w") as f:
                    f.write(lbl_content)

    print("Injecting negative background samples (rooms, blank devices, silhouettes)...")
    for bg_idx in range(35):
        sample_counter += 1
        total_train_samples += 1
        cw, ch = 640, 640
        # Background room gradient
        bg_img = np.full((ch, cw, 3), random.randint(120, 210), dtype=np.uint8)

        if random.random() > 0.5:
            # Human silhouette in room (head + shoulders, negative sample)
            head_cx = random.randint(int(cw * 0.25), int(cw * 0.55))
            head_cy = random.randint(int(ch * 0.35), int(ch * 0.50))
            head_r = random.randint(50, 85)
            skin_val = random.randint(140, 200)
            cv2.circle(bg_img, (head_cx, head_cy), head_r, (skin_val, skin_val, skin_val), -1)
            # Hair on top
            cv2.ellipse(bg_img, (head_cx, head_cy - 20), (head_r + 5, int(head_r * 0.6)), 0, 180, 360, (25, 25, 25), -1)
            # Shoulders
            cv2.ellipse(bg_img, (head_cx, head_cy + head_r + 80), (head_r * 2, 90), 0, 0, 360, (60, 60, 80), -1)
        else:
            # Blank phone with text lines only (no hero)
            pw = random.randint(180, 260)
            ph = int(pw * 2.0)
            px = random.randint(100, 350)
            py = random.randint(50, 200)
            cv2.rectangle(bg_img, (px, py), (px + pw, py + ph), (25, 25, 25), -1)
            cv2.rectangle(bg_img, (px + 6, py + 6), (px + pw - 6, py + ph - 6), (240, 240, 240), -1)
            # Dummy text lines
            for ly in range(py + 30, py + ph - 20, 14):
                lw = random.randint(int(pw * 0.4), int(pw * 0.8))
                cv2.rectangle(bg_img, (px + 15, ly), (px + 15 + lw, ly + 6), (180, 180, 180), -1)

        img_filename = f"bg_tr_{sample_counter:06d}.jpg"
        lbl_filename = f"bg_tr_{sample_counter:06d}.txt"
        cv2.imwrite(os.path.join(train_img_dir, img_filename), bg_img)
        with open(os.path.join(train_lbl_dir, lbl_filename), "w") as f:
            pass

    for bg_idx in range(8):
        sample_counter += 1
        total_val_samples += 1
        bg_img = np.full((640, 640, 3), random.randint(130, 210), dtype=np.uint8)
        img_filename = f"bg_val_{sample_counter:06d}.jpg"
        lbl_filename = f"bg_val_{sample_counter:06d}.txt"
        cv2.imwrite(os.path.join(val_img_dir, img_filename), bg_img)
        with open(os.path.join(val_lbl_dir, lbl_filename), "w") as f:
            pass

    abs_dataset_path = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""path: {abs_dataset_path}
train: images/train
val: images/val

names:
  0: soekarno
  1: dewantara
  2: kartini
  3: soedirman
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset generated. Train samples: {total_train_samples}, Val samples: {total_val_samples}")
    print(f"Dataset YAML written to {yaml_path}")

if __name__ == "__main__":
    build_curated_yolo_dataset()
