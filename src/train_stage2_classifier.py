import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from preprocessor import ArtefactPreprocessor

CLASSES = ["soekarno", "dewantara", "kartini", "soedirman", "negative"]

def balance_chroma(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(g3, 0.65, img_bgr, 0.35, 0)

def load_crop_from_box(img_path, box_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if os.path.exists(box_path):
        with open(box_path, "r") as f:
            line = f.readline().strip()
            if line:
                cleaned = line.replace(",", " ")
                parts = [float(p) for p in cleaned.split() if p.strip()]
                if len(parts) == 4:
                    if any(p > 1.0 for p in parts):
                        x1 = max(0, min(w - 10, int(parts[0])))
                        y1 = max(0, min(h - 10, int(parts[1])))
                        x2 = max(x1 + 10, min(w, int(parts[2])))
                        y2 = max(y1 + 10, min(h, int(parts[3])))
                    else:
                        bx, by, bw, bh = parts
                        x1 = max(0, int((bx - bw / 2.0) * w))
                        y1 = max(0, int((by - bh / 2.0) * h))
                        x2 = min(w, int((bx + bw / 2.0) * w))
                        y2 = min(h, int((by + bh / 2.0) * h))
                    if x2 - x1 > 20 and y2 - y1 > 20:
                        return img[y1:y2, x1:x2]
    pad = int(min(h, w) * 0.1)
    return img[pad:h-pad, pad:w-pad]

class HeroClassifierDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop_bgr, label = self.samples[idx]
        balanced = balance_chroma(crop_bgr)
        rgb = cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB)
        if self.transform:
            tensor = self.transform(rgb)
        else:
            tensor = T.functional.to_tensor(rgb)
        return tensor, label

def prepare_clean_dataset():
    samples = []
    print("Collecting clean hero crops from curated & multimodal archives...")

    # Classes 0-3: The 4 National Heroes
    for cls_id, cat in enumerate(CLASSES[:4]):
        base_files = []
        for src in ["data/curated", "data/multimodal"]:
            d = os.path.join(src, cat)
            if os.path.exists(d):
                base_files.extend(glob.glob(os.path.join(d, "*.jpg")))

        base_files = sorted(list(set(base_files)))
        print(f"  [{cat}] Processing {len(base_files)} base images...")

        for f in base_files:
            box_f = f.replace(".jpg", ".box")
            crop = load_crop_from_box(f, box_f)
            if crop is None or crop.size == 0 or crop.shape[0] < 25 or crop.shape[1] < 25:
                continue

            # Original
            samples.append((crop, cls_id))
            # Horizontal flip
            samples.append((cv2.flip(crop, 1), cls_id))

            # Rotated variants (-12, +12 deg)
            h, w = crop.shape[:2]
            for ang in [-12.0, 12.0]:
                M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
                rot = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                samples.append((rot, cls_id))

            # Scaled / Zoomed
            zh, zw = int(h * 0.88), int(w * 0.88)
            if zh > 20 and zw > 20:
                zy, zx = (h - zh) // 2, (w - zw) // 2
                samples.append((crop[zy:zy+zh, zx:zx+zw], cls_id))

    # Also add the verified phone Kartini crop as positive Kartini sample!
    if os.path.exists("crop_check.jpg"):
        k_crop = cv2.imread("crop_check.jpg")
        if k_crop is not None:
            for _ in range(8):
                samples.append((k_crop, 2))
                samples.append((cv2.flip(k_crop, 1), 2))

    # Class 4: Dedicated Diverse Negative Samples (Full-body, torso, diverse human faces, room backgrounds)
    print("Collecting diverse negative samples (diverse human visitors & backgrounds)...")
    neg_paths = sorted(glob.glob("data/negative_faces/*.jpg"))
    for np_path in neg_paths:
        im = cv2.imread(np_path)
        if im is None:
            continue
        ih, iw = im.shape[:2]
        # 1. Full image / full body
        samples.append((im, 4))
        samples.append((cv2.flip(im, 1), 4))

        # 2. Upper body / Torso crop
        if ih > 100 and iw > 80:
            torso_y1 = int(ih * 0.15)
            torso_y2 = int(ih * 0.75)
            torso_crop = im[torso_y1:torso_y2, :]
            if torso_crop.shape[0] > 25 and torso_crop.shape[1] > 25:
                samples.append((torso_crop, 4))
                samples.append((cv2.flip(torso_crop, 1), 4))

        # 3. Head / Face region crop
        if ih > 100 and iw > 80:
            head_y2 = int(ih * 0.45)
            head_crop = im[:head_y2, :]
            if head_crop.shape[0] > 25 and head_crop.shape[1] > 25:
                samples.append((head_crop, 4))
                samples.append((cv2.flip(head_crop, 1), 4))

    # Also include verified user face if available
    if os.path.exists("scratch/verified_user_face.jpg"):
        u_face = cv2.imread("scratch/verified_user_face.jpg")
        if u_face is not None:
            samples.append((u_face, 4))
            samples.append((cv2.flip(u_face, 1), 4))

    # Synthetic room patterns & silhouettes
    for _ in range(50):
        bg = np.full((128, 128, 3), random.randint(70, 220), dtype=np.uint8)
        cv2.line(bg, (random.randint(0, 128), random.randint(0, 128)),
                 (random.randint(0, 128), random.randint(0, 128)),
                 (random.randint(30, 90), random.randint(30, 90), random.randint(30, 90)), 3)
        samples.append((bg, 4))

    # Blank phone device screens
    for _ in range(30):
        screen = np.full((128, 90, 3), random.choice([25, 230]), dtype=np.uint8)
        for ly in range(20, 110, 12):
            cv2.line(screen, (10, ly), (random.randint(40, 80), ly), (120, 120, 120), 2)
        samples.append((screen, 4))

    random.shuffle(samples)
    print(f"Total clean Stage 2 samples prepared: {len(samples)}")
    return samples

def train_stage2_classifier(epochs=18, batch_size=32, lr=0.0008):
    random.seed(42)
    torch.manual_seed(42)

    samples = prepare_clean_dataset()
    split_idx = int(len(samples) * 0.85)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),
        T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.06),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(HeroClassifierDataset(train_samples, train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(HeroClassifierDataset(val_samples, val_transform), batch_size=batch_size, shuffle=False)

    print("Initializing MobileNetV3-Small with Pretrained Weights...", flush=True)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_model_state = None

    print(f"Training Stage 2 Classifier across {epochs} epochs...", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%", flush=True)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    os.makedirs("weights", exist_ok=True)
    out_path = "weights/stage2_classifier.pt"
    torch.save(best_model_state, out_path)
    print(f"Best Stage 2 Classifier saved to {out_path} (Best Val Acc: {best_val_acc*100:.1f}%)", flush=True)

if __name__ == "__main__":
    train_stage2_classifier()
