import os
import glob
import cv2
import torch
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
resnet = models.resnet18(weights='DEFAULT')
resnet.fc = torch.nn.Identity()
resnet.eval()

def get_emb(p):
    im = Image.open(p).convert('RGB')
    t = transform(im).unsqueeze(0)
    with torch.no_grad():
        return torch.nn.functional.normalize(resnet(t), p=2, dim=1)

deleted_count = 0
kept_count = 0

for h in ['soekarno', 'dewantara', 'kartini', 'soedirman']:
    ref = get_emb(f'data/references/{h}.jpg')
    files = sorted(glob.glob(f'data/curated/{h}/*.jpg'))
    for f in files:
        im = cv2.imread(f)
        h_px, w_px = im.shape[:2]
        aspect = w_px / float(h_px)
        sim = float(torch.mm(get_emb(f), ref.T)[0][0])

        box_file = f.replace('.jpg', '.box')

        if aspect > 1.15 or aspect < 0.50 or sim < 0.68:
            print(f"REMOVING non-hero / distorted: {f} (aspect={aspect:.2f}, sim={sim:.3f})")
            if os.path.exists(f):
                os.remove(f)
            if os.path.exists(box_file):
                os.remove(box_file)
            deleted_count += 1
        else:
            print(f"KEEPING verified pure hero: {f} (aspect={aspect:.2f}, sim={sim:.3f})")
            kept_count += 1

print(f"Purge complete: Removed {deleted_count} spurious files. Kept {kept_count} pristine authentic hero portraits.")
