import os
import glob
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F

def build_hero_prototypes(review_dir="data/review", output_path="weights/hero_prototypes.pt"):
    print("Building hero prototype embeddings...")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).eval()
    extractor = torch.nn.Sequential(*list(model.children())[:-1])

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def extract(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0)
        with torch.no_grad():
            feat = extractor(tensor).squeeze()
            return F.normalize(feat, p=2, dim=0)

    categories = ["soekarno", "dewantara", "kartini", "soedirman"]
    prototypes = {}

    for cat in categories:
        cat_dir = os.path.join(review_dir, cat)
        files = glob.glob(os.path.join(cat_dir, "*.*"))
        embs = []
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                embs.append(extract(img))
        if len(embs) > 0:
            proto = F.normalize(torch.stack(embs).mean(dim=0), p=2, dim=0)
            prototypes[cat] = proto
            print(f"Class '{cat}': aggregated {len(embs)} reference images.")
        else:
            print(f"Warning: No images found for class '{cat}' in {cat_dir}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(prototypes, output_path)
    print(f"Prototypes successfully saved to: {output_path}")

if __name__ == "__main__":
    build_hero_prototypes()
