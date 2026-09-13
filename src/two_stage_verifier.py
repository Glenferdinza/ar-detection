import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class TwoStageHeroVerifier:
    def __init__(self, model_path="weights/stage2_classifier.pt", beta_thresh=10.0, device="cpu"):
        self.device = torch.device(device)
        self.beta_thresh = beta_thresh
        self.preprocessor = ArtefactPreprocessor(beta_passing_grade=beta_thresh)
        self.model = None

        if os.path.exists(model_path):
            self._load_classifier(model_path)
        else:
            print(f"Warning: Stage 2 model not found at {model_path}")

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_classifier(self, model_path):
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, len(CLASSES))
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def verify_candidate(self, crop_bgr, yolo_cls_id=0, yolo_conf=0.50):
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
            return False, yolo_cls_id, yolo_conf, {"reason": "too_small"}

        # 1. Refine crop to trim dark screen borders or bezels
        refined = self.preprocessor.refine_inner_portrait(crop_bgr)

        # 2. Quality assessment
        is_artefact, quality_info = self.preprocessor.assess_quality(refined)
        if not is_artefact:
            return False, yolo_cls_id, yolo_conf, quality_info

        if self.model is None:
            return True, yolo_cls_id, yolo_conf, quality_info

        # 3. Chromatic balancing
        balanced = balance_chroma(refined)
        rgb = cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        probs_list = [float(p) for p in probs]
        top_cls = int(torch.argmax(probs).item())
        neg_prob = probs_list[4]

        # 4. Living human shield and background filter
        if top_cls == 4 or neg_prob > 0.35:
            quality_info["reason"] = "rejected_as_living_face_or_background"
            quality_info["probs"] = {CLASSES[i]: round(probs_list[i], 4) for i in range(5)}
            return False, yolo_cls_id, yolo_conf, quality_info

        # 5. National hero candidate verification
        hero_probs = probs_list[:4]
        best_hero_cls = int(np.argmax(hero_probs))
        best_hero_conf = hero_probs[best_hero_cls]

        # Production Grade: Ensure high hero confidence (> 0.50) and distinct margin over negative class
        if best_hero_conf < 0.50 or (best_hero_conf - neg_prob) < 0.15:
            quality_info["reason"] = "insufficient_hero_confidence"
            quality_info["probs"] = {CLASSES[i]: round(probs_list[i], 4) for i in range(5)}
            return False, yolo_cls_id, yolo_conf, quality_info

        if best_hero_cls == yolo_cls_id:
            final_cls = yolo_cls_id
            final_conf = (yolo_conf * 0.40) + (best_hero_conf * 0.60)
        else:
            if best_hero_conf >= 0.70:
                final_cls = best_hero_cls
                final_conf = best_hero_conf
            elif yolo_conf >= 0.82:
                final_cls = yolo_cls_id
                final_conf = yolo_conf
            else:
                quality_info["reason"] = "conflict_unresolved"
                quality_info["probs"] = {CLASSES[i]: round(probs_list[i], 4) for i in range(5)}
                return False, yolo_cls_id, yolo_conf, quality_info

        quality_info["probs"] = {CLASSES[i]: round(probs_list[i], 4) for i in range(5)}
        quality_info["verified_class"] = CLASSES[final_cls]
        quality_info["final_conf"] = final_conf
        return True, final_cls, final_conf, quality_info
