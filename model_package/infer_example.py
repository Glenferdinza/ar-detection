import os
import sys
import json
import cv2
import numpy as np

CLASSES = ["soekarno", "dewantara", "kartini", "soedirman", "negative"]

def load_ar_mapping(mapping_path="ar_mapping.json"):
    if not os.path.exists(mapping_path):
        mapping_path = os.path.join(os.path.dirname(__file__), "ar_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["class_id"]: c for c in data["classes"]}

def nms_boxes(candidates, iou_thresh=0.45):
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda b: b["conf"], reverse=True)
    keep = []
    while candidates:
        best = candidates.pop(0)
        keep.append(best)
        remaining = []
        bx1, by1, bx2, by2 = best["xyxy"]
        best_area = max(1, (bx2 - bx1) * (by2 - by1))
        for b in candidates:
            x1, y1, x2, y2 = b["xyxy"]
            ix1, iy1 = max(bx1, x1), max(by1, y1)
            ix2, iy2 = min(bx2, x2), min(by2, y2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            area_b = max(1, (x2 - x1) * (y2 - y1))
            iou = inter / float(best_area + area_b - inter)
            if iou < iou_thresh:
                remaining.append(b)
        candidates = remaining
    return keep

class ProductionARInference:
    def __init__(self, backend="onnx", model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))

        self.backend = backend
        self.ar_map = load_ar_mapping(os.path.join(model_dir, "ar_mapping.json"))

        if backend == "onnx":
            import onnxruntime as ort
            det_path = os.path.join(model_dir, "onnx", "hero_detector.onnx")
            ver_path = os.path.join(model_dir, "onnx", "hero_verifier.onnx")
            self.sess_det = ort.InferenceSession(det_path, providers=["CPUExecutionProvider"])
            self.sess_ver = ort.InferenceSession(ver_path, providers=["CPUExecutionProvider"])
        elif backend == "pytorch":
            import torch
            import torchvision.models as models
            import torch.nn as nn
            from ultralytics import YOLO
            det_path = os.path.join(model_dir, "pytorch", "hero_detector.pt")
            ver_path = os.path.join(model_dir, "pytorch", "hero_verifier.pt")
            self.yolo = YOLO(det_path)

            v_model = models.mobilenet_v3_small(weights=None)
            in_features = v_model.classifier[3].in_features
            v_model.classifier[3] = nn.Linear(in_features, len(CLASSES))
            v_model.load_state_dict(torch.load(ver_path, map_location="cpu"))
            v_model.eval()
            self.v_model = v_model
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _verify_crop(self, crop_bgr, yolo_cls, yolo_conf):
        h, w = crop_bgr.shape[:2]
        if h < 25 or w < 25:
            return False, yolo_cls, yolo_conf, {"reason": "too_small"}

        resized = cv2.resize(crop_bgr, (128, 128), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        if self.backend == "onnx":
            out = self.sess_ver.run(None, {"input": tensor})[0]
            logits = out[0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
        else:
            import torch
            with torch.no_grad():
                out = self.v_model(torch.from_numpy(tensor))
                probs = torch.softmax(out, dim=1).numpy()[0]

        top_cls = int(np.argmax(probs))
        neg_prob = float(probs[4])

        if top_cls == 4 or neg_prob > 0.35:
            return False, yolo_cls, yolo_conf, {"reason": "living_human_shield", "probs": probs.tolist()}

        hero_probs = probs[:4]
        best_hero_cls = int(np.argmax(hero_probs))
        best_hero_conf = float(hero_probs[best_hero_cls])

        if best_hero_conf < 0.50 or (best_hero_conf - neg_prob) < 0.15:
            return False, yolo_cls, yolo_conf, {"reason": "insufficient_hero_confidence", "probs": probs.tolist()}

        if best_hero_cls == yolo_cls:
            final_cls = yolo_cls
            final_conf = (yolo_conf * 0.40) + (best_hero_conf * 0.60)
        else:
            if best_hero_conf >= 0.70:
                final_cls = best_hero_cls
                final_conf = best_hero_conf
            elif yolo_conf >= 0.82:
                final_cls = yolo_cls
                final_conf = yolo_conf
            else:
                return False, yolo_cls, yolo_conf, {"reason": "conflict_unresolved", "probs": probs.tolist()}

        return True, final_cls, float(final_conf), {"probs": probs.tolist()}

    def predict(self, frame_bgr, conf_thresh=0.45):
        h_orig, w_orig = frame_bgr.shape[:2]
        candidates = []

        if self.backend == "pytorch":
            results = self.yolo.predict(frame_bgr, conf=max(0.20, conf_thresh - 0.20), imgsz=416, verbose=False)[0]
            for b in results.boxes:
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
                bw, bh = x2 - x1, y2 - y1
                if bw < 25 or bh < 25 or (bh / float(max(1, bw))) > 2.15:
                    continue
                candidates.append({"cls": cls_id, "conf": conf, "xyxy": [max(0, x1), max(0, y1), min(w_orig, x2), min(h_orig, y2)]})
        else:
            input_tensor = cv2.resize(frame_bgr, (416, 416))
            input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, ...]

            preds = self.sess_det.run(None, {"images": input_tensor})[0][0]
            boxes = preds[:4, :].T
            scores = preds[4:, :].T

            gain_x = w_orig / 416.0
            gain_y = h_orig / 416.0

            for i in range(len(boxes)):
                cls_id = int(np.argmax(scores[i]))
                score = float(scores[i][cls_id])
                if score < max(0.20, conf_thresh - 0.20):
                    continue

                cx, cy, bw, bh = boxes[i]
                x1 = int((cx - bw / 2.0) * gain_x)
                y1 = int((cy - bh / 2.0) * gain_y)
                x2 = int((cx + bw / 2.0) * gain_x)
                y2 = int((cy + bh / 2.0) * gain_y)

                bw_px = x2 - x1
                bh_px = y2 - y1

                if bw_px < 25 or bh_px < 25 or (bh_px / float(max(1, bw_px))) > 2.15:
                    continue

                candidates.append({
                    "cls": cls_id,
                    "conf": score,
                    "xyxy": [max(0, x1), max(0, y1), min(w_orig, x2), min(h_orig, y2)]
                })

        candidates = nms_boxes(candidates, iou_thresh=0.45)
        verified = []

        for cand in candidates:
            x1, y1, x2, y2 = cand["xyxy"]
            crop = frame_bgr[y1:y2, x1:x2]
            valid, v_cls, v_conf, meta = self._verify_crop(crop, cand["cls"], cand["conf"])
            if valid and v_conf >= conf_thresh:
                ar_meta = self.ar_map.get(v_cls, {})
                verified.append({
                    "class_id": v_cls,
                    "hero_name": ar_meta.get("display_name", f"Hero_{v_cls}"),
                    "ar_type": ar_meta.get("ar_type", "none"),
                    "asset": ar_meta.get("default_asset", ""),
                    "confidence": round(v_conf, 4),
                    "bbox": cand["xyxy"]
                })

        return verified

if __name__ == "__main__":
    test_img = "crop_check.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]

    if not os.path.exists(test_img):
        print(f"Sample image not found at {test_img}")
        sys.exit(0)

    engine = ProductionARInference(backend="onnx")
    frame = cv2.imread(test_img)
    detections = engine.predict(frame, conf_thresh=0.45)

    print(f"Processed image: {test_img}")
    if not detections:
        print("Status: NOT_DETECTED (Filter Living Human / No Hero)")
    else:
        for d in detections:
            print(f"Found: {d['hero_name']} (Conf: {d['confidence']*100:.1f}%)")
            print(f"BBox: {d['bbox']}")
            print(f"AR Target: {d['ar_type']} -> {d['asset']}")
