import os
import io
import json
import time
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from preprocessor import ArtefactPreprocessor
from two_stage_verifier import TwoStageHeroVerifier

app = FastAPI(title="National Heroes AR Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/media", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

with open("data/metadata.json", "r", encoding="utf-8") as f:
    METADATA = json.load(f)

HERO_MAP = {h["class_id"]: h for h in METADATA["heroes"]}
HERO_NAME_MAP = {h["name"].lower(): h for h in METADATA["heroes"]}

MODEL_PATH = "weights/best.pt" if os.path.exists("weights/best.pt") else "yolov8n.pt"
print(f"Loading detector from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

STAGE2_PATH = "weights/stage2_classifier.pt"
verifier = TwoStageHeroVerifier(STAGE2_PATH, beta_thresh=10.0) if os.path.exists(STAGE2_PATH) else None

preprocessor = ArtefactPreprocessor(beta_passing_grade=50.0, target_size=(640, 640))

@app.get("/")
def health_check():
    return {"status": "ONLINE", "model": MODEL_PATH, "classes": len(HERO_MAP)}

@app.post("/rest_api/last_request")
@app.post("/api/v1/detect")
async def detect_hero_artifact(
    file: UploadFile = File(...),
    usr_id: str = Form("anonymous_visitor")
):
    start_time = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    passed, preprocessed_img, quality_info = preprocessor.process(image_bgr)
    if not passed or preprocessed_img is None:
        return {
            "odr_id": f"ODR_{int(time.time() * 1000)}",
            "odr_usr_id": usr_id,
            "odr_status": "REJECTED_QUALITY",
            "odr_message": "Artefak citra terlalu buram atau pencahayaan kurang optimal. Posisikan kamera lebih stabil.",
            "assessment_beta": quality_info
        }

    results = model.predict(preprocessed_img, conf=0.30, imgsz=416, verbose=False)
    boxes = results[0].boxes

    h_img, w_img = preprocessed_img.shape[:2]
    total_area = h_img * w_img
    valid_candidates = []

    for b in boxes:
        conf = float(b.conf[0])
        class_id = int(b.cls[0])
        xyxy = [int(v) for v in b.xyxy[0].cpu().numpy()]
        x1, y1, x2, y2 = xyxy
        bw, bh = x2 - x1, y2 - y1

        if bw < 25 or bh < 25:
            continue
        if (bw * bh) > 0.98 * total_area:
            continue
        if (bh / float(max(1, bw))) > 2.15:
            continue

        valid_candidates.append({
            "cls": class_id,
            "conf": conf,
            "xyxy": xyxy
        })

    if len(valid_candidates) == 0:
        return {
            "odr_id": f"ODR_{int(time.time() * 1000)}",
            "odr_usr_id": usr_id,
            "odr_status": "NOT_DETECTED",
            "odr_message": "Objek pahlawan nasional tidak terdeteksi pada artefak.",
            "assessment_beta": quality_info
        }

    valid_candidates.sort(key=lambda c: c["conf"], reverse=True)
    verified_box = None
    verifier_meta = {}

    for cand in valid_candidates:
        x1, y1, x2, y2 = cand["xyxy"]
        crop = preprocessed_img[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]

        if verifier is not None:
            valid, v_cls, v_conf, v_meta = verifier.verify_candidate(crop, yolo_cls_id=cand["cls"], yolo_conf=cand["conf"])
            if valid:
                verified_box = {
                    "cls": v_cls,
                    "conf": v_conf,
                    "xyxy": cand["xyxy"]
                }
                verifier_meta = v_meta
                break
            else:
                verifier_meta = v_meta
        else:
            verified_box = cand
            break

    if verified_box is None:
        return {
            "odr_id": f"ODR_{int(time.time() * 1000)}",
            "odr_usr_id": usr_id,
            "odr_status": "NOT_DETECTED",
            "odr_message": "Objek pahlawan nasional tidak terverifikasi (terfilter oleh Living Human Shield / artefak non-pahlawan).",
            "assessment_beta": quality_info,
            "verifier_meta": verifier_meta
        }

    class_id = verified_box["cls"]
    conf = verified_box["conf"]
    xyxy = verified_box["xyxy"]

    hero_info = HERO_MAP.get(class_id, {
        "name": f"Hero_{class_id}",
        "title": "Tokoh Nasional",
        "bio": "Informasi artefak pahlawan nasional.",
        "ar_content": {"type": "illustration_2d", "asset_url": "/static/media/placeholder.png"}
    })

    elapsed_ms = (time.time() - start_time) * 1000.0

    return {
        "odr_id": f"ODR_{int(time.time() * 1000)}",
        "odr_usr_id": usr_id,
        "odr_status": "SUCCESS",
        "odr_hero": hero_info["name"],
        "odr_title": hero_info["title"],
        "odr_info": hero_info["bio"],
        "odr_media": hero_info["ar_content"],
        "odr_media_suite": hero_info.get("media_suite", {}),
        "bbox": [round(float(c), 2) for c in xyxy],
        "confidence": round(conf, 4),
        "inference_time_ms": round(elapsed_ms, 2),
        "assessment_beta": quality_info,
        "verifier_meta": verifier_meta
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
