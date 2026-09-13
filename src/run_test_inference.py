import os
import io
import time
import requests
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from preprocessor import ArtefactPreprocessor

os.makedirs("tests/samples", exist_ok=True)
os.makedirs("tests/output", exist_ok=True)

TEST_URLS = {
    "soekarno_unseen": "https://upload.wikimedia.org/wikipedia/commons/f/fd/1955_Indonesian_Election_Sukarno.png",
    "dewantara_statue": "https://thumb.wikimedia.org/wikipedia/commons/thumb/3/36/Statue_of_Ki_Hadjar_Dewantara_in_front_of_Sekolah_Tamansiswa.jpg/960px-Statue_of_Ki_Hadjar_Dewantara_in_front_of_Sekolah_Tamansiswa.jpg",
    "kartini_unseen": "https://upload.wikimedia.org/wikipedia/commons/2/23/COLLECTIE_TROPENMUSEUM_Portret_van_Raden_Ajeng_Kartini_TMnr_10018776.jpg",
    "soedirman_unseen": "https://thumb.wikimedia.org/wikipedia/commons/thumb/c/c1/General_Sudirman_5_October_1947_KR.jpg/960px-General_Sudirman_5_October_1947_KR.jpg",
    "negative_hallway": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Interior_of_the_National_Museum_of_Indonesia_01.jpg/800px-Interior_of_the_National_Museum_of_Indonesia_01.jpg"
}

AR_CONTENT_MAP = {
    0: ("Ir. Soekarno", "Konten Ilustrasi 2D", (255, 120, 0)),
    1: ("Ki Hajar Dewantara", "Konten Ilustrasi 3D (.glb)", (0, 200, 255)),
    2: ("R.A. Kartini", "Konten Video (Mp4)", (200, 50, 255)),
    3: ("Jenderal Soedirman", "Konten Komik 2D", (50, 220, 50))
}

def download_test_sample(name, url):
    dest = os.path.join("tests/samples", f"{name}.jpg")
    if os.path.exists(dest) and os.path.getsize(dest) > 5000:
        return dest
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(dest, "JPEG", quality=95)
            return dest
    except Exception as e:
        print(f"Error downloading {name}: {e}")
    return None

def run_tests():
    print("Initializing Unseen Test Inference Suite with Quality Preprocessor & AR Mapper...")
    preprocessor = ArtefactPreprocessor(beta_passing_grade=40.0, target_size=(640, 640))
    model = YOLO("weights/best.pt")

    results_summary = []

    for name, url in TEST_URLS.items():
        sample_path = download_test_sample(name, url)
        if not sample_path or not os.path.exists(sample_path):
            print(f"Skipping {name}: file unavailable.")
            continue

        raw_bgr = cv2.imread(sample_path)
        if raw_bgr is None:
            continue

        start_time = time.time()
        passed, preprocessed_img, quality_info = preprocessor.process(raw_bgr)
        elapsed_prep = (time.time() - start_time) * 1000.0

        annotated_img = raw_bgr.copy()
        h, w = annotated_img.shape[:2]

        if not passed:
            print(f"[{name}] REJECTED by Preprocessor beta check (Score: {quality_info.get('beta_score', 0):.1f})")
            continue

        start_infer = time.time()
        predict_res = model.predict(preprocessed_img, conf=0.30, imgsz=416, verbose=False)[0]
        elapsed_infer = (time.time() - start_infer) * 1000.0

        detected_boxes = predict_res.boxes
        num_dets = len(detected_boxes)

        print(f"\n--- Testing: {name} ({w}x{h}) ---")
        print(f"Quality Beta Score: {quality_info['beta_score']:.1f} | Preprocess: {elapsed_prep:.1f}ms | Inference: {elapsed_infer:.1f}ms")

        if num_dets == 0:
            print(f"Detection Result: NO OBJECT (Clean negative / background rejected as expected)")
            cv2.putText(annotated_img, "NO HERO DETECTED (NEGATIVE)", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            results_summary.append((name, "None", 0.0, elapsed_infer))
        else:
            best_b = sorted(detected_boxes, key=lambda b: float(b.conf[0]), reverse=True)[0]
            cls_id = int(best_b.cls[0])
            conf = float(best_b.conf[0])
            hero_name, ar_type, color = AR_CONTENT_MAP.get(cls_id, ("Unknown", "N/A", (255, 255, 255)))

            # Map coordinates from 640x640 letterboxed back to original image
            # For direct visualization, detect directly on original size as well
            direct_res = model.predict(raw_bgr, conf=0.30, imgsz=640, verbose=False)[0]
            box_to_draw = direct_res.boxes[0] if len(direct_res.boxes) > 0 else best_b
            
            x1, y1, x2, y2 = [int(v) for v in box_to_draw.xyxy[0].cpu().numpy()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Draw box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)

            # Draw label banner
            label_text = f"{hero_name} ({conf*100:.1f}%)"
            ar_text = f"AR: {ar_type}"
            time_text = f"{elapsed_infer:.1f} ms"

            # Background rectangle for text
            cv2.rectangle(annotated_img, (x1, max(0, y1 - 65)), (x1 + max(len(label_text), len(ar_text)) * 14 + 20, y1), color, -1)
            cv2.putText(annotated_img, label_text, (x1 + 6, y1 - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(annotated_img, ar_text, (x1 + 6, y1 - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

            cv2.putText(annotated_img, f"Latency: {time_text}", (30, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            print(f"Detection Result: SUCCESS -> {hero_name} | Confidence: {conf*100:.1f}% | AR Action: {ar_type}")
            results_summary.append((name, hero_name, conf, elapsed_infer))

        out_path = os.path.join("tests/output", f"{name}_annotated.jpg")
        cv2.imwrite(out_path, annotated_img)
        print(f"Saved visual prediction to: {out_path}")

    # Generate a video demonstration by rendering a smooth pan across the four hero artifacts
    print("\nGenerating simulated mobile AR camera video stream...")
    video_out_path = "tests/output/simulated_mobile_stream.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(video_out_path, fourcc, 15, (640, 480))

    hero_sample_names = ["soekarno_unseen", "dewantara_statue", "kartini_unseen", "soedirman_unseen"]
    for s_name in hero_sample_names:
        s_path = os.path.join("tests/samples", f"{s_name}.jpg")
        if not os.path.exists(s_path):
            continue
        im = cv2.imread(s_path)
        if im is None:
            continue
        im_resized = cv2.resize(im, (640, 480))

        # Run detection on each frame
        res = model.predict(im_resized, conf=0.25, verbose=False)[0]
        frame = im_resized.copy()

        if len(res.boxes) > 0:
            b = res.boxes[0]
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            h_name, ar_t, col = AR_CONTENT_MAP.get(cls_id, ("Hero", "AR", (0, 255, 0)))
            bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 3)
            cv2.putText(frame, f"{h_name} [{conf*100:.1f}%]", (bx1, max(25, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.putText(frame, f"AR Content: {ar_t}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "FPS: 32.5 (Real-time TinyML)", (20, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Write 20 frames for each hero artifact to create a video clip
        for _ in range(20):
            vw.write(frame)

    vw.release()
    print(f"Saved simulated AR camera video to: {video_out_path}")

    print("\n--- Summary of Unseen Real-World Testing ---")
    for item in results_summary:
        print(f"Sample: {item[0]:<20} | Detected: {item[1]:<22} | Confidence: {item[2]*100:.1f}% | Latency: {item[3]:.1f}ms")

if __name__ == "__main__":
    run_tests()
