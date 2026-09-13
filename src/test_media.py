import os
import sys
import argparse
import time
import cv2
from ultralytics import YOLO
from two_stage_verifier import TwoStageHeroVerifier
from temporal_tracker import ProductionARTracker

AR_CONTENT_MAP = {
    0: ("Ir. Soekarno", "Konten Ilustrasi 2D", (255, 120, 0)),
    1: ("Ki Hajar Dewantara", "Konten Ilustrasi 3D (.glb)", (0, 200, 255)),
    2: ("R.A. Kartini", "Konten Video (Mp4)", (200, 50, 255)),
    3: ("Jenderal Soedirman", "Konten Komik 2D", (50, 220, 50))
}

def draw_reticle(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    rw, rh = int(w * 0.54), int(h * 0.60)
    x1, y1 = cx - rw // 2, cy - rh // 2
    x2, y2 = cx + rw // 2, cy + rh // 2

    reticle_color = (180, 180, 180)
    line_len = 24
    thickness = 2

    # Reticle corners
    cv2.line(frame, (x1, y1), (x1 + line_len, y1), reticle_color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + line_len), reticle_color, thickness)
    cv2.line(frame, (x2, y1), (x2 - line_len, y1), reticle_color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + line_len), reticle_color, thickness)
    cv2.line(frame, (x1, y2), (x1 + line_len, y2), reticle_color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - line_len), reticle_color, thickness)
    cv2.line(frame, (x2, y2), (x2 - line_len, y2), reticle_color, thickness)
    cv2.line(frame, (x2, y2), (x2 - line_len, y2), reticle_color, thickness)

    guide_text = "AR TARGET: Arahkan Kamera ke Foto Pahlawan / Layar Gawai"
    tw = cv2.getTextSize(guide_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(frame, guide_text, (cx - tw // 2, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

def annotate_frame(frame, detections, elapsed_ms, conf_thresh=0.45, is_locked=False):
    h, w = frame.shape[:2]

    # HUD header
    cv2.rectangle(frame, (0, 0), (w, 42), (25, 25, 25), -1)
    cv2.line(frame, (0, 42), (w, 42), (60, 60, 60), 1)

    is_coasting = any(b.get('is_coasting', False) for b in detections)
    if is_coasting:
        status_text = "STATUS: ARTEFAK TERKUNCI (TRACKING MEMORY)"
        status_color = (0, 220, 255)
    elif is_locked:
        status_text = "STATUS: ARTEFAK TERKUNCI"
        status_color = (0, 255, 120)
    else:
        status_text = "STATUS: MENUNGGU ARTEFAK PAHLAWAN"
        status_color = (0, 200, 255)

    cv2.putText(frame, status_text, (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2)

    thresh_text = f"FILTER: {conf_thresh*100:.0f}% [Keys: -/+]"
    tw = cv2.getTextSize(thresh_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
    cv2.putText(frame, thresh_text, (w - tw - 15, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

    if not is_locked or len(detections) == 0:
        draw_reticle(frame)
        cv2.putText(frame, f"FPS: {1000.0/max(1.0, elapsed_ms):.1f} | Latency: {elapsed_ms:.1f} ms | Human Shield: AKTIF",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        return frame

    for b in detections:
        cls_id = b['cls']
        conf = b['conf']
        x1, y1, x2, y2 = b['xyxy']

        hero_name, ar_type, color = AR_CONTENT_MAP.get(cls_id, ("Pahlawan", "N/A", (0, 255, 0)))

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # AR card
        label = f"{hero_name} ({conf*100:.1f}%)"
        ar_lbl = f"AR ACTION: {ar_type}"
        card_w = max(len(label), len(ar_lbl)) * 12 + 25

        top_y = max(45, y1 - 55)
        cv2.rectangle(frame, (x1, top_y), (x1 + card_w, y1), color, -1)
        cv2.putText(frame, label, (x1 + 8, y1 - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
        cv2.putText(frame, ar_lbl, (x1 + 8, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)

    cv2.putText(frame, f"INFERENCE: {elapsed_ms:.1f} ms | FPS: {1000.0/max(1.0, elapsed_ms):.1f} | Real-Time Active",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 2)
    return frame

def balance_chroma(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(g3, 0.65, img_bgr, 0.35, 0)

def nms_boxes(boxes, iou_thresh=0.45):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b['conf'], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        remaining = []
        bx1, by1, bx2, by2 = best['xyxy']
        best_area = max(1, (bx2 - bx1) * (by2 - by1))
        for b in boxes:
            x1, y1, x2, y2 = b['xyxy']
            ix1, iy1 = max(bx1, x1), max(by1, y1)
            ix2, iy2 = min(bx2, x2), min(by2, y2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            area_b = max(1, (x2 - x1) * (y2 - y1))
            iou = inter / float(best_area + area_b - inter)
            if iou < iou_thresh:
                remaining.append(b)
        boxes = remaining
    return keep

def detect_heroes_fast(frame, model, verifier=None, conf_thresh=0.45, imgsz=416):
    h, w = frame.shape[:2]
    total_area = h * w
    max_allowed_area = 0.98 * total_area
    candidates = []

    # 1. Full-Frame Detection with Chromatic Balance
    balanced_full = balance_chroma(frame)
    res_full = model.predict(balanced_full, conf=max(0.20, conf_thresh - 0.15), imgsz=imgsz, verbose=False)[0]
    for b in res_full.boxes:
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
        bw, bh = x2 - x1, y2 - y1
        if bw < 25 or bh < 25:
            continue
        # Suppress oversized human body / room bounding boxes
        if (bw * bh) > max_allowed_area:
            continue
        # Suppress vertical standing human bodies from head to toe (aspect ratio > 2.15)
        if (bh / float(max(1, bw))) > 2.15:
            continue
        candidates.append({
            'cls': cls_id,
            'conf': conf,
            'xyxy': [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
        })

    # 2. Dual-Zone Reticle Priority Scan (central targeting area)
    # Only triggered if full-frame produced no candidates, avoiding redundant second pass latency
    if len(candidates) == 0:
        cx, cy = w // 2, h // 2
        rw, rh = int(w * 0.58), int(h * 0.65)
        rx1, ry1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
        rx2, ry2 = min(w, cx + rw // 2), min(h, cy + rh // 2)

        reticle_crop = frame[ry1:ry2, rx1:rx2]
        if reticle_crop.shape[0] >= 50 and reticle_crop.shape[1] >= 50:
            reticle_balanced = balance_chroma(reticle_crop)
            reticle_conf = max(0.18, conf_thresh - 0.15)
            res_reticle = model.predict(reticle_balanced, conf=reticle_conf, imgsz=imgsz, verbose=False)[0]
            for b in res_reticle.boxes:
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                lx1, ly1, lx2, ly2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
                lbw, lbh = lx2 - lx1, ly2 - ly1
                if lbw < 25 or lbh < 25:
                    continue
                gx1 = max(0, rx1 + lx1)
                gy1 = max(0, ry1 + ly1)
                gx2 = min(w, rx1 + lx2)
                gy2 = min(h, ry1 + ly2)
                if (gx2 - gx1) * (gy2 - gy1) > max_allowed_area:
                    continue
                if (lbh / float(max(1, lbw))) > 2.15:
                    continue
                candidates.append({
                    'cls': cls_id,
                    'conf': conf,
                    'xyxy': [gx1, gy1, gx2, gy2]
                })

    nms_proposals = nms_boxes(candidates)
    if verifier is None:
        return [b for b in nms_proposals if b['conf'] >= conf_thresh]

    verified = []
    for b in nms_proposals:
        x1, y1, x2, y2 = b['xyxy']
        crop = frame[y1:y2, x1:x2]
        is_hero, v_cls, v_conf, meta = verifier.verify_candidate(crop, yolo_cls_id=b['cls'], yolo_conf=b['conf'])
        if is_hero and v_conf >= conf_thresh:
            b['cls'] = v_cls
            b['conf'] = v_conf
            verified.append(b)

    return verified


def process_source(source_path, weights_path="weights/best.pt", conf_thresh=0.45, imgsz=416):
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    print(f"Loading detector from {weights_path}...")
    model = YOLO(weights_path)
    verifier = TwoStageHeroVerifier(model_path="weights/stage2_classifier.pt", beta_thresh=10.0)

    # Case 1: Webcam live
    if source_path in ["0", "webcam"]:
        print("Opening live webcam feed...")
        print(f"Confidence Gatekeeper default: {conf_thresh*100:.0f}% | Production Tracker Active")
        print("Controls: Press '-' or '[' to decrease threshold, '+' or ']' to increase, 'q' to exit.")

        cap = cv2.VideoCapture(0)
        tracker = ProductionARTracker(ema_alpha=0.65, max_lost_frames=8, hysteresis_conf=max(0.20, conf_thresh - 0.20))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            raw_boxes = detect_heroes_fast(frame, model, verifier=verifier, conf_thresh=conf_thresh, imgsz=imgsz)
            valid_boxes = tracker.update(raw_boxes)
            ms = (time.time() - t0) * 1000.0

            is_locked = len(valid_boxes) > 0
            annotated = annotate_frame(frame, valid_boxes, ms, conf_thresh=conf_thresh, is_locked=is_locked)
            cv2.imshow("National Heroes AR Detector Live", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in [ord('+'), ord('='), ord(']')]:
                conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
                tracker.hysteresis_conf = max(0.20, conf_thresh - 0.20)
                print(f"Threshold adjusted to: {conf_thresh*100:.0f}%")
            elif key in [ord('-'), ord('_'), ord('[')]:
                conf_thresh = max(0.15, round(conf_thresh - 0.05, 2))
                tracker.hysteresis_conf = max(0.20, conf_thresh - 0.20)
                print(f"Threshold adjusted to: {conf_thresh*100:.0f}%")

        cap.release()
        cv2.destroyAllWindows()
        return

    # Case 2: Video file
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    if any(source_path.lower().endswith(ext) for ext in video_extensions):
        print(f"Processing video: {source_path}...")
        cap = cv2.VideoCapture(source_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_name = f"output_{os.path.basename(source_path)}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_name, fourcc, fps, (w, h))

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tracker = ProductionARTracker(ema_alpha=0.65, max_lost_frames=8, hysteresis_conf=max(0.20, conf_thresh - 0.20))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            t0 = time.time()
            raw_boxes = detect_heroes_fast(frame, model, verifier=verifier, conf_thresh=conf_thresh, imgsz=imgsz)
            valid_boxes = tracker.update(raw_boxes)
            ms = (time.time() - t0) * 1000.0

            is_locked = len(valid_boxes) > 0
            annotated = annotate_frame(frame, valid_boxes, ms, conf_thresh=conf_thresh, is_locked=is_locked)
            writer.write(annotated)
            if frame_count % 30 == 0:
                print(f"Processed frame {frame_count}/{total_frames}...")

        cap.release()
        writer.release()
        print(f"Finished processing video. Saved annotated video to: {out_name}")
        return

    # Case 3: Single Image file
    if os.path.isfile(source_path):
        raw_bgr = cv2.imread(source_path)
        if raw_bgr is None:
            print(f"Error: Could not read image at {source_path}")
            return

        h, w = raw_bgr.shape[:2]
        print(f"\n--- Testing Image: {source_path} ({w}x{h}) ---")

        t0 = time.time()
        valid_boxes = detect_heroes_fast(raw_bgr, model, verifier=verifier, conf_thresh=conf_thresh, imgsz=imgsz)
        ms = (time.time() - t0) * 1000.0

        if len(valid_boxes) == 0:
            print(f"Result: NO HERO DETECTED (Below threshold {conf_thresh*100:.0f}%)")
        else:
            for b in valid_boxes:
                cls_id = b['cls']
                conf = b['conf']
                hero, ar, _ = AR_CONTENT_MAP.get(cls_id, ("Hero", "N/A", None))
                print(f"Result: {hero.upper()} (Confidence: {conf*100:.1f}%) | AR Action: {ar} | Latency: {ms:.1f} ms")

        is_locked = len(valid_boxes) > 0
        annotated = annotate_frame(raw_bgr.copy(), valid_boxes, ms, conf_thresh=conf_thresh, is_locked=is_locked)
        out_name = f"output_{os.path.basename(source_path)}"
        cv2.imwrite(out_name, annotated)
        print(f"Saved annotated image with bounding box to: {out_name}\n")
        return

    print(f"Source not recognized or file not found: {source_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test National Heroes AR Detector on Photo, Video, or Webcam")
    parser.add_argument("--source", type=str, required=True, help="Path to image, video file, or 'webcam'")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default 0.45)")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Model weights path")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference resolution (default 416)")
    args = parser.parse_args()

    process_source(args.source, weights_path=args.weights, conf_thresh=args.conf, imgsz=args.imgsz)
