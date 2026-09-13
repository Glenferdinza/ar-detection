import glob
import os
import cv2
import numpy as np

def analyze_image(path):
    im = cv2.imread(path)
    if im is None:
        return "CORRUPT"
    h, w = im.shape[:2]

    # Color saturation analysis: Historical portraits (1880-1950) are almost exclusively monochrome/sepia/grayscale or low saturation.
    # Modern color photos (like the woman in bright red hijab or colorful batik) have very high HSV saturation and high variance!
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    mean_sat = np.mean(sat)
    max_sat = np.max(sat)

    # Detect bright red backgrounds or modern vibrant colors
    red_mask1 = cv2.inRange(hsv, (0, 120, 100), (12, 255, 255))
    red_mask2 = cv2.inRange(hsv, (168, 120, 100), (180, 255, 255))
    red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
    red_ratio = red_pixels / float(h * w)

    return {
        "path": path,
        "filename": os.path.basename(path),
        "size": f"{w}x{h}",
        "mean_sat": round(float(mean_sat), 2),
        "max_sat": int(max_sat),
        "red_ratio": round(float(red_ratio), 4)
    }

for h in ["soekarno", "dewantara", "kartini", "soedirman"]:
    files = sorted(glob.glob(f"data/curated/{h}/*.jpg"))
    print(f"=== {h} ({len(files)} files) ===")
    for f in files:
        res = analyze_image(f)
        flag = " [SUSPICIOUS COLOR/VIBRANT]" if (res["red_ratio"] > 0.10 or res["mean_sat"] > 70) else ""
        print(f"  {res['filename']}: {res['size']} sat={res['mean_sat']} red={res['red_ratio']}{flag}")
