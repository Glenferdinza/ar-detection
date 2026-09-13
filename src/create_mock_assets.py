import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_assets(output_dir="static/media"):
    os.makedirs(output_dir, exist_ok=True)

    img_2d = Image.new("RGBA", (512, 512), (25, 30, 45, 255))
    draw = ImageDraw.Draw(img_2d)
    draw.rectangle([20, 20, 492, 492], outline=(220, 180, 50), width=4)
    draw.ellipse([156, 120, 356, 320], fill=(200, 160, 120), outline=(255, 255, 255), width=3)
    draw.rectangle([180, 100, 332, 160], fill=(20, 20, 20))
    draw.text((160, 360), "IR. SOEKARNO", fill=(255, 255, 255))
    draw.text((130, 400), "ILUSTRASI 2D AR KONTEN", fill=(220, 180, 50))
    img_2d.save(os.path.join(output_dir, "soekarno_2d.png"))
    print("Generated soekarno_2d.png")

    comic_img = Image.new("RGB", (800, 600), (245, 240, 230))
    draw = ImageDraw.Draw(comic_img)
    draw.rectangle([10, 10, 385, 590], outline=(0, 0, 0), width=4)
    draw.rectangle([405, 10, 790, 590], outline=(0, 0, 0), width=4)
    draw.text((50, 40), "PANEL 1: PERANG GERILYA", fill=(0, 0, 0))
    draw.text((50, 100), "Jenderal Soedirman memimpin", fill=(50, 50, 50))
    draw.text((50, 130), "pasukan dari atas tandu...", fill=(50, 50, 50))
    draw.text((440, 40), "PANEL 2: PERTAHANKAN NKRI", fill=(0, 0, 0))
    draw.text((440, 100), "'Tempat saya yang terbaik", fill=(50, 50, 50))
    draw.text((440, 130), "adalah di tengah-tengah", fill=(50, 50, 50))
    draw.text((440, 160), "anak buah saya!'", fill=(180, 20, 20))
    comic_img.save(os.path.join(output_dir, "soedirman_comic.png"))
    print("Generated soedirman_comic.png")

    video_path = os.path.join(output_dir, "kartini_history.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10
    writer = cv2.VideoWriter(video_path, fourcc, fps, (400, 300))
    for i in range(30):
        frame = np.full((300, 400, 3), (40, 30, 60), dtype=np.uint8)
        cv2.putText(frame, "R.A. KARTINI", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Habis Gelap Terbitlah Terang", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 255), 1)
        cv2.putText(frame, f"AR Video Stream: Frame {i+1}", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        writer.write(frame)
    writer.release()
    print("Generated kartini_history.mp4")

    glb_header = b"glTF" + (2).to_bytes(4, "little") + (76).to_bytes(4, "little")
    json_chunk = b'{"asset":{"version":"2.0","generator":"AR-Pahlawan-Mock"},"scenes":[{"nodes":[0]}],"nodes":[{"name":"Dewantara_3D"}]}'
    padding = (4 - (len(json_chunk) % 4)) % 4
    json_chunk += b" " * padding
    chunk_len = len(json_chunk).to_bytes(4, "little")
    chunk_type = b"JSON"
    glb_content = glb_header + chunk_len + chunk_type + json_chunk
    with open(os.path.join(output_dir, "dewantara_3d.glb"), "wb") as f:
        f.write(glb_content)
    print("Generated dewantara_3d.glb")

if __name__ == "__main__":
    generate_assets()
