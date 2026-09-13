import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HEROES = [
    {"id": "soekarno", "name": "Ir. Soekarno", "title": "Proklamator Kemerdekaan", "quote": "Beri aku 10 pemuda, niscaya akan kuguncangkan dunia.", "color": (220, 120, 40)},
    {"id": "dewantara", "name": "Ki Hajar Dewantara", "title": "Bapak Pendidikan", "quote": "Ing ngarsa sung tulada, ing madya mangun karsa, tut wuri handayani.", "color": (40, 180, 220)},
    {"id": "kartini", "name": "R.A. Kartini", "title": "Pelopor Emansipasi Wanita", "quote": "Habis Gelap Terbitlah Terang.", "color": (200, 50, 220)},
    {"id": "soedirman", "name": "Jenderal Soedirman", "title": "Panglima Besar Gerilya", "quote": "Robek-robeklah badanku, potong-potonglah jasad ini, jiwaku tak dapat kamu bunuh.", "color": (50, 200, 80)}
]

def make_glb(name):
    glb_header = b"glTF" + (2).to_bytes(4, "little") + (76).to_bytes(4, "little")
    json_chunk = f'{{"asset":{{"version":"2.0","generator":"AR-Pahlawan-Mock"}},"scenes":[{{"nodes":[0]}}],"nodes":[{{"name":"{name}_3D"}}]}}'.encode("utf-8")
    padding = (4 - (len(json_chunk) % 4)) % 4
    json_chunk += b" " * padding
    chunk_len = len(json_chunk).to_bytes(4, "little")
    chunk_type = b"JSON"
    return glb_header + chunk_len + chunk_type + json_chunk

def generate_assets(output_dir="static/media"):
    os.makedirs(output_dir, exist_ok=True)

    for h in HEROES:
        hid = h["id"]
        hname = h["name"]
        color = h["color"]

        # 1. Ilustrasi 2D
        img_2d = Image.new("RGBA", (512, 512), (25, 30, 45, 255))
        draw = ImageDraw.Draw(img_2d)
        draw.rectangle([20, 20, 492, 492], outline=color, width=4)
        draw.ellipse([156, 120, 356, 320], fill=(200, 160, 120), outline=(255, 255, 255), width=3)
        draw.text((140, 350), hname.upper(), fill=(255, 255, 255))
        draw.text((120, 390), "KONTEN ILUSTRASI 2D", fill=color)
        draw.text((80, 430), f"Tokoh: {h['title']}", fill=(180, 180, 180))
        img_2d.save(os.path.join(output_dir, f"{hid}_2d.png"))

        # 2. Model 3D (.glb)
        glb_data = make_glb(hname)
        with open(os.path.join(output_dir, f"{hid}_3d.glb"), "wb") as f:
            f.write(glb_data)

        # 3. Konten Video (.mp4)
        video_path = os.path.join(output_dir, f"{hid}_history.mp4" if hid == "kartini" else f"{hid}_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 10, (400, 300))
        for i in range(25):
            frame = np.full((300, 400, 3), (35, 25, 45), dtype=np.uint8)
            cv2.putText(frame, hname.upper(), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, h["title"], (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            cv2.putText(frame, f"Frame {i+1} / 25", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
            cv2.putText(frame, "AR Video Player Active", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 1)
            writer.write(frame)
        writer.release()

        # 4. Komik 2D (.png)
        comic_img = Image.new("RGB", (800, 600), (245, 240, 230))
        draw = ImageDraw.Draw(comic_img)
        draw.rectangle([10, 10, 385, 590], outline=(0, 0, 0), width=4)
        draw.rectangle([405, 10, 790, 590], outline=(0, 0, 0), width=4)
        draw.text((40, 40), f"PANEL 1: {hname.upper()}", fill=(0, 0, 0))
        draw.text((40, 90), h["title"], fill=(80, 80, 80))
        draw.text((40, 140), "Kisah Perjuangan Bangsa", fill=(120, 120, 120))
        draw.text((430, 40), "PANEL 2: PESAN PERJUANGAN", fill=(0, 0, 0))
        draw.text((430, 90), f'"{h["quote"][:38]}..."', fill=(180, 30, 30))
        draw.text((430, 150), "Dokumentasi Museum Nasional", fill=(100, 100, 100))
        comic_img.save(os.path.join(output_dir, f"{hid}_comic.png"))

    print(f"Generated complete 4x4 multimedia suite ({len(HEROES)*4} assets) in {output_dir}")

if __name__ == "__main__":
    generate_assets()
