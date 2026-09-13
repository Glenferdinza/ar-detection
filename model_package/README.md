# Panduan Integrasi Model AR Pahlawan Nasional

Paket model ini berisi arsitektur Two-Stage Detection & Verification untuk mendeteksi 4 Pahlawan Nasional Indonesia pada aplikasi AR Mobile dan backend REST API.

## Struktur File

```
model_package/
├── tflite/
│   ├── hero_detector.tflite    # Stage 1: YOLOv8n (Float16, 6.1 MB)
│   └── hero_verifier.tflite    # Stage 2: MobileNetV3-Small (Float16, 3.0 MB)
├── onnx/
│   ├── hero_detector.onnx      # Stage 1: Detector ONNX opset 12 (12.1 MB)
│   ├── hero_verifier.onnx      # Stage 2: Classifier ONNX opset 12 (6.0 MB)
│   └── hero_verifier.onnx.data # Tensor weights pendukung ONNX
├── pytorch/
│   ├── hero_detector.pt        # Weights PyTorch YOLOv8n (6.2 MB)
│   └── hero_verifier.pt        # Weights PyTorch MobileNetV3 (6.2 MB)
├── ar_mapping.json             # Pemetaan class ID ke aset AR
├── pipeline_spec.json          # Parameter teknis dan konstanta normalisasi
├── infer_example.py            # Contoh script inferensi siap pakai (ONNX/PyTorch)
└── README.md                   # Dokumen integrasi ini
```

## Spesifikasi Model

### 1. Stage 1: Detector (hero_detector)
* Fungsi: Deteksi lokasi bounding box artefak pahlawan pada frame kamera.
* Input Shape: 1 x 3 x 416 x 416 (RGB).
* Normalisasi: pixel / 255.0.
* Output: Bounding boxes dan confidence score per kelas (0..3).
* Ambang Deteksi Awal (Acquisition): 0.45.
* Ambang Retensi (Hysteresis): 0.25 (mencegah flickering AR saat pergerakan kamera).

### 2. Stage 2: Verifier (hero_verifier)
* Fungsi: Memverifikasi keaslian artefak serta memblokir wajah pengunjung / non-pahlawan (Living Human Shield).
* Input Shape: 1 x 3 x 128 x 128 (Crop RoI hasil Stage 1 dalam format RGB).
* Normalisasi:
  * Scale: pixel / 255.0
  * Mean: [0.485, 0.456, 0.406]
  * Std: [0.229, 0.224, 0.225]
* Output: Softmax logits 5 kelas:
  * Index 0: Ir. Soekarno
  * Index 1: Ki Hajar Dewantara
  * Index 2: R.A. Kartini
  * Index 3: Jenderal Soedirman
  * Index 4: Negative (Wajah pengunjung, tangan, latar belakang museum)

## Aturan Logika Integrasi (Wajib Diterapkan)

Untuk mencegah salah deteksi pada pengunjung di museum dan menjaga FPS tinggi:

1. Filter Rasio Aspek (Aspect Ratio Check)
   Tolak setiap bounding box hasil Stage 1 jika:
   (box_height / box_width) > 2.15
   Tujuan: Mengabaikan tubuh manusia yang berdiri tegak (rasio potret artefak berada pada 0.70 - 1.95).

2. Guardrail Verifikasi Stage 2
   Kandidat dari Stage 1 hanya dinyatakan valid jika:
   * Kelas prediksi Stage 2 bukan Index 4 (Negative).
   * Nilai probabilitas negatif <= 0.35.
   * Nilai probabilitas pahlawan tertinggi >= 0.50.
   * Selisih probabilitas pahlawan terhadap probabilitas negatif >= 0.15.

3. Formula Fused Confidence
   Jika kelas Stage 1 cocok dengan kelas Stage 2:
   final_conf = (0.40 * yolo_conf) + (0.60 * verifier_conf)

## Pemetaan Aset AR (Content Action Map)

* Class 0 (soekarno): Konten Ilustrasi 2D (.png) -> /static/media/soekarno_2d.png
* Class 1 (dewantara): Konten Ilustrasi 3D (.glb / .gltf) -> /static/media/dewantara_3d.glb
* Class 2 (kartini): Konten Video (.mp4) -> /static/media/kartini_history.mp4
* Class 3 (soedirman): Konten Komik 2D (.png carousel) -> /static/media/soedirman_comic.png
* Class 4 (negative): Non-Hero / Visitor -> Jangan render konten AR.

## Contoh Eksekusi

Jalankan script contoh inferensi menggunakan python dan ONNX Runtime:

```bash
python infer_example.py path/to/image.jpg
```

Untuk memilih backend PyTorch:
Ganti inisialisasi pada infer_example.py menjadi ProductionARInference(backend="pytorch").
