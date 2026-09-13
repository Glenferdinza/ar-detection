# AI-Driven Mobile AR for Indonesian National Heroes Detection

Sistem deteksi dan verifikasi terpadu berbasis Two-Stage Deep Learning untuk mendeteksi artefak 4 Tokoh Pahlawan Nasional Indonesia serta menyajikan konten Augmented Reality (AR) secara real-time pada perangkat mobile dan cloud REST API.

## Arsitektur Sistem

Pipeline menggunakan pendekatan dua tahap (Two-Stage Cascade):

1. Stage 1: Spatial Proposer (YOLOv8n)
   * Resolusi input: 416x416 RGB
   * Kecepatan inferensi: ~28 ms pada CPU (>35 FPS)
   * Tugas: Mendeteksi lokasi bounding box kandidat pahlawan secara cepat.
   * Filter Geometri: Membuang bounding box dengan aspek rasio (height / width) > 2.15 untuk mengeliminasi tubuh pengunjung museum yang berdiri tegak.

2. Stage 2: Fine-Grained Verifier & Living Human Shield (MobileNetV3-Small)
   * Resolusi input: 128x128 RGB
   * Klasifikasi 5 Kelas:
     - Class 0: Ir. Soekarno
     - Class 1: Ki Hajar Dewantara
     - Class 2: R.A. Kartini
     - Class 3: Jenderal Soedirman
     - Class 4: Negative (Wajah pengunjung hidup, tangan, ruangan, artefak non-pahlawan)
   * Guardrail: Memerlukan probabilitas pahlawan >= 0.50 dan margin selisih terhadap kelas negatif >= 0.15.

3. Temporal AR Tracker (ProductionARTracker)
   * Dual-Threshold Hysteresis: Ambang akuisisi 0.45, ambang retensi 0.25 untuk mencegah AR berkedip (flickering).
   * Exponential Moving Average (EMA, alpha=0.65): Menghaluskan koordinat bounding box tanpa getaran.
   * 8-Frame Coasting Memory: Menjaga jangkar AR tetap terkunci saat terjadi gerakan cepat tangan atau blur sesaat (~250 ms).

## Pemetaan Konten AR

* Ir. Soekarno (Class 0): Konten Ilustrasi 2D
* Ki Hajar Dewantara (Class 1): Konten Ilustrasi 3D (.glb / .gltf)
* R.A. Kartini (Class 2): Konten Video (Mp4)
* Jenderal Soedirman (Class 3): Konten Komik 2D

## Struktur Direktori

```
ar-pahlawan/
├── data/
│   ├── curated/            # Dataset citra arsip terkurasi 4 pahlawan
│   ├── multimodal/         # Variasi multimodal artefak pahlawan
│   ├── references/         # Citra referensi kanonikal
│   └── metadata.json       # Metadata biografi dan aset AR
├── dataset/                # Dataset teranotasi format YOLO
├── image/                  # Diagram alur arsitektur dan flowchart
├── model_package/          # Paket distribusi siap pakai untuk integrasi mobile/backend
│   ├── tflite/             # Model float16 untuk Android/iOS AR
│   ├── onnx/               # Model ONNX opset 12
│   ├── pytorch/            # Weights PyTorch .pt
│   ├── ar_mapping.json     # Pemetaan kelas ke format konten AR
│   ├── pipeline_spec.json  # Spesifikasi teknis pipeline
│   ├── infer_example.py    # Contoh script inferensi siap jalan
│   └── README.md           # Panduan integrasi teknis
├── src/
│   ├── app.py              # Cloud REST API (FastAPI)
│   ├── preprocessor.py     # Hough deskewing & Beta quality check
│   ├── temporal_tracker.py # Production AR Tracker (EMA + Hysteresis)
│   ├── test_media.py       # Pengujian media (kamera live, video, foto)
│   ├── two_stage_verifier.py# Pipeline verifikasi dua tahap
│   └── exporter.py         # Skrip konversi model ke TFLite dan ONNX
├── tests/                  # Suite pengujian otomatis (14 unit tests)
├── weights/                # Checkpoint model terlatih
└── requirements.txt        # Dependensi pustaka Python
```

## Persyaratan Sistem dan Instalasi

1. Buat dan aktifkan virtual environment:
   ```bash
   py -m venv venv
   .\venv\Scripts\activate
   ```

2. Pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Menjalankan Pengujian

Menjalankan seluruh suite pengujian otomatis:
```bash
python -m unittest discover tests/ -v
```

## Menjalankan REST API Server

Jalankan FastAPI service:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```
Endpoint deteksi:
* `POST /rest_api/last_request` (menerima multipart form dengan file citra)

## Menjalankan Inferensi Kamera Live

Menguji sistem menggunakan webcam:
```bash
python src/test_media.py --source webcam --conf 0.45 --imgsz 416
```
Kontrol:
* Tekan `+` atau `]` untuk menaikkan ambang confidence
* Tekan `-` atau `[` untuk menurunkan ambang confidence
* Tekan `q` untuk keluar
