# AGENTS.md - Project Directives and Operational Rules

This document outlines mandatory guidelines, environment constraints, and architecture standards for this project. All agents and automated workflows must strictly adhere to these rules.

## 1. Environment and Execution Standards
- Virtual Environment: Strictly use the virtual environment located at `./venv`.
- Python Command: Always invoke Python via `py` on Windows (for example: `.\venv\Scripts\py.exe` or `py -m venv venv`). Never invoke bare `python`.
- Package Management: All project dependencies must be installed strictly inside `./venv` (using `.\venv\Scripts\py -m pip install <package>`).
- Script Cleanliness: Never use decorative separator banners such as repeated equals (`====`) or repeated hyphens (`----`) that waste context tokens.
- Comments: Keep code comments strictly concise, functional, and essential. Avoid verbose explanations or obvious commentary.
- Character Constraints: Never use emoticons or emojis anywhere in code, logs, or documentation.

## 2. Domain and Architecture Alignment
- Project Target: AI-Driven Mobile AR for Indonesian National Heroes detection and content serving.
- Target Metric: Evaluation must aim for mAP >= 0.87 in compliance with project flowcharts (`image/diagram alir.png`).
- Core Detector: YOLOv8 (ultralytics) as specified in `image/rancangan ai-driven vr mengadopsi v8.png`.
- Preprocessing: Must implement Hough Transform deskewing and Artefact Assessment beta (sharpness/contrast check) before inference (`image/deep learning.png`).
- AR Content Mapping:
  1. Ir. Soekarno -> Konten Ilustrasi 2D
  2. Ki Hajar Dewantara -> Konten Ilustrasi 3D (.glb / .gltf)
  3. R.A. Kartini -> Konten Video (Mp4)
  4. Jenderal Soedirman -> Konten Komik 2D
- Mobile Integration: Support both cloud REST API (`POST /rest_api/last_request`) and edge export (.tflite and .onnx).
