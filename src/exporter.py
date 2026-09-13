import os
import shutil
import argparse
import numpy as np
from ultralytics import YOLO

def export_models(weights_path="weights/best.pt", imgsz=416):
    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}. Using base yolov8n.pt for export verification.")
        weights_path = "yolov8n.pt"

    print(f"Loading YOLO model from {weights_path}...")
    model = YOLO(weights_path)

    os.makedirs("weights/mobile", exist_ok=True)

    print("1. Exporting to ONNX format...")
    onnx_file = model.export(format="onnx", imgsz=imgsz, dynamic=False, opset=12)
    mobile_onnx = "weights/mobile/hero_detector.onnx"
    shutil.copy(onnx_file, mobile_onnx)
    print(f"ONNX exported and saved to {mobile_onnx}")

    print("2. Converting ONNX to TFLite via onnx2tf...")
    try:
        import onnx2tf
        import onnx2tf.onnx2tf
        import onnx2tf.utils.common_functions

        dummy_data_fn = lambda: np.ones((1, 3, imgsz, imgsz), dtype=np.float32)
        onnx2tf.onnx2tf.download_test_image_data = dummy_data_fn
        onnx2tf.utils.common_functions.download_test_image_data = dummy_data_fn

        onnx2tf.convert(
            input_onnx_file_path=mobile_onnx,
            output_folder_path="weights/mobile",
            not_use_onnxsim=True,
            output_integer_quantized_tflite=False
        )

        f16_src = "weights/mobile/hero_detector_float16.tflite"
        if not os.path.exists(f16_src):
            f16_src = "weights/mobile/best_float16.tflite"

        final_tflite = "weights/mobile/hero_detector.tflite"
        if os.path.exists(f16_src):
            shutil.copy(f16_src, final_tflite)
            print(f"TFLite exported successfully: {final_tflite} (Size: {os.path.getsize(final_tflite)} bytes)")
    except Exception as e:
        print(f"TFLite export notification: {e}")

    labels = ["soekarno", "dewantara", "kartini", "soedirman"]
    labels_file = "weights/mobile/labels.txt"
    with open(labels_file, "w") as f:
        for lbl in labels:
            f.write(f"{lbl}\n")
    print(f"Saved mobile labels to {labels_file}")

def export_stage2_classifier(weights_path="weights/stage2_classifier.pt"):
    if not os.path.exists(weights_path):
        print(f"Stage 2 weights not found at {weights_path}.")
        return

    import torch
    import torch.nn as nn
    import torchvision.models as models

    print(f"Loading Stage 2 MobileNetV3 classifier from {weights_path}...")
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 5)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    os.makedirs("weights/mobile", exist_ok=True)
    onnx_path = "weights/mobile/hero_verifier.onnx"
    dummy = torch.randn(1, 3, 128, 128)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12
    )
    print(f"Stage 2 ONNX exported to {onnx_path}")

    try:
        import onnx2tf
        import onnx2tf.onnx2tf
        import onnx2tf.utils.common_functions

        dummy_data_fn = lambda: np.ones((1, 3, 128, 128), dtype=np.float32)
        onnx2tf.onnx2tf.download_test_image_data = dummy_data_fn
        onnx2tf.utils.common_functions.download_test_image_data = dummy_data_fn

        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path="weights/mobile/verifier_tflite",
            not_use_onnxsim=True,
            output_integer_quantized_tflite=False
        )
        for cand in [
            "weights/mobile/verifier_tflite/hero_verifier_float16.tflite",
            "weights/mobile/verifier_tflite/hero_verifier_float32.tflite"
        ]:
            if os.path.exists(cand):
                shutil.copy(cand, "weights/mobile/hero_verifier.tflite")
                print(f"Stage 2 TFLite exported successfully: weights/mobile/hero_verifier.tflite")
                break
    except Exception as e:
        print(f"Stage 2 TFLite conversion notification: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--export_stage2", action="store_true", default=False)
    args = parser.parse_args()

    export_models(weights_path=args.weights, imgsz=args.imgsz)
    if args.export_stage2 or os.path.exists("weights/stage2_classifier.pt"):
        export_stage2_classifier()


