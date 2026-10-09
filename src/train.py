import os
import argparse
from ultralytics import YOLO

def train_detector(data_yaml="dataset/data.yaml", epochs=8, batch_size=16, imgsz=416, device="cpu", weights="weights/best.pt"):
    print(f"Initializing YOLOv8 training on {data_yaml} with Anti-Overfitting Regularization & Handheld Augmentations...")
    os.makedirs("weights", exist_ok=True)
    run_project = os.path.abspath("runs/detect")

    init_weights = weights if os.path.exists(weights) else "yolov8n.pt"
    print(f"Transfer learning checkpoint: {init_weights}")
    model = YOLO(init_weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=2,
        project=run_project,
        name="hero_train_pure",
        exist_ok=True,
        plots=True,
        save=True,
        # Anti-Overfitting hyperparameters
        weight_decay=0.0005,
        dropout=0.10,
        label_smoothing=0.03,
        cos_lr=True,
        warmup_epochs=1.0,
        mosaic=0.5,
        mixup=0.10,
        close_mosaic=3
    )

    print("Evaluating validation metrics on hold-out set...")
    metrics = model.val(data=data_yaml, split="val")

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    print(f"Validation mAP@0.50: {map50:.4f}")
    print(f"Validation mAP@0.50:0.95: {map50_95:.4f}")

    if map50 >= 0.95:
        print(f"OUTSTANDING ACCURACY mAP50 >= 0.95 ACHIEVED: {map50:.4f} (>95%).")
    elif map50 >= 0.87:
        print(f"Target mAP >= 0.87 ACHIEVED: {map50:.4f}. Ready for production integration.")
    else:
        print(f"Current mAP@0.50: {map50:.4f}.")

    best_weight_src = os.path.join(results.save_dir, "weights", "best.pt")
    best_weight_dst = os.path.join("weights", "best.pt")
    if os.path.exists(best_weight_src):
        import shutil
        shutil.copy(best_weight_src, best_weight_dst)
        print(f"Saved best model weights to {best_weight_dst}")

    return model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--weights", type=str, default="weights/best.pt")
    args = parser.parse_args()

    train_detector(epochs=args.epochs, batch_size=args.batch, imgsz=args.imgsz, device=args.device, weights=args.weights)

