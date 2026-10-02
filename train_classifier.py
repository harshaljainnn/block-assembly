"""
Trains a YOLO State Classification model on cropped block assembly stages.
Optimized for high-accuracy industrial inspection:
  - Uses normalized, focused assembly crops (384x384)
  - Color & illumination augmentations (robust to room lighting & webcam shifts)
  - Slight perspective and rotation invariance (tolerance to camera angles)

Usage:
    python train_classifier.py                               # Default: 60 epochs on yolo_dataset_cropped_cls
    python train_classifier.py --epochs 60 --model yolo11s-cls.pt --batch 16
    python train_classifier.py --resume                      # Resume interrupted training
"""

import os
import argparse
import torch
from ultralytics import YOLO

from block_config import PROJECT_ROOT

DEFAULT_CROPPED_DATASET = os.path.join(PROJECT_ROOT, "yolo_dataset_cropped_cls")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO State Classifier on Cropped Assemblies")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_CROPPED_DATASET,
        help="Path to classification dataset directory (default: yolo_dataset_cropped_cls)",
    )
    parser.add_argument("--model", type=str, default="yolo11s-cls.pt", help="Pretrained classification model (e.g. yolo11s-cls.pt or yolov8s-cls.pt)")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--imgsz", type=int, default=384, help="Input image resolution (default: 384)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--name", type=str, default="block_states_cropped", help="Run folder name under runs_classify/")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt checkpoint")
    args = parser.parse_args()

    last_pt = os.path.join(PROJECT_ROOT, "runs_classify", args.name, "weights", "last.pt")
    if args.resume:
        if not os.path.exists(last_pt):
            print(f"[ERROR] Cannot resume: checkpoint '{last_pt}' does not exist.")
            return
        print(f"Resuming training from checkpoint: {last_pt}")
        model = YOLO(last_pt)
        model.train(resume=True)
        return

    dataset_path = os.path.abspath(args.dataset).replace("\\", "/")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset directory '{dataset_path}' not found.")
        print("Please run 'python prepare_cropped_dataset.py' first.")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print("=" * 70)
    print(" YOLO STATE CLASSIFIER TRAINING")
    print(f" Dataset:     {dataset_path}")
    print(f" Base Model:  {args.model}")
    print(f" Device:      {'GPU: ' + device_name if device == 0 else 'CPU (No GPU found)'}")
    print(f" Image Size:  {args.imgsz}x{args.imgsz}")
    print(f" Epochs:      {args.epochs} | Batch: {args.batch}")
    print("=" * 70)

    if device == "cpu":
        print("[NOTE] Running on CPU. Training will take longer.")
        print("To train on a GPU laptop, clone/pull this branch and run:")
        print(f"   python train_classifier.py --epochs {args.epochs} --batch {args.batch}")
        print("=" * 70)

    model = YOLO(args.model)

    model.train(
        data=dataset_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=2,
        patience=20,
        optimizer="AdamW",
        lr0=0.0005,
        # Real-world webcam & lighting augmentations:
        hsv_h=0.015,     # subtle hue jitter
        hsv_s=0.40,      # saturation shifts for variable lighting
        hsv_v=0.40,      # brightness shifts for room light changes
        degrees=15.0,    # angle rotation tolerance
        translate=0.08,  # position shifts
        scale=0.15,      # distance / zoom tolerance
        fliplr=0.5,      # horizontal mirror
        flipud=0.0,
        project=os.path.abspath("runs_classify"),
        name=args.name,
        exist_ok=True,
    )

    best_pt = os.path.join(PROJECT_ROOT, "runs_classify", args.name, "weights", "best.pt")
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print(f" Best weights saved to: {best_pt}")
    print("=" * 70)
    print("Next step: Run 'python live_demo.py' to inspect live with high accuracy.")


if __name__ == "__main__":
    main()
