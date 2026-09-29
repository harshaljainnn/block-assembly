"""
Trains a YOLO11 State Classification model on block assembly stages.

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 60 --model yolo11s-cls.pt --batch 16
"""

import os
import argparse
import torch
from ultralytics import YOLO

from block_config import CLS_DATASET_DIR


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 State Classifier for Block Assembly")
    parser.add_argument("--model", type=str, default="yolo11s-cls.pt", help="Pretrained classification model")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=448, help="Input image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    if not os.path.exists(CLS_DATASET_DIR):
        print(f"[ERROR] '{CLS_DATASET_DIR}' not found. Please run 'python prepare_dataset.py' first.")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        try:
            torch.cuda.set_per_process_memory_fraction(0.48, 0)
        except Exception:
            pass
    print(f"Training YOLO11 State Classifier on device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

    model = YOLO(args.model)

    model.train(
        data=os.path.abspath(CLS_DATASET_DIR).replace("\\", "/"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=2,
        patience=20,
        optimizer="AdamW",
        lr0=0.0005,
        project=os.path.abspath("runs_classify"),
        name="block_states",
        exist_ok=True,
    )

    print("\nTraining complete.")
    print("Best weights saved to: runs_classify/block_states/weights/best.pt")
    print("Next step: Run 'python evaluate.py' to benchmark accuracy.")


if __name__ == "__main__":
    main()
