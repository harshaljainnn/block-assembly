"""
Generates the cropped state classification dataset for YOLO:
    yolo_dataset_cropped_cls/
        train/<state_name>/*.jpg
        val/<state_name>/*.jpg

Extracts tightly anchored, normalized assembly crops centered on the green beam
to eliminate desk clutter, hand noise, and background lighting variance.

Usage:
    python prepare_cropped_dataset.py
"""

import os
import shutil
import cv2
from block_config import EXTRACTED_DIR, PROJECT_ROOT
from block_cropper import crop_assembly

CROPPED_DATASET_DIR = os.path.join(PROJECT_ROOT, "yolo_dataset_cropped_cls")

STATE_MAP = {
    "state1": "state_1_greenblue",
    "state2": "state_2_green2blue",
    "state3": "state_3_first_red",
    "state4": "state_4_yellowred",
    "state5": "state_5_bothred",
    "state6": "state_6_yellowafter2red",
    "state7": "state_7_finalred",
    "state8": "state_8_complete",
}


def clear_and_make(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main():
    states_dir = os.path.join(EXTRACTED_DIR, "states")
    if not os.path.exists(states_dir):
        print(f"[ERROR] '{states_dir}' not found. Run extract_frames.py first.")
        return

    print("=" * 70)
    print(" GENERATING CROPPED STATE CLASSIFICATION DATASET")
    print(f" Target Directory: {CROPPED_DATASET_DIR}")
    print("=" * 70)

    clear_and_make(CROPPED_DATASET_DIR)
    for split in ["train", "val"]:
        for sname in STATE_MAP.values():
            os.makedirs(os.path.join(CROPPED_DATASET_DIR, split, sname), exist_ok=True)

    summary = {}

    for folder_name in sorted(os.listdir(states_dir)):
        src_dir = os.path.join(states_dir, folder_name)
        if not os.path.isdir(src_dir):
            continue

        canonical_name = STATE_MAP.get(folder_name, folder_name)
        train_dst = os.path.join(CROPPED_DATASET_DIR, "train", canonical_name)
        val_dst = os.path.join(CROPPED_DATASET_DIR, "val", canonical_name)

        n_train = 0
        n_val = 0
        n_skipped = 0

        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            split = "val" if "vid_val_" in fname else "train"
            dst_dir = val_dst if split == "val" else train_dst

            img_path = os.path.join(src_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            crop, bbox = crop_assembly(img, pad_ratio=0.22, min_size=180, ignore_hud=False)
            if crop is not None and crop.shape[0] >= 50 and crop.shape[1] >= 50:
                dst_path = os.path.join(dst_dir, fname)
                cv2.imwrite(dst_path, crop)
                if split == "train":
                    n_train += 1
                else:
                    n_val += 1
            else:
                n_skipped += 1

        summary[canonical_name] = (n_train, n_val, n_skipped)
        print(f"  {canonical_name:<26}: {n_train:>3} train crops, {n_val:>3} val crops (skipped {n_skipped})")

    total_train = sum(s[0] for s in summary.values())
    total_val = sum(s[1] for s in summary.values())
    print("-" * 70)
    print(f"  Total Processed: {total_train} train crops, {total_val} validation crops.")
    print("=" * 70)
    print("\nDataset ready! Next step: Run 'python train_classifier.py --dataset yolo_dataset_cropped_cls'")


if __name__ == "__main__":
    main()
