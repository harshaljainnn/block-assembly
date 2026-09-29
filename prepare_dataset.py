"""
Dataset preparation script for Block Assembly Quality Inspection.
Prepares datasets for BOTH:
  1. YOLO Object Detection (yolo_dataset_det/ with images/, labels/, and data.yaml)
     Auto-annotates bounding boxes using the plastic-color segmentation engine.
  2. YOLO State Classification (yolo_dataset_cls/ with train/<state>/ and val/<state>/)

Usage:
    python prepare_dataset.py
"""

import os
import shutil
import cv2
import yaml

from block_config import (
    EXTRACTED_DIR,
    DET_DATASET_DIR,
    CLS_DATASET_DIR,
    BLOCK_CLASSES,
    ASSEMBLY_STATES,
)
from block_detector import BlockDetector


def clear_and_make(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def convert_bbox_to_yolo(bbox, img_w, img_h):
    """Converts (x, y, w, h) in pixels to normalized YOLO format (cx, cy, w, h)."""
    bx, by, bw, bh = bbox
    cx = (bx + bw / 2.0) / img_w
    cy = (by + bh / 2.0) / img_h
    nw = bw / float(img_w)
    nh = bh / float(img_h)
    return max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)), max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))


def prepare_detection_dataset(detector):
    """
    Builds the YOLO object detection dataset with auto-generated bounding box labels.
    """
    print("\n--- Generating Object Detection Dataset (yolo_dataset_det/) ---")
    clear_and_make(DET_DATASET_DIR)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DET_DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DET_DATASET_DIR, "labels", split), exist_ok=True)

    total_annotated = {"train": 0, "val": 0}

    for subfolder in ["parts", "states"]:
        sub_path = os.path.join(EXTRACTED_DIR, subfolder)
        if not os.path.exists(sub_path):
            continue

        for cat in sorted(os.listdir(sub_path)):
            cat_dir = os.path.join(sub_path, cat)
            if not os.path.isdir(cat_dir):
                continue

            for fname in sorted(os.listdir(cat_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                split = "val" if "vid_val_" in fname else "train"
                src_img_path = os.path.join(cat_dir, fname)
                img = cv2.imread(src_img_path)
                if img is None:
                    continue

                h, w = img.shape[:2]
                detections = detector.detect_color_plastic(img)

                # Copy image to images/<split>/
                unique_name = f"{subfolder}_{cat}_{fname}"
                dst_img_path = os.path.join(DET_DATASET_DIR, "images", split, unique_name)
                shutil.copy2(src_img_path, dst_img_path)

                # Write YOLO format label file
                label_name = os.path.splitext(unique_name)[0] + ".txt"
                dst_lbl_path = os.path.join(DET_DATASET_DIR, "labels", split, label_name)

                with open(dst_lbl_path, "w") as f:
                    for d in detections:
                        cls_id = d["class_id"]
                        cx, cy, nw, nh = convert_bbox_to_yolo(d["bbox"], w, h)
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                total_annotated[split] += 1

    # Write data.yaml for YOLO detection training
    abs_det_dir = os.path.abspath(DET_DATASET_DIR).replace("\\", "/")
    data_yaml = {
        "path": abs_det_dir,
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(BLOCK_CLASSES)},
    }
    yaml_path = os.path.join(DET_DATASET_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Detection dataset prepared:")
    print(f"  Train: {total_annotated['train']} images with labels")
    print(f"  Val:   {total_annotated['val']} images with labels")
    print(f"  Config: {yaml_path}")


def prepare_classification_dataset():
    """
    Builds the folder structure expected by YOLO classification:
        yolo_dataset_cls/
            train/<state_name>/*.jpg
            val/<state_name>/*.jpg
    """
    print("\n--- Generating State Classification Dataset (yolo_dataset_cls/) ---")
    clear_and_make(CLS_DATASET_DIR)

    states_dir = os.path.join(EXTRACTED_DIR, "states")
    if not os.path.exists(states_dir):
        print(f"[WARNING] '{states_dir}' not found. Run extract_frames.py first.")
        return

    counts = {}
    for state_name in sorted(os.listdir(states_dir)):
        s_dir = os.path.join(states_dir, state_name)
        if not os.path.isdir(s_dir):
            continue

        train_dst = os.path.join(CLS_DATASET_DIR, "train", state_name)
        val_dst = os.path.join(CLS_DATASET_DIR, "val", state_name)
        os.makedirs(train_dst, exist_ok=True)
        os.makedirs(val_dst, exist_ok=True)

        n_tr, n_va = 0, 0
        for fname in sorted(os.listdir(s_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            src_file = os.path.join(s_dir, fname)
            if "vid_val_" in fname:
                shutil.copy2(src_file, os.path.join(val_dst, fname))
                n_va += 1
            else:
                shutil.copy2(src_file, os.path.join(train_dst, fname))
                n_tr += 1

        counts[state_name] = (n_tr, n_va)

    print("Classification dataset prepared:")
    for st, (tr, va) in counts.items():
        print(f"  {st:<28}: {tr} train, {va} val")


def main():
    if not os.path.exists(EXTRACTED_DIR):
        print(f"[ERROR] '{EXTRACTED_DIR}' does not exist. Please run 'python extract_frames.py' first.")
        return

    detector = BlockDetector()
    prepare_detection_dataset(detector)
    prepare_classification_dataset()

    print("\n" + "=" * 65)
    print(" DATASET PREPARATION COMPLETE")
    print("=" * 65)
    print("Next steps:")
    print("  1. Train Object Detector:    python train_detector.py")
    print("  2. Train State Classifier:  python train_classifier.py")


if __name__ == "__main__":
    main()
