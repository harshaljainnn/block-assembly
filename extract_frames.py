"""
Frame extraction script for Block Assembly dataset videos.
Extracts frames from all 12 dataset videos in DATASET/ with Laplacian blur filtering,
plastic-color visibility validation, and temporal 80/20 train/val partitioning.

Usage:
    python extract_frames.py                  # Auto-extracts from all 12 dataset videos
    python extract_frames.py --fps 4.0        # Custom extraction frame rate (default: 4.0)
    python extract_frames.py --blur_thresh 25 # Set custom blur threshold
"""

import os
import argparse
import cv2
import numpy as np

from block_config import (
    VIDEOS_DIR,
    EXTRACTED_DIR,
    VIDEO_MAPPING,
    BLOCK_CLASSES,
    ASSEMBLY_STATES,
)


def is_blurry(gray_frame, threshold=20.0):
    """Calculates Laplacian variance. Low variance indicates motion blur."""
    if threshold <= 0:
        return False
    var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return var < threshold


def has_block_content(frame, min_area=350):
    """
    Checks if at least one colored plastic block is visible.
    Eliminates frames where hands completely occlude the object or the block is out of frame.
    """
    b, g, r = cv2.split(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    blue = (b > 70) & (b > 1.15 * r.astype(np.float32)) & (h >= 90) & (h <= 135) & (s > 50)
    green = (g > 65) & (g > 1.15 * r.astype(np.float32)) & (h >= 35) & (h <= 85) & (s > 50)
    red = (r > 95) & (r > 1.50 * g.astype(np.float32)) & ((h <= 14) | (h >= 166)) & (s > 100)
    yellow = (r > 95) & (g > 75) & (r > 1.35 * b.astype(np.float32)) & (h >= 15) & (h <= 35) & (s > 85)

    mask = (blue | green | red | yellow).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) > min_area for c in cnts)


def clean_existing_frames(target_dir):
    """Removes previously extracted vid_* frames from target directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return 0

    removed = 0
    for f in os.listdir(target_dir):
        if f.startswith("vid_") and f.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(target_dir, f))
            removed += 1
    return removed


def extract_from_video(
    video_path,
    target_category,
    is_part_video=False,
    fps=4.0,
    blur_thresh=20.0,
    val_split=0.20,
    check_visibility=True,
    clean_old=True,
):
    """
    Extracts frames from a single video, filtering blur and applying temporal train/val split.
    """
    if not os.path.exists(video_path):
        print(f"  [ERROR] Video file not found: {video_path}")
        return None

    subfolder = "parts" if is_part_video else "states"
    out_dir = os.path.join(EXTRACTED_DIR, subfolder, target_category)
    os.makedirs(out_dir, exist_ok=True)

    if clean_old:
        num_cleaned = clean_existing_frames(out_dir)
        if num_cleaned > 0:
            print(f"  [INFO] Cleaned {num_cleaned} old video frames from '{out_dir}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Could not open video: {video_path}")
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if total_frames > 0 else 0

    step = max(1, int(round(video_fps / fps)))
    split_frame = int(total_frames * (1.0 - val_split))

    print(f"\nProcessing '{os.path.basename(video_path)}' -> {subfolder}/{target_category}")
    print(f"  Video: {total_frames} frames ({duration_sec:.1f}s) @ {video_fps:.1f} fps")
    print(f"  Sampling rate: every {step} frames (~{video_fps / step:.1f} fps)")
    print(f"  Temporal Split: frames 0..{split_frame-1} -> TRAIN, {split_frame}..{total_frames} -> VAL")

    frame_idx = 0
    saved_train = 0
    saved_val = 0
    skipped_blur = 0
    skipped_visibility = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Blur Check
            if is_blurry(gray, blur_thresh):
                skipped_blur += 1
                frame_idx += 1
                continue

            # 2. Block Visibility Check
            if check_visibility and not has_block_content(frame):
                skipped_visibility += 1
                frame_idx += 1
                continue

            # 3. Temporal train/val assignment
            split_tag = "train" if frame_idx < split_frame else "val"
            fname = f"vid_{split_tag}_{frame_idx:06d}.jpg"
            save_path = os.path.join(out_dir, fname)
            cv2.imwrite(save_path, frame)

            if split_tag == "train":
                saved_train += 1
            else:
                saved_val += 1

        frame_idx += 1

    cap.release()

    print(f"  Results: {saved_train} train frames, {saved_val} val frames saved.")
    print(f"  Filtered out: {skipped_blur} blurry, {skipped_visibility} without visible blocks.")
    return {
        "target": target_category,
        "type": subfolder,
        "train": saved_train,
        "val": saved_val,
        "blurry": skipped_blur,
        "no_block": skipped_visibility,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract video frames for Block Assembly inspection.")
    parser.add_argument("--videos_dir", type=str, default=VIDEOS_DIR, help="Folder containing dataset videos")
    parser.add_argument("--fps", type=float, default=4.0, help="Target extraction FPS (default: 4.0)")
    parser.add_argument(
        "--blur_thresh",
        type=float,
        default=20.0,
        help="Laplacian variance blur threshold (default: 20.0)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.20,
        help="Fraction of video duration reserved for validation (default: 0.20)",
    )
    parser.add_argument(
        "--no_clean",
        action="store_true",
        help="Do not delete old frames before extracting",
    )

    args = parser.parse_args()

    if not os.path.exists(args.videos_dir):
        print(f"[ERROR] Videos directory '{args.videos_dir}' does not exist.")
        return

    available_videos = [f for f in os.listdir(args.videos_dir) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    print(f"Found {len(available_videos)} video files in '{args.videos_dir}'. Starting extraction...\n")

    results = []
    for vname in sorted(available_videos):
        vpath = os.path.join(args.videos_dir, vname)
        info = VIDEO_MAPPING.get(vname)
        if not info:
            print(f"[WARNING] Skipping unmapped video: {vname}")
            continue

        is_part = info["type"] == "part"
        target_name = info["target"]

        res = extract_from_video(
            video_path=vpath,
            target_category=target_name,
            is_part_video=is_part,
            fps=args.fps,
            blur_thresh=args.blur_thresh,
            val_split=args.val_split,
            clean_old=not args.no_clean,
        )
        if res:
            results.append(res)

    print("\n" + "=" * 65)
    print(" EXTRACTION SUMMARY")
    print("=" * 65)
    total_tr, total_va = 0, 0
    for r in results:
        print(f"  [{r['type'].upper()}] {r['target']:<24}: {r['train']} train, {r['val']} val (filtered {r['blurry']} blur, {r['no_block']} no-block)")
        total_tr += r["train"]
        total_va += r["val"]
    print("-" * 65)
    print(f"  Total: {total_tr} train frames, {total_va} val frames extracted.")
    print("=" * 65)
    print("\nNext step: Run 'python prepare_dataset.py' to generate detection & classification datasets.")


if __name__ == "__main__":
    main()
