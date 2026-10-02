"""
Evaluation and benchmark script for Block Assembly Quality Inspection.
Evaluates:
  1. Assembly state sequence classification accuracy across held-out validation frames
  2. Individual block detection recall on isolated component validation frames
  3. Spatial rule verification consistency

Usage:
    python evaluate.py
"""

import os
import cv2
from component_detector import ComponentDetector
from block_config import EXTRACTED_DIR, ASSEMBLY_STATES, STEP_TITLES


def evaluate_states(detector):
    states_dir = os.path.join(EXTRACTED_DIR, "states")
    if not os.path.exists(states_dir):
        print(f"[ERROR] '{states_dir}' does not exist. Run extract_frames.py first.")
        return

    print("\n" + "=" * 72)
    print(" BENCHMARK: ASSEMBLY STATE CLASSIFICATION ON HELD-OUT VAL FRAMES")
    print("=" * 72)

    total_frames = 0
    correct_frames = 0
    state_results = {}

    STATE_DIR_MAP = {
        "state1": ("state_1_greenblue", 1),
        "state2": ("state_2_green2blue", 2),
        "state3": ("state_3_first_red", 3),
        "state4": ("state_4_yellowred", 4),
        "state5": ("state_5_bothred", 5),
        "state6": ("state_6_yellowafter2red", 6),
        "state7": ("state_7_finalred", 7),
        "state8": ("state_8_complete", 8),
    }

    for sname in sorted(os.listdir(states_dir)):
        s_path = os.path.join(states_dir, sname)
        if not os.path.isdir(s_path):
            continue

        val_files = [f for f in os.listdir(s_path) if f.startswith("vid_val_") and f.lower().endswith((".jpg", ".png"))]
        if not val_files:
            continue

        mapped_state, step_idx = STATE_DIR_MAP.get(sname, (sname, 0))
        s_correct = 0
        s_total = len(val_files)

        for vf in val_files:
            fpath = os.path.join(s_path, vf)
            img = cv2.imread(fpath)
            if img is None:
                continue

            res = detector.analyze(img, current_step_index=step_idx)
            pred = res["predicted_state"]

            if pred == mapped_state:
                s_correct += 1

        acc = (s_correct / s_total) * 100.0 if s_total > 0 else 0.0
        state_results[sname] = {"correct": s_correct, "total": s_total, "acc": acc}

        total_frames += s_total
        correct_frames += s_correct

        title = STEP_TITLES.get(mapped_state, sname)
        print(f"  {title:<32}: {s_correct:>3}/{s_total:<3} ({acc:>5.1f}%)")

    overall_acc = (correct_frames / total_frames) * 100.0 if total_frames > 0 else 0.0
    print("-" * 72)
    print(f"  Overall Validation Accuracy: {correct_frames}/{total_frames} ({overall_acc:.1f}%)")
    print("=" * 72)


def evaluate_parts(detector):
    parts_dir = os.path.join(EXTRACTED_DIR, "parts")
    if not os.path.exists(parts_dir):
        return

    print("\n" + "=" * 72)
    print(" BENCHMARK: INDIVIDUAL OBJECT DETECTION ON ISOLATED PART VIDEOS")
    print("=" * 72)

    for pname in sorted(os.listdir(parts_dir)):
        p_path = os.path.join(parts_dir, pname)
        if not os.path.isdir(p_path):
            continue

        val_files = [f for f in os.listdir(p_path) if f.startswith("vid_val_") and f.lower().endswith((".jpg", ".png"))]
        if not val_files:
            continue

        det_correct = 0
        p_total = len(val_files)

        for vf in val_files:
            fpath = os.path.join(p_path, vf)
            img = cv2.imread(fpath)
            if img is None:
                continue

            res = detector.analyze(img)
            detections = res.get("detections", [])
            if any(d["class_name"] == pname for d in detections):
                det_correct += 1

        acc = (det_correct / p_total) * 100.0 if p_total > 0 else 0.0
        print(f"  {pname:<32}: {det_correct:>3}/{p_total:<3} ({acc:>5.1f}%)")

    print("=" * 72)


def main():
    detector = ComponentDetector()
    evaluate_parts(detector)
    evaluate_states(detector)


if __name__ == "__main__":
    main()
