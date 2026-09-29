"""
Unified CLI Entry Point for Block Assembly Quality Inspection System.
Provides an interactive menu and direct command-line flags.

Usage:
    python main.py             # Interactive menu
    python main.py --extract   # Extract frames from dataset videos
    python main.py --prepare   # Prepare detection & classification datasets
    python main.py --train-det # Train YOLO Object Detector
    python main.py --train-cls # Train YOLO State Classifier
    python main.py --eval      # Run benchmark evaluation
    python main.py --demo      # Launch live inspection HUD
"""

import sys
import argparse
import subprocess


def run_cmd(script_name, extra_args=None):
    cmd = [sys.executable, script_name]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n>>> Running: {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def interactive_menu():
    while True:
        print("\n" + "=" * 55)
        print(" BLOCK ASSEMBLY QUALITY INSPECTION SYSTEM")
        print("=" * 55)
        print("  1. Extract Frames from Dataset Videos")
        print("  2. Prepare Datasets (Detection + Classification)")
        print("  3. Train YOLO Object Detector (Block Localizer)")
        print("  4. Train YOLO State Classifier (Assembly Stages)")
        print("  5. Run Benchmark Evaluation on Validation Set")
        print("  6. Launch Live Inspection HUD (Camera / Video)")
        print("  0. Exit")
        print("=" * 55)

        choice = input("Select an option [0-6]: ").strip()

        if choice == "1":
            run_cmd("extract_frames.py")
        elif choice == "2":
            run_cmd("prepare_dataset.py")
        elif choice == "3":
            run_cmd("train_detector.py")
        elif choice == "4":
            run_cmd("train_classifier.py")
        elif choice == "5":
            run_cmd("evaluate.py")
        elif choice == "6":
            src = input("Enter video path, image path, or camera index (press Enter for default webcam '0'): ").strip()
            if not src or src == "0":
                run_cmd("live_demo.py")
            elif src.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                run_cmd("live_demo.py", ["--video", src])
            elif src.lower().endswith((".jpg", ".jpeg", ".png")):
                run_cmd("live_demo.py", ["--image", src])
            else:
                run_cmd("live_demo.py", ["--camera", src])
        elif choice == "0":
            print("Exiting.")
            break
        else:
            print("Invalid option. Please choose between 0 and 6.")


def main():
    parser = argparse.ArgumentParser(description="Block Assembly Inspection CLI")
    parser.add_argument("--extract", action="store_true", help="Extract video frames")
    parser.add_argument("--prepare", action="store_true", help="Prepare datasets")
    parser.add_argument("--train-det", action="store_true", help="Train YOLO object detector")
    parser.add_argument("--resume", action="store_true", help="Resume YOLO object detector from last checkpoint")
    parser.add_argument("--train-cls", action="store_true", help="Train YOLO state classifier")
    parser.add_argument("--eval", action="store_true", help="Run benchmark evaluation")
    parser.add_argument("--demo", action="store_true", help="Launch live inspection HUD")
    parser.add_argument("--video", type=str, default=None, help="Video path for demo")
    parser.add_argument("--image", type=str, default=None, help="Image path for demo")

    args = parser.parse_args()

    if args.extract:
        run_cmd("extract_frames.py")
    elif args.prepare:
        run_cmd("prepare_dataset.py")
    elif args.train_det:
        det_args = ["--resume"] if args.resume else []
        run_cmd("train_detector.py", det_args)
    elif args.resume:
        run_cmd("train_detector.py", ["--resume"])
    elif args.train_cls:
        run_cmd("train_classifier.py")
    elif args.eval:
        run_cmd("evaluate.py")
    elif args.demo:
        demo_args = []
        if args.video:
            demo_args.extend(["--video", args.video])
        elif args.image:
            demo_args.extend(["--image", args.image])
        run_cmd("live_demo.py", demo_args)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
