"""
Live Visual Assembly Checker for Block Assembly Quality Inspection.
Combines:
  - Real-time Object Detection (bounding boxes & incoming block tracking)
  - Assembly Graph Spatial Verification (relative positions & connections)
  - Dual-Perception Corroboration & Incoming Part Validator
  - State Machine Strict Sequential Enforcement & Temporal Smoothing HUD

Usage:
    python live_demo.py                             # Default built-in webcam (index 0)
    python live_demo.py --camera 1                  # External USB / DroidCam PC client
    python live_demo.py --camera http://<IP>:4747/video # DroidCam WiFi IP stream
    python live_demo.py --video <path_to_video.mp4> # Test on recorded video
    python live_demo.py --image <path_to_image.jpg> # Inspect single image
"""

import os
import argparse
import cv2
import numpy as np

from component_detector import ComponentDetector
from state_machine import AssemblyStateMachine
from block_config import BLOCK_COLORS_BGR, STEP_TITLES


def draw_hud(frame, result, state_machine, smoothed=True):
    """
    Renders an industrial quality inspection HUD onto the camera frame.
    Displays:
      - Bounding boxes around all detected blocks
      - Incoming object callout (highlighting part in hand)
      - Top status banner (PASS / ADVANCED / HOLDING / ERROR)
      - Right-hand sequential assembly checklist (all 9 steps)
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    state = result.get("predicted_state", "state_0_unstarted")
    conf = result.get("confidence", 0.0)
    is_valid = result.get("is_valid", True)
    diagnostic = result.get("diagnostic", "")

    if smoothed:
        status, detail, consensus_state, votes_ratio = state_machine.update_smoothed(
            state, is_valid_spatial=is_valid, diagnostic=diagnostic
        )
        display_state = consensus_state if consensus_state else state
    else:
        status, detail = state_machine.update(state)
        votes_ratio = "1/1"
        display_state = state

    is_error = status == "error" or state_machine.error_active
    is_done = state_machine.is_complete()

    # 1. Draw Bounding Boxes around Detected Blocks
    detections = result.get("detections", [])
    for d in detections:
        bx, by, bw, bh = d["bbox"]
        cname = d["class_name"]
        dconf = d["confidence"]
        box_col = BLOCK_COLORS_BGR.get(cname, (200, 200, 200))

        # Box
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_col, 2)

        # Label badge with high-contrast text (black for yellow, white for others)
        lbl = f"{cname.replace('_block', '')} {dconf*100:.0f}%"
        text_col = (0, 0, 0) if cname == "yellow_block" else (255, 255, 255)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_y1 = max(0, by - 18)
        badge_y2 = max(18, by)
        cv2.rectangle(frame, (bx, badge_y1), (bx + tw + 6, badge_y2), box_col, -1)
        cv2.putText(frame, lbl, (bx + 3, badge_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1)

    # 2. Highlight Incoming Object (Part in Hand)
    incoming = result.get("incoming_object")
    if incoming:
        ix, iy, iw, ih = incoming["bbox"]
        is_exp = incoming["is_expected"]
        badge_col = (0, 220, 0) if is_exp else (0, 0, 255)

        # Pulsing / Thick border around incoming part
        cv2.rectangle(frame, (ix - 3, iy - 3), (ix + iw + 3, iy + ih + 3), badge_col, 3)
        tag = "[INCOMING: VALID]" if is_exp else "[INCOMING: WRONG PART!]"
        cv2.putText(frame, tag, (ix, max(25, iy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, badge_col, 2)

    # 3. Top Status Banner
    header_h = 160
    if is_error:
        header_color = (20, 20, 160)   # Crimson Red
        border_color = (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (w, h), border_color, 4)
    elif is_done:
        header_color = (20, 120, 20)   # Forest Green
        border_color = (0, 220, 0)
        cv2.rectangle(frame, (0, 0), (w, h), border_color, 4)
    elif status == "advanced":
        header_color = (15, 60, 95)    # Dark Navy / Blue-Amber Accent
    else:
        header_color = (22, 22, 22)    # Industrial Dark Slate

    # Apply semi-transparent header bar directly to ROI
    sub_header = frame[0:header_h, 0:w]
    header_bg = np.full(sub_header.shape, header_color, dtype=np.uint8)
    frame[0:header_h, 0:w] = cv2.addWeighted(sub_header, 0.12, header_bg, 0.88, 0)

    panel_w = 240 if w >= 640 else 190
    start_x = max(w - panel_w, int(w * 0.58))

    scale = 0.65 if w >= 640 else 0.50
    sub_scale = 0.50 if w >= 640 else 0.40

    # Right side checklist background container
    sub_chk = frame[4:header_h - 4, max(0, start_x - 6):w - 4]
    chk_bg = np.full(sub_chk.shape, (10, 10, 10), dtype=np.uint8)
    frame[4:header_h - 4, max(0, start_x - 6):w - 4] = cv2.addWeighted(sub_chk, 0.20, chk_bg, 0.80, 0)

    # Left Side Info
    if is_error:
        cv2.putText(frame, "SEQUENCE REJECTED!", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        err_msg = detail or state_machine.error_detail or diagnostic or "Sequence defect detected"
        if len(err_msg) > 42 and w < 700:
            err_msg = err_msg[:39] + "..."
        cv2.putText(frame, err_msg, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (200, 230, 255), 1)
        cv2.putText(frame, f"Detected: {display_state} ({conf*100:.0f}%)", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (255, 255, 255), 1)
        if incoming:
            cv2.putText(frame, incoming["message"][:45], (12, 118), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (0, 255, 255), 1)
        cv2.putText(frame, "Correct missing part or press 'r' to reset", (12, 145), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (0, 255, 255), 1)
    elif is_done:
        cv2.putText(frame, "100% COMPLETE & VERIFIED!", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        cv2.putText(frame, "All 9 assembly stages verified in order.", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 255, 220), 1)
        cv2.putText(frame, "Figure structure complete!", (12, 102), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 255, 220), 1)
        cv2.putText(frame, "READY FOR NEXT UNIT - Press 'r' to reset", (12, 138), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (0, 255, 255), 1)
    else:
        cur_title = state_machine.get_step_title(state_machine.current_index)
        cv2.putText(frame, f"INSPECTION: {cur_title.upper()}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        cv2.putText(frame, f"LIVE: {display_state} ({conf*100:.0f}%)", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 220, 220), 1)
        stat_color = (0, 220, 255) if status == "advanced" else (220, 180, 50)
        cv2.putText(frame, f"STATUS: {status.upper()} (Dwell: {votes_ratio})", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, stat_color, 1)
        next_step = state_machine.get_step_title(state_machine.current_index + 1)
        cv2.putText(frame, f"Next: [{next_step}]", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (180, 180, 180), 1)
        if incoming:
            inc_col = (0, 255, 0) if incoming["is_expected"] else (50, 50, 255)
            cv2.putText(frame, incoming["message"][:48], (12, 145), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, inc_col, 1)

    steps = state_machine.get_steps_for_hud()
    for i, s in enumerate(steps):
        st = s["status"]
        if st == "verified":
            tag = "[PASS] "
            col = (60, 235, 60)   # Green
        elif st == "skipped":
            tag = "[MISS] "
            col = (40, 40, 255)   # Red
        elif st in ["current", "error_current"]:
            tag = "[NOW ] "
            col = (50, 180, 255) if st == "error_current" else (240, 210, 40)
        else:
            tag = "[    ] "
            col = (140, 140, 140) # Dim Gray

        title = s["title"].split(". ", 1)[-1]
        text = f"{tag}{i}.{title[:14]}"
        cv2.putText(frame, text, (start_x, 18 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

    return frame


def run_image(image_path, detector, sm):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return
    img = cv2.imread(image_path)
    res = detector.analyze(img, current_step_index=sm.current_index)
    annotated = draw_hud(img, res, sm, smoothed=False)

    out_path = "demo_output.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\nAssembly State: {res['predicted_state']} (Confidence: {res['confidence']:.2f})")
    print(f"Detected Blocks: {len(res['detections'])}")
    print(f"Diagnostic: {res['diagnostic']}")
    if res.get("incoming_object"):
        print(f"Incoming: {res['incoming_object']['message']}")
    print(f"Annotated result saved to '{out_path}'")


def run_video(video_source, detector, sm):
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_source}")
        print("Tip: If using DroidCam client, try device index 1 or 2:")
        print("     python live_demo.py --camera 1")
        print("     Or use the direct WiFi IP URL from the DroidCam app:")
        print("     python live_demo.py --camera http://<PHONE_IP>:4747/video")
        return

    print(f"\nLive Inspection started on [{video_source}].")
    print("Controls: 'q' = Quit, 'r' = Reset sequence.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = detector.analyze(frame, current_step_index=sm.current_index)
        annotated = draw_hud(frame, res, sm, smoothed=True)

        cv2.imshow("Block Assembly Checker (Live HUD)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            sm.reset()
            print("[INFO] Sequence tracker reset to Step 0.")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live Block Assembly Inspection HUD")
    parser.add_argument("--image", type=str, default=None, help="Inspect a single image file")
    parser.add_argument("--video", type=str, default=None, help="Run inspection on a recorded video file")
    parser.add_argument("--camera", type=str, default="0", help="Webcam index (0, 1) or IP URL")
    args = parser.parse_args()

    detector = ComponentDetector()
    sm = AssemblyStateMachine()

    if args.image:
        run_image(args.image, detector, sm)
    elif args.video:
        run_video(args.video, detector, sm)
    else:
        run_video(args.camera, detector, sm)


if __name__ == "__main__":
    main()
