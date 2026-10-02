"""
Live Visual Assembly Checker for Block Assembly Quality Inspection.
Combines:
  - Real-time Object Detection (bounding boxes & incoming block tracking)
  - Focused Assembly Cropping (Green-beam anchor with Picture-in-Picture display)
  - State Machine Strict Sequential Enforcement & Temporal Consensus Smoothing
  - Hand-Stillness Protection (tolerant to in-flight hand placement)
  - Optional non-blocking FastAPI Dashboard Bridge for live performance tracking

Usage:
    python live_demo.py                                     # Auto-detects working webcam
    python live_demo.py --camera 1                          # External USB / DroidCam PC client
    python live_demo.py --camera http://<IP>:4747/video     # DroidCam WiFi IP stream
    python live_demo.py --video <path_to_video.mp4>         # Test on recorded video
    python live_demo.py --image <path_to_image.jpg>         # Inspect single image
    python live_demo.py --dashboard http://localhost:8000   # Stream events to FastAPI dashboard
"""

import os
import time
import argparse
import threading
import cv2
import numpy as np

from component_detector import ComponentDetector
from state_machine import AssemblyStateMachine
from block_config import BLOCK_COLORS_BGR, STEP_TITLES


class DashboardBridge:
    """
    Non-blocking background bridge that posts real-time assembly events
    to the FastAPI performance dashboard backend (backend/main.py).
    """

    def __init__(self, api_url=None, operator_id="OP001", api_key=""):
        if api_url:
            self.api_url = str(api_url).strip().strip(")'\"`").rstrip("/")
        else:
            self.api_url = None
        self.operator_id = operator_id
        self.api_key = api_key
        self.cycle_id = None
        self.enabled = bool(self.api_url)

        self._latest_jpeg = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        if self.enabled:
            self._start_frame_streamer()

    def update_frame(self, frame):
        """
        Compresses the annotated HUD frame to JPEG and hands it off to the background streamer.
        """
        if not self.enabled:
            return
        try:
            h, w = frame.shape[:2]
            if w > 960:
                scale = 960.0 / w
                frame = cv2.resize(frame, (960, int(h * scale)), interpolation=cv2.INTER_AREA)

            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ret:
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()
        except Exception:
            pass

    def _start_frame_streamer(self):
        def worker():
            import requests
            session = requests.Session()
            headers = {"Content-Type": "image/jpeg"}
            if self.api_key:
                headers["x-cv-api-key"] = self.api_key
            url = f"{self.api_url}/api/camera/frame"

            while not self._stop_event.is_set():
                jpeg_data = None
                with self._frame_lock:
                    jpeg_data = self._latest_jpeg
                    self._latest_jpeg = None

                if jpeg_data:
                    try:
                        session.post(url, data=jpeg_data, headers=headers, timeout=0.5)
                    except Exception:
                        pass
                time.sleep(0.033)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _async_post(self, endpoint, data):
        if not self.enabled:
            return

        def worker():
            try:
                import requests
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["x-cv-api-key"] = self.api_key
                url = f"{self.api_url}{endpoint}"
                resp = requests.post(url, json=data, headers=headers, timeout=2.0)
                if not resp.ok:
                    print(f"[DashboardBridge WARN] POST {endpoint} returned {resp.status_code}")
            except Exception as e:
                print(f"[DashboardBridge WARN] POST {endpoint} failed: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def start_cycle(self):
        if not self.enabled:
            return

        def worker():
            try:
                import requests
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["x-cv-api-key"] = self.api_key
                url = f"{self.api_url}/api/assembly/start"
                resp = requests.post(url, json={"operator_id": self.operator_id}, headers=headers, timeout=2.5)
                if resp.ok:
                    res_json = resp.json()
                    self.cycle_id = res_json.get("assembly", {}).get("cycle_id")
                    print(f"[DashboardBridge] Connected! Tracking Assembly Cycle: {self.cycle_id}")
                else:
                    print(f"[DashboardBridge ERROR] Could not register cycle: {resp.status_code} {resp.text}")
            except Exception as e:
                print(f"[DashboardBridge ERROR] Connection failed to {self.api_url}: {e}")

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=1.0)

    def log_event(self, state_index, state_name, status, confidence=1.0, diagnostic=""):
        if not self.enabled or not self.cycle_id:
            return
        self._async_post(
            "/api/assembly/event",
            {
                "cycle_id": self.cycle_id,
                "state_index": state_index,
                "state_name": state_name,
                "status": status,
                "confidence": float(confidence),
                "diagnostic": diagnostic,
            },
        )

    def end_cycle(self, status="pass", completed=8):
        if not self.enabled or not self.cycle_id:
            return
        cid = self.cycle_id
        self.cycle_id = None
        self._async_post(
            "/api/assembly/end",
            {
                "cycle_id": cid,
                "status": status,
                "states_completed": completed,
            },
        )


def draw_hud(frame, result, state_machine, smoothed=True, fps=None):
    """
    Renders an industrial quality inspection HUD onto the camera frame.
    Displays:
      - Bounding boxes around all detected blocks
      - Focused assembly crop bounding box
      - Incoming object callout (highlighting part in hand)
      - Top status banner (PASS / ADVANCED / HOLDING / ERROR)
      - Right-hand sequential assembly checklist (all 9 steps)
      - Picture-in-Picture (PiP) inset of the cropped assembly
      - Real-time FPS and latency counter
    """
    h, w = frame.shape[:2]

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
        status, detail = state_machine.update(state, is_valid_spatial=is_valid, diagnostic=diagnostic)
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

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_col, 2)

        lbl = f"{cname.replace('_block', '')} {dconf*100:.0f}%"
        text_col = (0, 0, 0) if cname == "yellow_block" else (255, 255, 255)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_y1 = max(0, by - 18)
        badge_y2 = max(18, by)
        cv2.rectangle(frame, (bx, badge_y1), (bx + tw + 6, badge_y2), box_col, -1)
        cv2.putText(frame, lbl, (bx + 3, badge_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1)

    # 2. Highlight Assembly ROI Box
    crop_bbox = result.get("crop_bbox")
    if crop_bbox and crop_bbox[2] > 0 and crop_bbox[3] > 0 and not is_error:
        cx, cy, cw, ch = crop_bbox
        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 200, 220), 1)

    # 3. Highlight Incoming Object (Part in Hand)
    incoming = result.get("incoming_object")
    if incoming:
        ix, iy, iw, ih = incoming["bbox"]
        is_exp = incoming["is_expected"]
        badge_col = (0, 220, 0) if is_exp else (0, 0, 255)

        cv2.rectangle(frame, (ix - 3, iy - 3), (ix + iw + 3, iy + ih + 3), badge_col, 3)
        tag = "[INCOMING: VALID]" if is_exp else "[INCOMING: WRONG PART!]"
        cv2.putText(frame, tag, (ix, max(25, iy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, badge_col, 2)

    # 4. Top Status Banner
    header_h = 160
    if is_error:
        header_color = (20, 20, 160)   # Crimson Red
        border_color = (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (w, h), border_color, 4)
    elif is_done:
        header_color = (20, 120, 20)   # Forest Green
        border_color = (0, 220, 0)
        cv2.rectangle(frame, (0, 0), (w, h), border_color, 4)
    elif status == "assembling":
        header_color = (15, 75, 120)   # Warm Amber / Dark Teal Accent
    elif status in ["advanced", "holding"] and state_machine.current_index > 0:
        header_color = (15, 85, 30)    # Emerald Green for Verified State
    elif status == "advanced":
        header_color = (15, 60, 95)    # Dark Navy / Blue-Amber Accent
    else:
        header_color = (22, 22, 22)    # Industrial Dark Slate

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

    # Left Side Status Text
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
        cv2.putText(frame, "AUTO-RESETTING FOR NEXT UNIT...", (12, 138), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (100, 255, 100), 1)
    elif status == "assembling":
        cur_title = state_machine.get_step_title(state_machine.current_index)
        cv2.putText(frame, f"INSPECTION: {cur_title.upper()}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        prompt_txt = detail or diagnostic or "Parts present on table. Please assemble."
        if prompt_txt.startswith("ASSEMBLING: "):
            prompt_txt = prompt_txt[12:]
        cv2.putText(frame, f"ACTION: {prompt_txt[:44]}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (0, 220, 255), 1)
        cv2.putText(frame, f"STATUS: ASSEMBLING (Parts Detected)", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (50, 200, 255), 1)
        next_step = state_machine.get_step_title(state_machine.current_index + 1)
        cv2.putText(frame, f"Target: [{next_step}]", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (180, 180, 180), 1)
        if incoming:
            inc_col = (0, 255, 0) if incoming["is_expected"] else (50, 50, 255)
            cv2.putText(frame, incoming["message"][:48], (12, 145), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, inc_col, 1)
    else:
        cur_title = state_machine.get_step_title(state_machine.current_index)
        cv2.putText(frame, f"INSPECTION: {cur_title.upper()}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        if state_machine.current_index > 0:
            cv2.putText(frame, f"VERIFIED: {cur_title} PASSED", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (100, 255, 100), 1)
            next_step = state_machine.get_step_title(state_machine.current_index + 1)
            cv2.putText(frame, f"STATUS: PASSED (Waiting for next part)", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (80, 240, 120), 1)
            cv2.putText(frame, f"Next Step: [{next_step}]", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (200, 220, 255), 1)
        else:
            cv2.putText(frame, f"LIVE: {display_state} ({conf*100:.0f}%)", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 220, 220), 1)
            cv2.putText(frame, f"STATUS: {status.upper()} (Dwell: {votes_ratio})", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 180, 50), 1)
            next_step = state_machine.get_step_title(state_machine.current_index + 1)
            cv2.putText(frame, f"Next Step: [{next_step}]", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (180, 180, 180), 1)
        if incoming:
            inc_col = (0, 255, 0) if incoming["is_expected"] else (50, 50, 255)
            cv2.putText(frame, incoming["message"][:48], (12, 145), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, inc_col, 1)

    steps = state_machine.get_steps_for_hud()
    for i, s in enumerate(steps):
        st = s["status"]
        if st in ["verified", "current_passed"]:
            tag = "[PASS] "
            col = (60, 235, 60)
        elif st == "next":
            tag = "[NEXT] "
            col = (240, 210, 40)
        elif st == "error_current":
            tag = "[ERR ] "
            col = (40, 40, 255)
        elif st == "current":
            tag = "[NOW ] "
            col = (200, 200, 200)
        else:
            tag = "[    ] "
            col = (110, 110, 110)

        title = s["title"].split(". ", 1)[-1]
        text = f"{tag}{i}.{title[:14]}"
        cv2.putText(frame, text, (start_x, 18 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

    # 5. Flash banner if reset just occurred
    if getattr(state_machine, "reset_flash", 0) > 0:
        state_machine.reset_flash -= 1
        b_h = 50
        by1 = int(h * 0.42)
        by2 = by1 + b_h
        cv2.rectangle(frame, (0, by1), (w, by2), (0, 180, 0), -1)
        cv2.rectangle(frame, (0, by1), (w, by2), (255, 255, 255), 2)
        msg = "SYSTEM RESET TO STEP 0"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        tx = max(10, int((w - tw) / 2))
        cv2.putText(frame, msg, (tx, by1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # 6. Bottom control bar & clickable buttons
    bar_h = 32

    # Picture-in-Picture (PiP) inset for the Cropped Assembly ROI
    crop_img = result.get("crop")
    if crop_img is not None and crop_img.size > 0:
        pip_size = 110
        pip_crop = cv2.resize(crop_img, (pip_size, pip_size))
        pip_x1 = 12
        pip_y1 = h - bar_h - pip_size - 8
        pip_x2 = pip_x1 + pip_size
        pip_y2 = pip_y1 + pip_size
        if pip_y1 > header_h:
            frame[pip_y1:pip_y2, pip_x1:pip_x2] = pip_crop
            cv2.rectangle(frame, (pip_x1, pip_y1), (pip_x2, pip_y2), (0, 220, 255), 1)
            cv2.putText(frame, "CROP ROI", (pip_x1 + 4, pip_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1)

    cv2.rectangle(frame, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "[R/Space] Reset  |  [S] Snapshot  |  [Q] Quit", (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    # Live FPS readout on bottom bar
    if fps is not None:
        ms = 1000.0 / max(fps, 1e-3)
        cv2.putText(frame, f"FPS: {fps:.1f} ({ms:.0f}ms)", (int(w * 0.42), h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 220), 1)

    # Clickable Buttons:
    btn_r_x1 = max(w - 230, int(w * 0.65))
    btn_r_x2 = btn_r_x1 + 105
    cv2.rectangle(frame, (btn_r_x1, h - bar_h + 3), (btn_r_x2, h - 3), (40, 50, 180), -1)
    cv2.putText(frame, "RESET (R)", (btn_r_x1 + 14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    btn_s_x1 = btn_r_x2 + 8
    btn_s_x2 = btn_s_x1 + 105
    cv2.rectangle(frame, (btn_s_x1, h - bar_h + 3), (btn_s_x2, h - 3), (120, 80, 20), -1)
    cv2.putText(frame, "SNAP (S)", (btn_s_x1 + 16, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

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


def open_working_camera(requested_source):
    """
    Attempts to open the requested camera, supporting:
      1. Direct IP stream (e.g. DroidCam WiFi: http://<ip>:4747/video or 192.168.x.x:4747)
      2. Device index (0, 1, 2) via DirectShow or default backend
    """
    if isinstance(requested_source, str) and not requested_source.isdigit():
        src = requested_source.strip()
        if src.lower() in ("phone", "droid", "droidcam"):
            src = "http://192.168.2.217:4747/video"
        # Auto-format DroidCam IP if missing protocol or endpoint
        if ":" in src and not src.startswith("http://") and not src.startswith("https://") and not src.startswith("rtsp://"):
            src = f"http://{src}"
        if src.startswith("http://") and ":4747" in src and not src.endswith("/video") and not src.endswith("/mjpegfeed"):
            src = f"{src.rstrip('/')}/video"

        print(f"[INFO] Connecting to camera stream: {src} ...")
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap, src
            cap.release()
        return None, requested_source

    requested_idx = int(requested_source) if isinstance(requested_source, str) else requested_source
    candidate_indices = [requested_idx] + [i for i in [0, 1, 2, 3] if i != requested_idx]

    for idx in candidate_indices:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap, idx
                cap.release()

    return None, requested_source


def run_video(video_source, detector, sm, dashboard=None):
    cap, active_src = open_working_camera(video_source)
    if cap is None:
        print(f"[ERROR] Could not open video source: {video_source}")
        print("Tip: If using DroidCam client, try device index 1 or 2:")
        print("     python live_demo.py --camera 1")
        print("     Or use the direct WiFi IP URL from the DroidCam app:")
        print("     python live_demo.py --camera http://<PHONE_IP>:4747/video")
        return

    print(f"\nLive Inspection active on camera [{active_src}].")
    print("Controls: 'r'/Space = Reset, 's' = Snapshot, 'q'/Esc = Quit. (Or click on-screen buttons)")
    if dashboard and dashboard.enabled:
        print(f"Dashboard Bridge connected to: {dashboard.api_url}")
        dashboard.start_cycle()

    window_name = "Block Assembly Checker (Live HUD)"
    button_events = {"reset": False, "snap": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_h = param.get("h", 480)
            frame_w = param.get("w", 640)
            bar_top = frame_h - 32
            if y >= bar_top:
                btn_r_x1 = max(frame_w - 230, int(frame_w * 0.65))
                btn_r_x2 = btn_r_x1 + 105
                btn_s_x1 = btn_r_x2 + 8
                btn_s_x2 = btn_s_x1 + 105
                if btn_r_x1 <= x <= btn_r_x2:
                    button_events["reset"] = True
                elif btn_s_x1 <= x <= btn_s_x2:
                    button_events["snap"] = True

    cv2.namedWindow(window_name)
    mouse_param = {"w": 640, "h": 480}
    cv2.setMouseCallback(window_name, on_mouse, mouse_param)

    prev_time = time.perf_counter()
    prev_state_idx = -1
    prev_error_active = False
    last_event_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.perf_counter()
        dt = curr_time - prev_time
        prev_time = curr_time
        fps = (1.0 / dt) if dt > 0 else 30.0

        mouse_param["w"] = frame.shape[1]
        mouse_param["h"] = frame.shape[0]

        res = detector.analyze(frame, current_step_index=sm.current_index)
        annotated = draw_hud(frame.copy(), res, sm, smoothed=True, fps=fps)

        # Stream the live AI HUD frame (with bounding boxes, labels, and PiP) to dashboard
        if dashboard and dashboard.enabled:
            dashboard.update_frame(annotated)

        # Log state advancements, defects/errors, or periodic telemetry to Dashboard backend
        if dashboard and dashboard.enabled:
            if not dashboard.cycle_id:
                dashboard.start_cycle()

            now = time.perf_counter()
            state_changed = (sm.current_index != prev_state_idx)
            error_changed = (sm.error_active != prev_error_active)
            heartbeat_due = (now - last_event_time > 1.2)

            if state_changed or error_changed or heartbeat_due:
                prev_state_idx = sm.current_index
                prev_error_active = sm.error_active
                last_event_time = now

                if sm.error_active:
                    ev_status = "error"
                    diag = sm.error_detail or res.get("diagnostic", "Assembly defect detected")
                elif sm.is_complete():
                    ev_status = "completed"
                    diag = "Assembly Completed & Verified"
                elif state_changed:
                    ev_status = "advanced"
                    diag = res.get("diagnostic", f"Advanced to Step {sm.current_index}")
                else:
                    ev_status = "holding"
                    diag = res.get("diagnostic", "Holding step verification")

                dashboard.log_event(
                    state_index=sm.current_index,
                    state_name=sm.current_state(),
                    status=ev_status,
                    confidence=res.get("confidence", 0.0),
                    diagnostic=diag,
                )

                if sm.is_complete() and state_changed:
                    dashboard.end_cycle(status="pass", completed=8)

        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        do_reset = key in (ord("r"), ord("R"), 32) or button_events["reset"]
        do_snap = key in (ord("s"), ord("S")) or button_events["snap"]

        button_events["reset"] = False
        button_events["snap"] = False

        if key in (ord("q"), ord("Q"), 27):
            break
        elif do_reset:
            sm.reset()
            prev_state_idx = -1
            prev_error_active = False
            last_event_time = 0.0
            if dashboard and dashboard.enabled:
                dashboard.start_cycle()
            print("[INFO] Sequence tracker reset to Step 0.")
        elif do_snap:
            os.makedirs("snapshots", exist_ok=True)
            ts = int(time.time())
            snap_path = f"snapshots/snapshot_{ts}.jpg"
            cv2.imwrite(snap_path, annotated)
            raw_path = f"snapshots/raw_{ts}.jpg"
            cv2.imwrite(raw_path, frame)
            print(f"[INFO] Saved snapshot to '{snap_path}' and '{raw_path}'")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live Block Assembly Inspection HUD")
    parser.add_argument("--image", type=str, default=None, help="Inspect a single image file")
    parser.add_argument("--video", type=str, default=None, help="Run inspection on a recorded video file")
    parser.add_argument("--camera", type=str, default="0", help="Webcam index (0, 1) or IP URL")
    parser.add_argument("--dashboard", type=str, default=None, help="Optional FastAPI dashboard backend URL (e.g. http://localhost:8000)")
    parser.add_argument("--operator", type=str, default="OP001", help="Operator ID for dashboard session (default: OP001)")
    parser.add_argument("--api-key", type=str, default=os.getenv("CV_API_KEY", "assembly-local-key"), help="CV API key for dashboard authentication")
    args = parser.parse_args()

    detector = ComponentDetector()
    sm = AssemblyStateMachine()
    dashboard = DashboardBridge(api_url=args.dashboard, operator_id=args.operator, api_key=args.api_key) if args.dashboard else None

    if args.image:
        run_image(args.image, detector, sm)
    elif args.video:
        run_video(args.video, detector, sm, dashboard=dashboard)
    else:
        run_video(args.camera, detector, sm, dashboard=dashboard)


if __name__ == "__main__":
    main()
