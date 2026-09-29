"""
Block Detector module: Localizes and identifies individual blocks.
Supports:
  1. Deep learning YOLOv8 object detection (when weights are available in runs_detect/block_detector/weights/best.pt)
  2. Plastic-ratio color segmentation engine (zero-training fallback, skin-tone invariant)
  3. Incoming part tracking (isolates object in hand from the assembled cluster)
"""

import os
import cv2
import numpy as np
from block_config import BLOCK_CLASSES, BLOCK_COLORS_BGR

YOLO_DET_MODEL_PATH = "runs_detect/block_detector/weights/best.pt"


class BlockDetector:
    def __init__(self, model_path=YOLO_DET_MODEL_PATH, conf_threshold=0.40):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.yolo_model = None

        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(model_path)
                print(f"[BlockDetector] Loaded trained YOLO model from '{model_path}'")
            except Exception as e:
                print(f"[BlockDetector] Could not load YOLO model ({e}). Using plastic-ratio detector.")
        else:
            print("[BlockDetector] No trained YOLO weights found. Using plastic-ratio color detector.")

    def detect_color_plastic(self, img, min_area=600):
        """
        Segment blocks using high-saturation plastic color ratios that strictly reject human skin.
        Returns: list of dicts {class_name, class_id, bbox: (x, y, w, h), center: (cx, cy), area, confidence, source}
        """
        h, w = img.shape[:2]
        b, g, r = cv2.split(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        # 1. Blue: High Blue channel relative to Red and Green, Hue 90-135
        blue_mask = (
            (b > 75)
            & (b > 1.25 * r.astype(np.float32))
            & (b > 1.15 * g.astype(np.float32))
            & (hue >= 90)
            & (hue <= 135)
            & (sat > 65)
        )

        # 2. Green: High Green channel, Hue 35-85
        green_mask = (
            (g > 70)
            & (g > 1.20 * r.astype(np.float32))
            & (g > 1.15 * b.astype(np.float32))
            & (hue >= 35)
            & (hue <= 85)
            & (sat > 60)
        )

        # 3. Red: High Red channel relative to Green & Blue, high saturation (strictly eliminates skin)
        red_mask = (
            (r > 100)
            & (r > 1.65 * g.astype(np.float32))
            & (r > 1.65 * b.astype(np.float32))
            & ((hue <= 12) | (hue >= 168))
            & (sat > 115)
            & (val > 65)
        )

        # 4. Yellow: High Red & Green, low Blue, Hue 16-34
        yellow_mask = (
            (r > 100)
            & (g > 80)
            & (r > 1.45 * b.astype(np.float32))
            & (g > 1.35 * b.astype(np.float32))
            & (hue >= 16)
            & (hue <= 34)
            & (sat > 95)
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        masks = {
            "blue_block": blue_mask.astype(np.uint8) * 255,
            "green_block": green_mask.astype(np.uint8) * 255,
            "red_block": red_mask.astype(np.uint8) * 255,
            "yellow_block": yellow_mask.astype(np.uint8) * 255,
        }

        detections = []
        for cls_name, raw_mask in masks.items():
            cls_id = BLOCK_CLASSES.index(cls_name)
            # Morphological cleaning
            cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

            cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area >= min_area:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    aspect = max(bw, bh) / max(min(bw, bh), 1)
                    if aspect > 4.5:
                        continue

                    cx, cy = bx + bw // 2, by + bh // 2
                    hull = cv2.convexHull(c)
                    solidity = float(area) / max(cv2.contourArea(hull), 1)
                    conf = min(0.96, max(0.55, 0.45 + 0.50 * solidity))

                    detections.append({
                        "class_name": cls_name,
                        "class_id": cls_id,
                        "bbox": (bx, by, bw, bh),
                        "center": (cx, cy),
                        "area": area,
                        "confidence": float(conf),
                        "source": "color_plastic",
                    })

        # Sort largest area first
        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections

    def detect_yolo(self, img):
        """Runs inference with the trained YOLO object detection model."""
        res = self.yolo_model.predict(img, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = res.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
            cx, cy = bx + bw // 2, by + bh // 2
            area = bw * bh

            detections.append({
                "class_name": cls_name,
                "class_id": cls_id,
                "bbox": (bx, by, bw, bh),
                "center": (cx, cy),
                "area": area,
                "confidence": conf,
                "source": "yolo",
            })
        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections

    def detect(self, img):
        """
        Unified detection entry point.
        Uses YOLO detector if available, otherwise falls back to plastic-ratio color detector.
        """
        if self.yolo_model is not None:
            return self.detect_yolo(img)
        return self.detect_color_plastic(img)

    def identify_incoming_object(self, detections):
        """
        Identifies an object presented by hand or held separately from the main cluster.
        Returns the detection dict of the candidate incoming block, or None.
        """
        if not detections:
            return None

        if len(detections) == 1:
            return detections[0]

        centers = np.array([d["center"] for d in detections])
        median_center = np.median(centers, axis=0)

        # Calculate distances of each block to the cluster center of mass
        dists = [np.hypot(c[0] - median_center[0], c[1] - median_center[1]) for c in centers]
        max_dist_idx = int(np.argmax(dists))

        # If the farthest block is physically separated (> 110 px), it is an incoming part
        if dists[max_dist_idx] > 110:
            return detections[max_dist_idx]

        return None
