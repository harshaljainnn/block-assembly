"""
ComponentDetector: Unified inference coordinator for Block Assembly Quality Inspection.
Combines:
  1. Micro Perception: BlockDetector (YOLO / Plastic Color Object Detector)
     - Localizes individual blocks, counts parts, tracks incoming block in hand
  2. Macro Perception: YOLO State Classifier
     - Classifies overall assembly structure (runs_classify/block_states/weights/best.pt)
  3. Spatial Rule Engine: AssemblyGraph
     - Verifies joints, adjacencies, and detects loose or detached blocks
  4. Dual-Perception Mutual Corroboration
     - Blends micro part detection with macro state classification
  5. Incoming Object Checker
     - Validates part in hand against next required assembly step
"""

import os
import cv2
import numpy as np
from block_detector import BlockDetector, YOLO_DET_MODEL_PATH
from assembly_graph import AssemblyGraph, are_adjacent, compute_iou
from block_config import ASSEMBLY_STATES, STEP_TITLES, NEXT_REQUIRED_PART, EXPECTED_PARTS_PER_STATE, PROJECT_ROOT
from block_cropper import crop_assembly, is_workspace_empty

CROPPED_CLS_MODEL_PATH = os.path.join(PROJECT_ROOT, "runs_classify", "block_states_cropped", "weights", "best.pt")
YOLO_CLS_MODEL_PATH = os.path.join(PROJECT_ROOT, "runs_classify", "block_states", "weights", "best.pt")


class ComponentDetector:
    def __init__(
        self,
        det_model_path=YOLO_DET_MODEL_PATH,
        cls_model_path=None,
        conf_threshold=0.40,
    ):
        self.block_detector = BlockDetector(model_path=det_model_path, conf_threshold=conf_threshold)
        self.assembly_graph = AssemblyGraph()

        if cls_model_path is None:
            if os.path.exists(CROPPED_CLS_MODEL_PATH):
                cls_model_path = CROPPED_CLS_MODEL_PATH
                self.is_cropped_model = True
            else:
                cls_model_path = YOLO_CLS_MODEL_PATH
                self.is_cropped_model = False
        else:
            self.is_cropped_model = "cropped" in cls_model_path

        self.cls_model = None
        if os.path.exists(cls_model_path):
            try:
                from ultralytics import YOLO
                self.cls_model = YOLO(cls_model_path)
                model_type = "Cropped Assembly Classifier" if self.is_cropped_model else "Full-frame Classifier"
                print(f"[ComponentDetector] Loaded {model_type} from '{cls_model_path}'")
            except Exception as e:
                print(f"[ComponentDetector] Could not load state classifier ({e}).")

    def corroborate_stacked_reds(self, detections, img, current_step_index=0, cls_pred=None):
        """
        Ensures that when the assembly is at Step 5 or beyond (where 2 red blocks are stacked in the body),
        the body red block is properly recognized and split into TWO separate red blocks if YOLO detected them as 1.
        Strictly respects physical orientation:
          - Taller than wide (ch > cw) -> splits HORIZONTALLY along Y into TOP & BOTTOM (stacked).
          - Wider than tall (cw > ch) -> splits VERTICALLY along X into LEFT & RIGHT (side-by-side).
          - Single 2x2 blocks (aspect < 1.25 and area < 8500) are NEVER split.
        """
        is_step5_plus = (current_step_index >= 5) or (
            current_step_index >= 4 and cls_pred in [
                "state_5_bothred", "state_6_yellowafter2red", "state_7_finalred", "state_8_complete"
            ]
        )
        if not is_step5_plus or not detections or img is None:
            return detections

        greens = [d for d in detections if d.get("class_name") == "green_block"]
        gbox = greens[0]["bbox"] if greens else None
        reds = [d for d in detections if d.get("class_name") == "red_block" and self.block_detector.is_real_block(d, img)]

        if not reds:
            return detections

        # Identify body red blocks on the assembly:
        # Tier 1: reds directly adjacent to green beam
        # Tier 2: reds stacked/adjacent to Tier 1 reds
        body_reds = []
        if gbox is not None:
            tier1 = [r for r in reds if are_adjacent(r["bbox"], gbox, max_gap=90)]
            body_reds.extend(tier1)
            tier2 = [r for r in reds if r not in body_reds and any(are_adjacent(r["bbox"], t["bbox"], max_gap=60) for t in tier1)]
            body_reds.extend(tier2)
        else:
            body_reds = [r for r in reds if r.get("area", 0) >= 4000]

        # If the body already has 2 or more red blocks, never split again!
        if len(body_reds) >= 2:
            return detections

        target_r = body_reds[0] if body_reds else max(reds, key=lambda x: x["area"])
        bx, by, bw, bh = target_r["bbox"]
        h_img, w_img = img.shape[:2]

        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w_img, bx + bw), min(h_img, by + bh)
        cw, ch = x2 - x1, y2 - y1

        if cw < 30 or ch < 30:
            return detections

        aspect = max(cw, ch) / max(min(cw, ch), 1)
        target_area = target_r.get("area", cw * ch)

        # NEVER split a single 2x2 red block!
        if aspect < 1.25 and target_area < 8500:
            return detections

        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))

        # Check orientation strictly based on cw and ch:
        # NEVER use gbox to invert or override the red block's physical geometry!
        if cw > ch:
            # WIDER than tall: side-by-side blocks!
            # Cut axis is strictly VERTICAL along X (producing Left and Right blocks)
            if cw < 1.35 * ch and target_area < 11000:
                return detections

            c1, c2 = int(0.30 * cw), int(0.70 * cw)
            col_sums = np.sum(sobelx, axis=0) if cw > 10 else [0]
            best_c = c1 + int(np.argmax(col_sums[c1:c2])) if c2 > c1 else cw // 2
            w1 = best_c
            w2 = bw - w1
            if min(w1, w2) / max(w1, w2) < 0.40:
                w1 = cw // 2
                w2 = bw - w1

            if w1 >= 25 and w2 >= 25:
                conf = target_r.get("confidence", 0.92)
                cls_id = target_r.get("class_id", 2)
                new_r1 = {
                    "class_name": "red_block",
                    "class_id": cls_id,
                    "bbox": (bx, by, w1, bh),
                    "center": (bx + w1 // 2, by + bh // 2),
                    "area": w1 * bh,
                    "confidence": float(conf),
                    "source": "stacked_corroborated",
                }
                new_r2 = {
                    "class_name": "red_block",
                    "class_id": cls_id,
                    "bbox": (bx + w1, by, w2, bh),
                    "center": (bx + w1 + w2 // 2, by + bh // 2),
                    "area": w2 * bh,
                    "confidence": float(conf),
                    "source": "stacked_corroborated",
                }
                res = [d for d in detections if d != target_r] + [new_r1, new_r2]
                res.sort(key=lambda d: d["area"], reverse=True)
                return res

        elif ch > cw:
            # TALLER than wide: stacked blocks!
            # Cut axis is strictly HORIZONTAL along Y (producing Top and Bottom blocks)
            if ch < 1.48 * cw and target_area < 11000:
                return detections

            r1, r2 = int(0.30 * ch), int(0.70 * ch)
            row_sums = np.sum(sobely, axis=1) if ch > 10 else [0]
            best_r = r1 + int(np.argmax(row_sums[r1:r2])) if r2 > r1 else ch // 2
            h1 = best_r
            h2 = bh - h1
            if min(h1, h2) / max(h1, h2) < 0.45:
                h1 = ch // 2
                h2 = bh - h1

            if h1 >= 25 and h2 >= 25:
                conf = target_r.get("confidence", 0.92)
                cls_id = target_r.get("class_id", 2)
                new_r1 = {
                    "class_name": "red_block",
                    "class_id": cls_id,
                    "bbox": (bx, by, bw, h1),
                    "center": (bx + bw // 2, by + h1 // 2),
                    "area": bw * h1,
                    "confidence": float(conf),
                    "source": "stacked_corroborated",
                }
                new_r2 = {
                    "class_name": "red_block",
                    "class_id": cls_id,
                    "bbox": (bx, by + h1, bw, h2),
                    "center": (bx + bw // 2, by + h1 + h2 // 2),
                    "area": bw * h2,
                    "confidence": float(conf),
                    "source": "stacked_corroborated",
                }
                res = [d for d in detections if d != target_r] + [new_r1, new_r2]
                res.sort(key=lambda d: d["area"], reverse=True)
                return res

        return detections


    def split_merged_blue_feet(self, blue_dets, img, green_det=None):
        """
        If a single blue detection encompasses both feet along the green beam or two adjacent feet,
        splits it into two separate foot detections.
        Strictly respects orientation:
          - cw > ch: cuts vertically along X into Left Foot and Right Foot.
          - ch > cw: cuts horizontally along Y into Top Foot and Bottom Foot.
        """
        if not blue_dets or img is None:
            return blue_dets

        gbox = green_det["bbox"] if green_det is not None else None
        h_img, w_img = img.shape[:2]
        result = []

        is_vertical_beam = False
        is_horizontal_beam = False
        if gbox is not None:
            gw, gh = gbox[2], gbox[3]
            if gh >= gw * 1.2:
                is_vertical_beam = True
            elif gw >= gh * 1.2:
                is_horizontal_beam = True

        for bdet in blue_dets:
            bx, by, bw, bh = bdet["bbox"]
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(w_img, bx + bw), min(h_img, by + bh)
            cw, ch = x2 - x1, y2 - y1

            if cw < 35 or ch < 35:
                result.append(bdet)
                continue

            aspect = max(cw, ch) / max(min(cw, ch), 1)
            area = cw * ch

            # Single foot protection
            if aspect < 1.25 and area < 8500:
                result.append(bdet)
                continue

            split_axis = None
            if ch > cw:
                # TALLER than wide: stacked along Y
                if ch >= 1.28 * cw or (is_vertical_beam and ch >= 50) or area >= 10500:
                    split_axis = "horizontal"
            elif cw > ch:
                # WIDER than tall: side-by-side along X
                if cw >= 1.28 * ch or (is_horizontal_beam and cw >= 50) or area >= 10500:
                    split_axis = "vertical"

            if split_axis is None:
                result.append(bdet)
                continue

            crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            if split_axis == "horizontal":
                r1, r2 = int(0.25 * ch), int(0.75 * ch)
                sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
                row_sums = np.sum(sobely, axis=1) if ch > 10 else [0]
                best_r = r1 + int(np.argmax(row_sums[r1:r2])) if r2 > r1 else ch // 2
                h1 = best_r
                h2 = bh - h1
                if min(h1, h2) / max(h1, h2) < 0.40:
                    h1 = ch // 2
                    h2 = bh - h1
                if h1 >= 25 and h2 >= 25:
                    conf = bdet.get("confidence", 0.90)
                    result.append({
                        "class_name": "blue_block",
                        "class_id": 0,
                        "bbox": (bx, by, bw, h1),
                        "center": (bx + bw // 2, by + h1 // 2),
                        "area": bw * h1,
                        "confidence": conf,
                        "source": "split_feet",
                    })
                    result.append({
                        "class_name": "blue_block",
                        "class_id": 0,
                        "bbox": (bx, by + h1, bw, h2),
                        "center": (bx + bw // 2, by + h1 + h2 // 2),
                        "area": bw * h2,
                        "confidence": conf,
                        "source": "split_feet",
                    })
                    continue

            elif split_axis == "vertical":
                c1, c2 = int(0.25 * cw), int(0.75 * cw)
                sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
                col_sums = np.sum(sobelx, axis=0) if cw > 10 else [0]
                best_c = c1 + int(np.argmax(col_sums[c1:c2])) if c2 > c1 else cw // 2
                w1 = best_c
                w2 = bw - w1
                if min(w1, w2) / max(w1, w2) < 0.40:
                    w1 = cw // 2
                    w2 = bw - w1
                if w1 >= 25 and w2 >= 25:
                    conf = bdet.get("confidence", 0.90)
                    result.append({
                        "class_name": "blue_block",
                        "class_id": 0,
                        "bbox": (bx, by, w1, bh),
                        "center": (bx + w1 // 2, by + bh // 2),
                        "area": w1 * bh,
                        "confidence": conf,
                        "source": "split_feet",
                    })
                    result.append({
                        "class_name": "blue_block",
                        "class_id": 0,
                        "bbox": (bx + w1, by, w2, bh),
                        "center": (bx + w1 + w2 // 2, by + bh // 2),
                        "area": w2 * bh,
                        "confidence": conf,
                        "source": "split_feet",
                    })
                    continue

            result.append(bdet)

        return result

    def cluster_feet_along_beam(self, blue_dets, green_det=None, max_dim=150):
        """
        Groups blue block boxes that belong to the same physical foot.
        Because the green beam passes over/under the blue feet, each foot often produces
        separate detections for its top and bottom exposed studs/halves.
        Enforces physical unit dimensions (max_dim <= 150 px) and spatial proximity,
        ensuring distinct separated feet along the beam are NEVER merged into one.
        """
        if not blue_dets:
            return []

        # Sort by area descending so primary body is cluster anchor
        sorted_dets = sorted(blue_dets, key=lambda d: d.get("area", d["bbox"][2] * d["bbox"][3]), reverse=True)
        clusters = []

        gbox = green_det["bbox"] if green_det is not None else None
        is_horiz = (gbox[2] >= gbox[3]) if gbox is not None else True

        for d in sorted_dets:
            b = d["bbox"]
            bx, by, bw, bh = b
            bcx, bcy = bx + bw // 2, by + bh // 2

            merged = False
            for c in clusters:
                cb = c["bbox"]
                cx, cy, cw, ch = cb
                ccx, ccy = cx + cw // 2, cy + ch // 2

                x1 = min(bx, cx)
                y1 = min(by, cy)
                x2 = max(bx + bw, cx + cw)
                y2 = max(by + bh, cy + ch)
                uw = x2 - x1
                uh = y2 - y1

                # Physical size constraint: union cannot exceed single foot dimensions
                if uw > max_dim or uh > max_dim:
                    continue

                can_merge = False
                # Significant overlap of same foot
                if compute_iou(b, cb) > 0.20:
                    can_merge = True
                elif gbox is not None:
                    # Foot split across the green beam (top stud and bottom stud of the same foot)
                    if is_horiz:
                        # Beam is horizontal: same foot means matching X coordinate along beam
                        if abs(bcx - ccx) <= 55 and uw <= 140 and (y1 <= gbox[1] + gbox[3] + 25 and y2 >= gbox[1] - 25):
                            can_merge = True
                    else:
                        # Beam is vertical: same foot means matching Y coordinate along beam
                        if abs(bcy - ccy) <= 55 and uh <= 140 and (x1 <= gbox[0] + gbox[2] + 25 and x2 >= gbox[0] - 25):
                            can_merge = True

                if can_merge:
                    c["bbox"] = (x1, y1, uw, uh)
                    c["confidence"] = max(c["confidence"], d["confidence"])
                    c["boxes"].append(b)
                    merged = True
                    break

            if not merged:
                clusters.append({
                    "class_name": "blue_block",
                    "class_id": 0,
                    "bbox": b,
                    "confidence": d["confidence"],
                    "boxes": [b],
                })

        feet = []
        for c in clusters:
            bx, by, bw, bh = c["bbox"]
            feet.append({
                "class_name": "blue_block",
                "class_id": 0,
                "bbox": (bx, by, bw, bh),
                "center": (bx + bw // 2, by + bh // 2),
                "area": bw * bh,
                "confidence": c["confidence"],
                "source": "clustered_foot",
            })

        # The base has physically at most 2 blue feet.
        # If > 2 feet are detected, filter out spurious slivers / stud noise and keep the 2 valid feet.
        if len(feet) > 2 and gbox is not None:
            feet.sort(key=lambda f: (1 if are_adjacent(f["bbox"], gbox, max_gap=45) else 0, f["area"]), reverse=True)
            filtered_feet = []
            for f in feet:
                is_sub = False
                for kf in filtered_feet:
                    if are_adjacent(f["bbox"], kf["bbox"], max_gap=25) or compute_iou(f["bbox"], kf["bbox"]) > 0.10:
                        is_sub = True
                        break
                if not is_sub:
                    filtered_feet.append(f)
                if len(filtered_feet) == 2:
                    break
            if len(filtered_feet) == 2:
                return filtered_feet

        return feet


    def cluster_same_color_blocks(self, dets, max_dim=150):
        """
        Groups bounding boxes of the same color that belong to the same physical block (e.g. stud + body).
        Enforces physical unit dimension constraint and prevents merging two distinct full blocks.
        """
        if not dets or len(dets) <= 1:
            return dets

        sorted_dets = sorted(dets, key=lambda d: d.get("area", d["bbox"][2] * d["bbox"][3]), reverse=True)
        clusters = []

        for d in sorted_dets:
            b = d["bbox"]
            bx, by, bw, bh = b
            area_b = b[2] * b[3]
            merged = False
            for c in clusters:
                cb = c["bbox"]
                cx, cy, cw, ch = cb
                area_cb = cb[2] * cb[3]
                x1 = min(bx, cx)
                y1 = min(by, cy)
                x2 = max(bx + bw, cx + cw)
                y2 = max(by + bh, cy + ch)
                uw = x2 - x1
                uh = y2 - y1

                # Must not exceed physical dimensions of a single 2x2 block
                if uw > max_dim or uh > max_dim:
                    continue

                # NEVER merge two full-sized blocks! Only merge if one is a small stud sliver
                if min(area_b, area_cb) >= 3500 and min(b[2], b[3]) >= 26 and min(cb[2], cb[3]) >= 26:
                    continue

                # Check proximity or overlap
                if compute_iou(b, cb) > 0.15 or are_adjacent(b, cb, max_gap=12):
                    c["bbox"] = (x1, y1, uw, uh)
                    c["center"] = (x1 + uw // 2, y1 + uh // 2)
                    c["area"] = uw * uh
                    c["confidence"] = max(c["confidence"], d["confidence"])
                    merged = True
                    break

            if not merged:
                clusters.append(dict(d))

        return clusters

    def deduplicate_detections(self, detections, img=None, current_step_index=0):
        """
        Suppresses duplicate stud slivers, resolves split feet across the green beam,
        and ensures true physical block counts across all colors.
        """
        if not detections:
            return []
        by_class = {}
        for d in detections:
            by_class.setdefault(d["class_name"], []).append(d)

        cleaned = []
        for cname, c_dets in by_class.items():
            c_dets.sort(key=lambda x: x["area"], reverse=True)
            kept = []
            for d in c_dets:
                boxA = d["bbox"]
                is_dup = False
                for k in kept:
                    boxB = k["bbox"]
                    xA = max(boxA[0], boxB[0])
                    yA = max(boxA[1], boxB[1])
                    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
                    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    containment = inter / float(boxA[2] * boxA[3] + 1e-6)
                    iou = compute_iou(boxA, boxB)
                    if iou > 0.30 or containment > 0.50:
                        is_dup = True
                        break
                    # Stud sliver attached to top/bottom of block
                    if d["area"] < 3500 and are_adjacent(boxA, boxB, max_gap=15):
                        is_dup = True
                        break
                if not is_dup:
                    kept.append(d)
            cleaned.extend(kept)

        # Separate blocks by color for physical clustering
        blues = [d for d in cleaned if d["class_name"] == "blue_block"]
        greens = [d for d in cleaned if d["class_name"] == "green_block"]
        reds = [d for d in cleaned if d["class_name"] == "red_block"]
        yellows = [d for d in cleaned if d["class_name"] == "yellow_block"]

        gdet = greens[0] if greens else None

        # 1. Blue feet: split wide/tall boxes covering both feet only when not strictly at Step 1 with single foot
        if blues and img is not None and current_step_index != 1:
            blues = self.split_merged_blue_feet(blues, img, gdet)
        resolved_blues = self.cluster_feet_along_beam(blues, gdet) if len(blues) > 1 else blues
        if current_step_index == 1 and len(resolved_blues) > 1 and gdet is not None:
            # Step 1 physically has only 1 blue foot; cluster all blue stud slivers near green beam into one foot
            bx1 = min(b["bbox"][0] for b in resolved_blues)
            by1 = min(b["bbox"][1] for b in resolved_blues)
            bx2 = max(b["bbox"][0] + b["bbox"][2] for b in resolved_blues)
            by2 = max(b["bbox"][1] + b["bbox"][3] for b in resolved_blues)
            resolved_blues = [{
                "class_name": "blue_block",
                "class_id": 0,
                "bbox": (bx1, by1, bx2 - bx1, by2 - by1),
                "center": (bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2),
                "area": (bx2 - bx1) * (by2 - by1),
                "confidence": max(b["confidence"] for b in resolved_blues),
                "source": "step1_clustered",
            }]

        # 2. Red blocks: at Step 3, ensure any secondary stud detection on the 1st red block is merged
        if current_step_index == 3 and len(reds) > 1 and gdet is not None:
            gbox = gdet["bbox"]
            body_reds = [r for r in reds if are_adjacent(r["bbox"], gbox, max_gap=80)]
            if len(body_reds) >= 2:
                bx1 = min(r["bbox"][0] for r in body_reds)
                by1 = min(r["bbox"][1] for r in body_reds)
                bx2 = max(r["bbox"][0] + r["bbox"][2] for r in body_reds)
                by2 = max(r["bbox"][1] + r["bbox"][3] for r in body_reds)
                merged_r = {
                    "class_name": "red_block",
                    "class_id": 2,
                    "bbox": (bx1, by1, bx2 - bx1, by2 - by1),
                    "center": (bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2),
                    "area": (bx2 - bx1) * (by2 - by1),
                    "confidence": max(r["confidence"] for r in body_reds),
                    "source": "merged_step3",
                }
                other_reds = [r for r in reds if r not in body_reds]
                reds = [merged_r] + other_reds

        resolved_reds = self.cluster_same_color_blocks(reds, max_dim=150) if len(reds) > 1 else reds
        resolved_yellows = self.cluster_same_color_blocks(yellows, max_dim=150) if len(yellows) > 1 else yellows

        return greens + resolved_blues + resolved_reds + resolved_yellows

    def analyze(self, img_or_path, current_step_index=0):
        """
        Analyzes a single frame:
          - Detects individual block parts and counts
          - Enforces empty-workspace check for State 0
          - Identifies incoming part held in hand
          - Evaluates spatial assembly graph
          - Corroborates with whole-assembly classifier (physically validated)
          - Validates incoming block against next required step
        """
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path

        if img is None:
            return {
                "error": "Failed to read image",
                "predicted_state": "state_0_unstarted",
                "confidence": 0.0,
                "is_valid": False,
                "diagnostic": "Image load failure",
                "detections": [],
                "incoming_object": None,
                "part_counts": {},
                "spatial_checks": [],
                "signals": {},
            }

        # 1. Detect all blocks in frame with YOLO/plastic detector
        raw_detections = self.block_detector.detect(img, current_step_index=current_step_index)
        cleaned_detections = self.deduplicate_detections(raw_detections, img=img, current_step_index=current_step_index)


        # 2. Strict Empty Frame Check: if workspace has 0 blocks, it is strictly state_0_unstarted
        if not cleaned_detections:
            return {
                "predicted_state": "state_0_unstarted",
                "confidence": 1.0,
                "is_valid": True,
                "diagnostic": "Workspace clear (present parts to begin)",
                "detections": [],
                "incoming_object": None,
                "part_counts": {},
                "spatial_checks": [],
                "signals": {"cls_pred": None, "cls_conf": 0.0, "graph_state": "state_0_unstarted"},
            }

        # 3. Assembly Cropping & Macro State Classifier
        crop_img, crop_bbox = crop_assembly(img)
        cls_pred = None
        cls_conf = 0.0
        det_colors = {d["class_name"] for d in cleaned_detections}

        if self.cls_model is not None:
            cls_input = crop_img if (self.is_cropped_model and crop_img is not None) else img
            res_cls = self.cls_model.predict(cls_input, imgsz=384, verbose=False)[0]
            cand_pred = res_cls.names[res_cls.probs.top1]
            cand_conf = float(res_cls.probs.top1conf)

            # Sanity-check: if YOLO missed a faint yellow or red block, verify with plastic color detection in crop
            if cand_conf >= 0.70 and crop_img is not None and crop_img.size > 0:
                if cand_pred in ["state_4_yellowred", "state_5_bothred", "state_6_yellowafter2red", "state_7_finalred", "state_8_complete"] and "yellow_block" not in det_colors:
                    crop_colors = self.block_detector.detect_color_plastic(crop_img)
                    if any(c["class_name"] == "yellow_block" for c in crop_colors):
                        det_colors.add("yellow_block")
                if cand_pred in ["state_3_first_red", "state_4_yellowred", "state_5_bothred", "state_6_yellowafter2red", "state_7_finalred", "state_8_complete"] and "red_block" not in det_colors:
                    crop_colors = self.block_detector.detect_color_plastic(crop_img)
                    if any(c["class_name"] == "red_block" for c in crop_colors):
                        det_colors.add("red_block")

            is_physically_consistent = True
            if cand_pred in ["state_3_first_red", "state_4_yellowred", "state_5_bothred", "state_6_yellowafter2red", "state_7_finalred", "state_8_complete"]:
                if "red_block" not in det_colors and cand_conf < 0.90:
                    is_physically_consistent = False
            if cand_pred in ["state_4_yellowred", "state_5_bothred", "state_6_yellowafter2red", "state_7_finalred", "state_8_complete"]:
                if "yellow_block" not in det_colors and cand_conf < 0.90:
                    is_physically_consistent = False
            if cand_pred in ["state_1_greenblue", "state_2_green2blue", "state_3_first_red"] and len(cleaned_detections) < 4:
                if "green_block" not in det_colors and cand_conf < 0.90:
                    is_physically_consistent = False

            if is_physically_consistent:
                cls_pred = cand_pred
                cls_conf = cand_conf

        # 4. Multi-red splitting for Step 5+
        detections = self.corroborate_stacked_reds(cleaned_detections, img, current_step_index=current_step_index, cls_pred=cls_pred)

        # 5. Incoming object tracking
        incoming = self.block_detector.identify_incoming_object(detections)
        assembly_detections = detections

        # If incoming block is needed for current/next step, do NOT exclude it from assembly evaluation!
        if incoming is not None and len(detections) > 1:
            target_idx = min(current_step_index + 1, len(ASSEMBLY_STATES) - 1)
            target_sname = ASSEMBLY_STATES[target_idx]
            needed_parts = EXPECTED_PARTS_PER_STATE.get(target_sname, {}).get("parts", {})
            curr_parts = {}
            for d in detections:
                curr_parts[d["class_name"]] = curr_parts.get(d["class_name"], 0) + 1

            inc_cls = incoming["class_name"]
            # Only isolate if count strictly exceeds what the next step expects
            if curr_parts.get(inc_cls, 0) > needed_parts.get(inc_cls, 0):
                assembly_detections = [d for d in detections if d != incoming]
            else:
                assembly_detections = detections

        # 6. Spatial Assembly Graph Evaluation (Physical Ground Truth)
        target_state = ASSEMBLY_STATES[current_step_index] if current_step_index < len(ASSEMBLY_STATES) else None
        graph_eval = self.assembly_graph.evaluate(assembly_detections, current_target_step=target_state)

        inferred_state = graph_eval["inferred_state"]
        conf = graph_eval["confidence"]
        is_valid = graph_eval["is_valid"]
        diagnostic = graph_eval["diagnostic"]

        # Dual-Perception: Corroborate with classifier
        if cls_pred is not None:
            # Check if spatial graph flagged an explicit physical violation (e.g. feet on opposite sides, illegal attachment)
            is_explicit_violation = (not is_valid) and any(
                diagnostic.startswith(prefix) for prefix in ["INCORRECT", "WRONG"]
            )

            next_state = ASSEMBLY_STATES[current_step_index + 1] if current_step_index + 1 < len(ASSEMBLY_STATES) else None

            # 1. High-confidence cropped classifier corroboration for target state or next step
            # Can resolve bounding box clustering/overlap ambiguities, but CANNOT override an explicit physical defect!
            if not is_explicit_violation:
                if cls_conf >= 0.48 and cls_pred == target_state:
                    inferred_state = cls_pred
                    conf = max(conf, cls_conf)
                    is_valid = True
                    diagnostic = f"PASS: {target_state} verified by inspection ({cls_conf*100:.1f}%)"
                elif cls_conf >= 0.75 and cls_pred == next_state and (is_valid or not diagnostic.startswith("INCORRECT")):
                    inferred_state = cls_pred
                    conf = max(conf, cls_conf)
                    is_valid = True
                    diagnostic = f"PASS: Advanced to {next_state} verified by inspection ({cls_conf*100:.1f}%)"
            elif is_valid:
                if graph_eval["inferred_state"] == cls_pred:
                    conf = min(0.99, max(conf, (conf + cls_conf) / 2.0))

        # 7. Incoming Object Callout
        incoming_info = None
        if incoming is not None:
            incoming_cls = incoming["class_name"]
            expected_block = NEXT_REQUIRED_PART.get(current_step_index)
            is_expected = (expected_block is None) or (incoming_cls == expected_block)

            next_idx = min(current_step_index + 1, len(ASSEMBLY_STATES) - 1)
            next_title = STEP_TITLES.get(ASSEMBLY_STATES[next_idx], "Next Step")

            if is_expected:
                msg = f"[INCOMING: VALID] {incoming_cls} for {next_title}"
            else:
                msg = f"[INCOMING: WRONG PART!] Expected {expected_block} for {next_title}, but detected {incoming_cls}"

            incoming_info = {
                "class_name": incoming_cls,
                "confidence": incoming["confidence"],
                "bbox": incoming["bbox"],
                "is_expected": is_expected,
                "expected_block": expected_block,
                "message": msg,
            }

        return {
            "predicted_state": inferred_state,
            "confidence": conf,
            "is_valid": is_valid,
            "diagnostic": diagnostic,
            "detections": detections,
            "incoming_object": incoming_info,
            "crop": crop_img,
            "crop_bbox": crop_bbox,
            "part_counts": graph_eval["part_counts"],
            "spatial_checks": graph_eval["spatial_checks"],
            "signals": {
                "cls_pred": cls_pred,
                "cls_conf": cls_conf,
                "graph_state": graph_eval["inferred_state"],
            },
        }
