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
        the body red block is properly recognized and split into TWO separate red blocks.
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

        # Identify body red blocks on the assembly
        if gbox is not None:
            body_reds = [r for r in reds if are_adjacent(r["bbox"], gbox, max_gap=90)]
        else:
            body_reds = [r for r in reds if r.get("area", 0) >= 6000]

        # Check if the body already has 2+ stacked/adjacent red blocks
        body_already_split = False
        if len(body_reds) >= 2:
            for i in range(len(body_reds)):
                for j in range(i + 1, len(body_reds)):
                    if are_adjacent(body_reds[i]["bbox"], body_reds[j]["bbox"], max_gap=50):
                        body_already_split = True
                        break
                if body_already_split:
                    break

        if body_already_split:
            return detections

        target_r = max(body_reds, key=lambda x: x["area"]) if body_reds else max(reds, key=lambda x: x["area"])
        bx, by, bw, bh = target_r["bbox"]
        h_img, w_img = img.shape[:2]

        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w_img, bx + bw), min(h_img, by + bh)
        cw, ch = x2 - x1, y2 - y1

        if cw >= 25 and ch >= 25:
            crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            if gbox is not None:
                split_horiz = (bh >= cw) or (gbox[2] > gbox[3])
            else:
                split_horiz = (bh >= cw)

            if split_horiz and ch >= 40:
                sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
                r1, r2 = int(0.35 * ch), int(0.65 * ch)
                row_sums = np.sum(sobely, axis=1) if ch > 10 else [0]
                best_r = r1 + int(np.argmax(row_sums[r1:r2])) if r2 > r1 else ch // 2

                h1 = best_r
                h2 = bh - h1
                if min(h1, h2) / max(h1, h2) < 0.50:
                    h1 = ch // 2
                    h2 = bh - h1

                if h1 >= 20 and h2 >= 20:
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

            elif not split_horiz and cw >= 40:
                sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
                c1, c2 = int(0.35 * cw), int(0.65 * cw)
                col_sums = np.sum(sobelx, axis=0) if cw > 10 else [0]
                best_c = c1 + int(np.argmax(col_sums[c1:c2])) if c2 > c1 else cw // 2

                w1 = best_c
                w2 = bw - w1
                if min(w1, w2) / max(w1, w2) < 0.50:
                    w1 = cw // 2
                    w2 = bw - w1

                if w1 >= 20 and w2 >= 20:
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

        return detections

    def cluster_feet_along_beam(self, blue_dets, green_det=None, max_dim=230):
        """
        Groups blue block boxes that belong to the same physical foot.
        Because the green beam passes over/under the blue feet, each foot often produces
        separate detections for its top and bottom exposed studs/halves.
        Enforces physical unit dimensions (max_dim <= 230 px) and spatial proximity,
        ensuring distinct separated blocks are NEVER merged into one.
        """
        if not blue_dets:
            return []

        # Sort by area descending so primary body is cluster anchor
        sorted_dets = sorted(blue_dets, key=lambda d: d.get("area", d["bbox"][2] * d["bbox"][3]), reverse=True)
        clusters = []

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

                # Adjacency or overlap check
                dx = max(0, max(bx - (cx + cw), cx - (bx + bw)))
                dy = max(0, max(by - (cy + ch), cy - (by + bh)))

                # If separated by large gap, never merge
                if dx > 40 and dy > 40:
                    continue

                can_merge = False
                if compute_iou(b, cb) > 0.05:
                    can_merge = True
                elif dx <= 25 and dy <= 45:
                    can_merge = True
                elif dy <= 25 and dx <= 45:
                    can_merge = True
                elif green_det is not None:
                    # Foot split across the green beam (top stud and bottom stud of the same foot)
                    gbox = green_det["bbox"]
                    is_horiz = gbox[2] >= gbox[3]
                    if is_horiz and abs(bcx - ccx) <= 45 and dy <= 95:
                        can_merge = True
                    elif (not is_horiz) and abs(bcy - ccy) <= 45 and dx <= 95:
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
        return feet

    def cluster_same_color_blocks(self, dets, max_dim=210):
        """
        Groups bounding boxes of the same color that belong to the same physical block (e.g. stud + body).
        Enforces physical unit dimension constraint (uw <= max_dim and uh <= max_dim).
        """
        if not dets or len(dets) <= 1:
            return dets

        sorted_dets = sorted(dets, key=lambda d: d.get("area", d["bbox"][2] * d["bbox"][3]), reverse=True)
        clusters = []

        for d in sorted_dets:
            b = d["bbox"]
            bx, by, bw, bh = b
            merged = False
            for c in clusters:
                cb = c["bbox"]
                cx, cy, cw, ch = cb
                x1 = min(bx, cx)
                y1 = min(by, cy)
                x2 = max(bx + bw, cx + cw)
                y2 = max(by + bh, cy + ch)
                uw = x2 - x1
                uh = y2 - y1

                # Must not exceed physical dimensions of a single 2x2 block
                if uw > max_dim or uh > max_dim:
                    continue

                # Check proximity or overlap
                if compute_iou(b, cb) > 0.05 or are_adjacent(b, cb, max_gap=25):
                    c["bbox"] = (x1, y1, uw, uh)
                    c["center"] = (x1 + uw // 2, y1 + uh // 2)
                    c["area"] = uw * uh
                    c["confidence"] = max(c["confidence"], d["confidence"])
                    merged = True
                    break

            if not merged:
                clusters.append(dict(d))

        return clusters

    def deduplicate_detections(self, detections):
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
                    if d["area"] < 8000 and are_adjacent(boxA, boxB, max_gap=15):
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
        resolved_blues = self.cluster_feet_along_beam(blues, gdet) if len(blues) > 1 else blues
        resolved_reds = self.cluster_same_color_blocks(reds, max_dim=210) if len(reds) > 1 else reds
        resolved_yellows = self.cluster_same_color_blocks(yellows, max_dim=210) if len(yellows) > 1 else yellows

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
        raw_detections = self.block_detector.detect(img)
        cleaned_detections = self.deduplicate_detections(raw_detections)

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
            res_cls = self.cls_model.predict(cls_input, verbose=False)[0]
            cand_pred = res_cls.names[res_cls.probs.top1]
            cand_conf = float(res_cls.probs.top1conf)

            # Sanity-check: reject hallucinations that require parts not physically present
            is_physically_consistent = True
            if cand_pred in ["state_3_first_red", "state_4_yellowred", "state_5_bothred", "state_6_yellowafter2red", "state_7_finalred", "state_8_complete"]:
                if "red_block" not in det_colors:
                    is_physically_consistent = False
            if cand_pred in ["state_4_yellowred", "state_5_bothred", "state_6_yellowafter2red", "state_7_finalred", "state_8_complete"]:
                if "yellow_block" not in det_colors:
                    is_physically_consistent = False
            if cand_pred in ["state_1_greenblue", "state_2_green2blue", "state_3_first_red"] and len(cleaned_detections) < 4:
                if "green_block" not in det_colors:
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

        # Dual-Perception: Corroborate with classifier ONLY if physically validated and not overriding an invalid check
        if cls_pred is not None and is_valid:
            if graph_eval["inferred_state"] == cls_pred:
                conf = min(0.99, max(conf, (conf + cls_conf) / 2.0))
            elif cls_pred == "state_8_complete" and current_step_index == 7:
                # Macro classifier recognizes the complete 9-part figure at Step 7: advance to Step 8!
                inferred_state = "state_8_complete"
                conf = max(conf, cls_conf)
                diagnostic = "PASS: Complete 9-part block figure verified!"
            elif cls_conf >= 0.80 and cls_pred == target_state:
                inferred_state = cls_pred
                conf = cls_conf

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
