"""
Assembly Graph & Spatial Rule Engine for Block Assembly.
Evaluates detected block bounding boxes against the expected multi-part assembly graph,
verifying both component counts and relative spatial constraints (adjacency, alignment).
Provides diagnostic error messages pinpointing which joint/block is incorrect.
"""

import numpy as np
from block_config import EXPECTED_PARTS_PER_STATE, ASSEMBLY_STATES, STEP_TITLES


def compute_iou(boxA, boxB):
    """Computes Intersection over Union between two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def are_adjacent(boxA, boxB, max_gap=90):
    """
    Checks if two bounding boxes touch or are closely adjacent (within max_gap pixels).
    """
    xA1, yA1, wA, hA = boxA
    xA2, yA2 = xA1 + wA, yA1 + hA

    xB1, yB1, wB, hB = boxB
    xB2, yB2 = xB1 + wB, yB1 + hB

    # Horizontal distance between boxes
    dx = max(0, max(xA1 - xB2, xB1 - xA2))
    # Vertical distance between boxes
    dy = max(0, max(yA1 - yB2, yB1 - yA2))

    return dx <= max_gap and dy <= max_gap


class AssemblyGraph:
    def __init__(self):
        pass

    def evaluate(self, detections, current_target_step=None):
        """
        Evaluates a list of detections against the expected assembly sequence.
        Returns:
            {
                "inferred_state": state_name,
                "confidence": float,
                "is_valid": bool,
                "diagnostic": str,
                "part_counts": {class_name: count},
                "spatial_checks": list of dicts
            }
        """
        if not detections:
            return {
                "inferred_state": "state_0_unstarted",
                "confidence": 0.0,
                "is_valid": True,
                "diagnostic": "Workspace clear (present parts to begin)",
                "part_counts": {},
                "spatial_checks": [],
            }

        # 1. Count detected parts by class
        part_counts = {}
        parts_by_class = {}
        for d in detections:
            cname = d["class_name"]
            part_counts[cname] = part_counts.get(cname, 0) + 1
            if cname not in parts_by_class:
                parts_by_class[cname] = []
            parts_by_class[cname].append(d)

        total_blocks = len(detections)

        # 2. Check each assembly stage from most advanced down to base
        inferred_state = "state_0_unstarted"
        diagnostic = "Workspace active"
        is_valid = True
        spatial_checks = []

        for state_name in reversed(ASSEMBLY_STATES):
            spec = EXPECTED_PARTS_PER_STATE.get(state_name, {})
            req_parts = spec.get("parts", {})
            min_tot = spec.get("min_total", 0)

            has_required_counts = True
            for req_cls, req_cnt in req_parts.items():
                if part_counts.get(req_cls, 0) < req_cnt:
                    has_required_counts = False
                    break

            if has_required_counts and total_blocks >= min_tot:
                inferred_state = state_name
                break

        # 3. Spatial Relationship Checks for the Inferred State
        if inferred_state == "state_1_greenblue":
            blue_boxes = parts_by_class.get("blue_block", [])
            green_boxes = parts_by_class.get("green_block", [])
            if blue_boxes and green_boxes:
                adj = are_adjacent(blue_boxes[0]["bbox"], green_boxes[0]["bbox"])
                spatial_checks.append({"rule": "Green-Blue Adjacency", "passed": adj})
                if not adj:
                    is_valid = False
                    diagnostic = "ALIGNMENT: Green beam not attached to Blue base block"
                else:
                    diagnostic = "PASS: Green beam + 1 Blue foot attached"

        elif inferred_state == "state_2_green2blue":
            blue_boxes = parts_by_class.get("blue_block", [])
            green_boxes = parts_by_class.get("green_block", [])
            if len(blue_boxes) >= 2 and green_boxes:
                adj1 = are_adjacent(blue_boxes[0]["bbox"], green_boxes[0]["bbox"])
                adj2 = are_adjacent(blue_boxes[1]["bbox"], green_boxes[0]["bbox"])
                all_attached = adj1 and adj2
                spatial_checks.append({"rule": "Both Blue Feet Attached", "passed": all_attached})
                if not all_attached:
                    is_valid = False
                    diagnostic = "LOOSE FOOT: One of the Blue feet is detached from Green beam"
                else:
                    diagnostic = "PASS: Both Blue feet securely attached (2-legged base)"

        elif inferred_state == "state_3_first_red":
            red_boxes = parts_by_class.get("red_block", [])
            green_boxes = parts_by_class.get("green_block", [])
            if red_boxes and green_boxes:
                adj = are_adjacent(red_boxes[0]["bbox"], green_boxes[0]["bbox"])
                spatial_checks.append({"rule": "Red-Green Adjacency", "passed": adj})
                if not adj:
                    is_valid = False
                    diagnostic = "ALIGNMENT: Red block not connected to Green beam"
                else:
                    diagnostic = "PASS: First Red block attached to Green beam"

        elif inferred_state == "state_4_yellowred":
            yellow_boxes = parts_by_class.get("yellow_block", [])
            red_boxes = parts_by_class.get("red_block", [])
            if yellow_boxes and red_boxes:
                adj = are_adjacent(yellow_boxes[0]["bbox"], red_boxes[0]["bbox"])
                spatial_checks.append({"rule": "Yellow-Red Joint", "passed": adj})
                if not adj:
                    is_valid = False
                    diagnostic = "ALIGNMENT: Yellow block not connected adjacent to Red block"
                else:
                    diagnostic = "PASS: First Yellow block connected next to Red block"

        elif inferred_state in ["state_5_bothred", "state_6_yellowafter2red"]:
            # Check connectivity across all parts in cluster
            connected = True
            for i in range(len(detections)):
                boxA = detections[i]["bbox"]
                has_adj = any(are_adjacent(boxA, detections[j]["bbox"]) for j in range(len(detections)) if j != i)
                if not has_adj:
                    connected = False
                    break
            spatial_checks.append({"rule": "Mid-Assembly Cluster Integrity", "passed": connected})
            if not connected:
                is_valid = False
                diagnostic = "LOOSE PART: One or more blocks are detached from the assembly"
            else:
                diagnostic = f"PASS: {STEP_TITLES.get(inferred_state, inferred_state)} verified"

        elif inferred_state in ["state_7_finalred", "state_8_complete"]:
            connected = True
            for i in range(len(detections)):
                boxA = detections[i]["bbox"]
                has_adj = any(are_adjacent(boxA, detections[j]["bbox"]) for j in range(len(detections)) if j != i)
                if not has_adj:
                    connected = False
                    break
            spatial_checks.append({"rule": "Complete Structure Integrity", "passed": connected})
            if not connected:
                is_valid = False
                diagnostic = "STRUCTURAL DEFECT: Blocks are detached or out of alignment"
            else:
                diagnostic = "PASS: Complete 9-part block figure verified!"

        # Confidence is mean confidence of participating detections
        conf = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0

        return {
            "inferred_state": inferred_state,
            "confidence": conf,
            "is_valid": is_valid,
            "diagnostic": diagnostic,
            "part_counts": part_counts,
            "spatial_checks": spatial_checks,
        }
