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


def are_adjacent(boxA, boxB, max_gap=28):
    """
    Checks if two bounding boxes touch or are closely adjacent (within max_gap pixels).
    Lego blocks when assembled have zero or tiny gaps.
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


def are_stacked(boxA, boxB, gbox=None, max_gap=48):
    """
    Checks if two blocks are stacked on top of each other.
    Differentiates a vertical or horizontal column stack from non-adjacent blocks.
    """
    if not are_adjacent(boxA, boxB, max_gap=max_gap):
        return False
    cA = (boxA[0] + boxA[2] // 2, boxA[1] + boxA[3] // 2)
    cB = (boxB[0] + boxB[2] // 2, boxB[1] + boxB[3] // 2)
    dx = abs(cA[0] - cB[0])
    dy = abs(cA[1] - cB[1])

    is_vert_stack = (dy >= 0.40 * min(boxA[3], boxB[3])) and (dy > dx * 0.60)
    is_horiz_stack = (dx >= 0.40 * min(boxA[2], boxB[2])) and (dx > dy * 0.60)
    return is_vert_stack or is_horiz_stack



def cluster_boxes(boxes, max_merge_dist=40):
    """
    Groups bounding boxes that overlap or are very close (like multiple stud detections on the same block)
    into distinct physical block bounding boxes.
    """
    if not boxes:
        return []
    clusters = []
    for b in boxes:
        box = b["bbox"] if isinstance(b, dict) else b
        merged = False
        for c in clusters:
            if are_adjacent(box, tuple(c), max_gap=max_merge_dist):
                x1 = min(c[0], box[0])
                y1 = min(c[1], box[1])
                x2 = max(c[0] + c[2], box[0] + box[2])
                y2 = max(c[1] + c[3], box[1] + box[3])
                c[0], c[1], c[2], c[3] = x1, y1, x2 - x1, y2 - y1
                merged = True
                break
        if not merged:
            clusters.append(list(box))
    return [tuple(c) for c in clusters]


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
        # Case 0: Empty workspace
        if not detections:
            is_clear_init = current_target_step in [None, "state_0_unstarted"]
            diag = "Workspace clear (present 1 Green beam + 1 Blue foot to begin)" if is_clear_init else "Assembly in hand / Workspace clear"
            return {
                "inferred_state": "state_0_unstarted" if is_clear_init else current_target_step,
                "confidence": 1.0,
                "is_valid": True,
                "diagnostic": diag,
                "part_counts": {},
                "spatial_checks": [],
            }

        # 1. Count detected parts by class
        part_counts = {}
        parts_by_class = {}
        for d in detections:
            cname = d["class_name"]
            part_counts[cname] = part_counts.get(cname, 0) + 1
            parts_by_class.setdefault(cname, []).append(d)

        greens = parts_by_class.get("green_block", [])
        blues = parts_by_class.get("blue_block", [])
        reds = parts_by_class.get("red_block", [])
        yellows = parts_by_class.get("yellow_block", [])
        total_blocks = len(detections)
        spatial_checks = []

        curr_idx = ASSEMBLY_STATES.index(current_target_step) if current_target_step in ASSEMBLY_STATES else None

        # --------------------------------------------------------------------------------
        # TARGET-DRIVEN SEQUENTIAL EVALUATION
        # --------------------------------------------------------------------------------
        if curr_idx is not None:
            # -------------------------------------------------------------------------
            # UNIVERSAL TOPOLOGY INVARIANT 1: YELLOW BLOCK SEPARATION
            # In the animal figure, yellow blocks serve three strictly disjoint roles:
            # Yellow 1 = Tail (at base), Yellow 2 = Neck (mid-body), Yellow 3 = Crown (head top).
            # They are physically separated by the red torso and red head.
            # NO TWO YELLOW BLOCKS EVER TOUCH EACH OTHER DIRECTLY.
            # -------------------------------------------------------------------------
            if len(yellows) >= 2:
                for i in range(len(yellows)):
                    for j in range(i + 1, len(yellows)):
                        if are_adjacent(yellows[i]["bbox"], yellows[j]["bbox"], max_gap=25):
                            return {
                                "inferred_state": current_target_step,
                                "confidence": 0.95,
                                "is_valid": False,
                                "diagnostic": "INCORRECT ASSEMBLY: Yellow blocks are connected directly together! Tail, Neck, and Crown must be separated by Red blocks.",
                                "part_counts": part_counts,
                                "spatial_checks": [{"rule": "Yellow Block Separation", "passed": False}],
                            }

            # -------------------------------------------------------------------------
            # UNIVERSAL TOPOLOGY INVARIANT 2: BASE FEET EXCLUSIVITY
            # Blue feet only serve as the base support.
            # In advanced steps (Step 6-8), the upper components (neck, head, crown)
            # must NOT be attached directly to the blue feet.
            # -------------------------------------------------------------------------
            if curr_idx >= 6 and len(blues) >= 2 and len(yellows) >= 2:
                yellows_touching_feet = [y for y in yellows if any(are_adjacent(y["bbox"], b["bbox"], max_gap=25) for b in blues)]
                if len(yellows_touching_feet) >= 2:
                    return {
                        "inferred_state": current_target_step,
                        "confidence": 0.95,
                        "is_valid": False,
                        "diagnostic": "INCORRECT ASSEMBLY: Multiple Yellow blocks attached to base feet! Only the Tail can be near the base; Neck and Crown must be at the top.",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Upper Parts Off Base Feet", "passed": False}],
                    }

            # Step 0: Starting the assembly
            if curr_idx == 0:
                if len(greens) == 0:
                    return {
                        "inferred_state": "state_0_unstarted",
                        "confidence": 0.95,
                        "is_valid": False,
                        "diagnostic": f"INCORRECT PARTS: Missing Green beam! Found {len(blues)} Blue block(s). Step 1 requires 1 Green beam + 1 Blue foot.",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Green Beam Present", "passed": False}],
                    }
                if len(reds) > 0 or len(yellows) > 0:
                    return {
                        "inferred_state": "state_0_unstarted",
                        "confidence": 0.95,
                        "is_valid": False,
                        "diagnostic": "INCORRECT PARTS: Red/Yellow block present! Step 1 requires 1 Green beam + 1 Blue foot.",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Base Stage Parts Only", "passed": False}],
                    }
                if len(blues) == 1:
                    adj = are_adjacent(blues[0]["bbox"], greens[0]["bbox"], max_gap=32)
                    spatial_checks.append({"rule": "Green-Blue Adjacency", "passed": adj})
                    if adj:
                        return {
                            "inferred_state": "state_1_greenblue",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 1 Complete (Green beam + 1 Blue foot attached)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_1_greenblue",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: Please attach Blue foot to Green beam",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                elif len(blues) >= 2:
                    # User directly presents 2-legged base
                    gbox = greens[0]["bbox"]
                    att = [b for b in blues if are_adjacent(b["bbox"], gbox, max_gap=35)]
                    loose = [b for b in blues if b not in att]

                    is_horiz = gbox[2] >= gbox[3]
                    beam_len = max(gbox[2], gbox[3])
                    coords = [b["bbox"][0] + b["bbox"][2] // 2 if is_horiz else b["bbox"][1] + b["bbox"][3] // 2 for b in att]
                    span = max(coords) - min(coords) if coords else 0

                    both_feet_attached = (len(att) >= 2) and (len(loose) == 0) and (span >= 0.35 * beam_len)
                    spatial_checks.append({"rule": "Both Blue Feet Attached", "passed": both_feet_attached})
                    if both_feet_attached:
                        return {
                            "inferred_state": "state_2_green2blue",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 2 Complete (Both Blue feet attached - 2-legged base)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    elif len(loose) > 0 or len(att) == 1 or span < 0.35 * beam_len:
                        return {
                            "inferred_state": "state_2_green2blue",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: Please attach 2nd Blue foot to Green beam",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_2_green2blue",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: Please attach Blue feet to Green beam",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                else:
                    return {
                        "inferred_state": "state_1_greenblue",
                        "confidence": 0.90,
                        "is_valid": False,
                        "diagnostic": "ASSEMBLING: Green beam detected. Please introduce 1st Blue foot.",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Blue Foot Present", "passed": False}],
                    }

            # Step 1: Green + 1 Blue base verified, waiting for 2nd Blue foot (Step 2)
            elif curr_idx == 1:
                if len(reds) > 0 or len(yellows) > 0:
                    return {
                        "inferred_state": "state_1_greenblue",
                        "confidence": 0.92,
                        "is_valid": False,
                        "diagnostic": "INCORRECT PART: Red/Yellow block introduced! Step 2 requires 2nd Blue foot.",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Correct Next Part (Blue foot)", "passed": False}],
                    }
                if len(greens) == 0:
                    return {
                        "inferred_state": "state_1_greenblue",
                        "confidence": 0.90,
                        "is_valid": True,
                        "diagnostic": "Assembly in hand / Workspace clear",
                        "part_counts": part_counts,
                        "spatial_checks": [],
                    }
                gbox = greens[0]["bbox"]
                if len(blues) == 1:
                    # Holding Step 1
                    adj = are_adjacent(blues[0]["bbox"], gbox, max_gap=32)
                    return {
                        "inferred_state": "state_1_greenblue",
                        "confidence": 0.95,
                        "is_valid": adj,
                        "diagnostic": "PASS: Step 1 Complete (Green beam + 1 Blue foot attached)" if adj else "ASSEMBLING: Please attach Blue foot to Green beam",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Green-Blue Adjacency", "passed": adj}],
                    }
                elif len(blues) >= 2:
                    # Step 2 candidate
                    att = [b for b in blues if are_adjacent(b["bbox"], gbox, max_gap=35)]
                    loose = [b for b in blues if b not in att]

                    is_horiz = gbox[2] >= gbox[3]
                    beam_len = max(gbox[2], gbox[3])
                    coords = [b["bbox"][0] + b["bbox"][2] // 2 if is_horiz else b["bbox"][1] + b["bbox"][3] // 2 for b in att]
                    span = max(coords) - min(coords) if coords else 0

                    # Check same-side alignment: both feet must project in the SAME direction from the green beam
                    gc = (gbox[0] + gbox[2] // 2, gbox[1] + gbox[3] // 2)
                    same_side = True
                    if len(att) >= 2:
                        b1_center = (att[0]["bbox"][0] + att[0]["bbox"][2] // 2, att[0]["bbox"][1] + att[0]["bbox"][3] // 2)
                        b2_center = (att[1]["bbox"][0] + att[1]["bbox"][2] // 2, att[1]["bbox"][1] + att[1]["bbox"][3] // 2)
                        d1 = b1_center[1] - gc[1] if is_horiz else b1_center[0] - gc[0]
                        d2 = b2_center[1] - gc[1] if is_horiz else b2_center[0] - gc[0]
                        if (d1 > 15 and d2 < -15) or (d1 < -15 and d2 > 15):
                            same_side = False

                    feet_separated = span >= 0.35 * beam_len
                    both_feet_attached = (len(att) >= 2) and (len(loose) == 0) and feet_separated and same_side
                    spatial_checks.append({"rule": "Both Blue Feet Attached at Ends", "passed": (len(att) >= 2) and feet_separated})
                    spatial_checks.append({"rule": "Feet on Same Side of Beam", "passed": same_side})

                    if not same_side:
                        return {
                            "inferred_state": "state_2_green2blue",
                            "confidence": 0.92,
                            "is_valid": False,
                            "diagnostic": "INCORRECT ASSEMBLY: Both Blue feet must be on the SAME side of the Green beam to form legs (not opposite sides)!",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    elif both_feet_attached:
                        return {
                            "inferred_state": "state_2_green2blue",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 2 Complete (Both Blue feet attached - 2-legged base)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_2_green2blue",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: Please attach 2nd Blue foot to Green beam",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                else:
                    return {
                        "inferred_state": "state_1_greenblue",
                        "confidence": 0.90,
                        "is_valid": False,
                        "diagnostic": "ASSEMBLING: Please attach Blue foot to Green beam",
                        "part_counts": part_counts,
                        "spatial_checks": [],
                    }

            # Step 2: Green + 2 Blue feet verified (2-legged base), waiting for 1st Red block (Step 3)
            elif curr_idx == 2:
                if len(yellows) > 0:
                    return {
                        "inferred_state": "state_2_green2blue",
                        "confidence": 0.92,
                        "is_valid": False,
                        "diagnostic": "INCORRECT PART: Yellow block introduced! Step 3 requires 1st Red block.",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Correct Next Part (Red block)", "passed": False}],
                    }
                if len(reds) == 0:
                    # Holding Step 2
                    gbox = greens[0]["bbox"] if greens else None
                    att = [b for b in blues if are_adjacent(b["bbox"], gbox, max_gap=32)] if gbox else []
                    same_side = True
                    if len(att) >= 2 and gbox:
                        is_horiz = gbox[2] >= gbox[3]
                        gc = (gbox[0] + gbox[2] // 2, gbox[1] + gbox[3] // 2)
                        b1_center = (att[0]["bbox"][0] + att[0]["bbox"][2] // 2, att[0]["bbox"][1] + att[0]["bbox"][3] // 2)
                        b2_center = (att[1]["bbox"][0] + att[1]["bbox"][2] // 2, att[1]["bbox"][1] + att[1]["bbox"][3] // 2)
                        d1 = b1_center[1] - gc[1] if is_horiz else b1_center[0] - gc[0]
                        d2 = b2_center[1] - gc[1] if is_horiz else b2_center[0] - gc[0]
                        if (d1 > 15 and d2 < -15) or (d1 < -15 and d2 > 15):
                            same_side = False
                    is_ok = len(att) >= 2 and same_side if greens else True
                    if not same_side:
                        diag = "INCORRECT ASSEMBLY: Both Blue feet must be on the SAME side of the Green beam to form legs!"
                    elif is_ok:
                        diag = "PASS: Step 2 Complete (Both Blue feet attached - 2-legged base)"
                    else:
                        diag = "ASSEMBLING: Please attach Blue feet to Green beam"
                    return {
                        "inferred_state": "state_2_green2blue",
                        "confidence": 0.95,
                        "is_valid": is_ok,
                        "diagnostic": diag,
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "2-Legged Base Intact", "passed": is_ok}],
                    }
                else:
                    # Step 3 candidate: 1st Red block introduced
                    gbox = greens[0]["bbox"] if greens else None
                    red_att = any(are_adjacent(r["bbox"], gbox, max_gap=35) for r in reds) if gbox else False
                    spatial_checks.append({"rule": "Red Block Attached to Green Beam", "passed": red_att})
                    if red_att:
                        return {
                            "inferred_state": "state_3_first_red",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 3 Complete (First Red block attached to Green beam)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_3_first_red",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: Please attach Red block onto Green beam",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }

            # Step 3: First Red verified, waiting for 1st Yellow block (Step 4)
            elif curr_idx == 3:
                # Check if there are distinct separate red blocks (not duplicate detections on the same red block)
                distinct_reds = False
                if len(reds) >= 2:
                    for i in range(len(reds)):
                        for j in range(i + 1, len(reds)):
                            r1, r2 = reds[i]["bbox"], reds[j]["bbox"]
                            c1 = (r1[0] + r1[2] // 2, r1[1] + r1[3] // 2)
                            c2 = (r2[0] + r2[2] // 2, r2[1] + r2[3] // 2)
                            dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                            ux1 = min(r1[0], r2[0])
                            uy1 = min(r1[1], r2[1])
                            ux2 = max(r1[0] + r1[2], r2[0] + r2[2])
                            uy2 = max(r1[1] + r1[3], r2[1] + r2[3])
                            union_w, union_h = ux2 - ux1, uy2 - uy1
                            if dist > 80 or union_w > 220 or union_h > 220:
                                distinct_reds = True
                                break

                if distinct_reds and len(yellows) == 0:
                    return {
                        "inferred_state": "state_3_first_red",
                        "confidence": 0.92,
                        "is_valid": False,
                        "diagnostic": "INCORRECT ORDER: Second Red introduced before First Yellow! Step 4 requires Yellow block.",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Correct Next Part (Yellow block)", "passed": False}],
                    }
                if len(yellows) == 0:
                    # Holding Step 3: Must strictly verify that at least 1 Red block is attached to Green beam!
                    if len(reds) == 0:
                        return {
                            "inferred_state": "state_2_green2blue",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: Please attach 1st Red block onto Green beam",
                            "part_counts": part_counts,
                            "spatial_checks": [{"rule": "First Red Attached", "passed": False}],
                        }
                    gbox = greens[0]["bbox"] if greens else None
                    red_att = any(are_adjacent(r["bbox"], gbox, max_gap=35) for r in reds) if gbox else False
                    if red_att:
                        return {
                            "inferred_state": "state_3_first_red",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 3 Complete (First Red block attached to Green beam)",
                            "part_counts": part_counts,
                            "spatial_checks": [{"rule": "First Red Attached", "passed": True}],
                        }
                    else:
                        return {
                            "inferred_state": "state_3_first_red",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: Please attach Red block onto Green beam",
                            "part_counts": part_counts,
                            "spatial_checks": [{"rule": "First Red Attached", "passed": False}],
                        }
                else:
                    # Step 4 candidate: First Yellow block introduced (Tail)
                    touches_red = any(are_adjacent(y["bbox"], r["bbox"], max_gap=38) for y in yellows for r in reds)
                    touches_green = any(are_adjacent(y["bbox"], g["bbox"], max_gap=38) for y in yellows for g in greens) if greens else True
                    spatial_checks.append({"rule": "Yellow Tail Joint", "passed": touches_red or touches_green})
                    if touches_red or touches_green:
                        return {
                            "inferred_state": "state_4_yellowred",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 4 Complete (First Yellow block connected next to Red)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_4_yellowred",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "INCORRECT ATTACHMENT: First Yellow block (tail) must be attached to Green beam next to Red block",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }

            # Step 4: First Yellow connected, waiting for 2nd Red block (stacked) (Step 5)
            elif curr_idx == 4:
                if len(yellows) >= 2 and len(reds) < 2:
                    return {
                        "inferred_state": "state_4_yellowred",
                        "confidence": 0.92,
                        "is_valid": False,
                        "diagnostic": "INCORRECT ORDER: Second Yellow introduced before Second Red! Step 5 requires 2nd Red block.",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Correct Next Part (2nd Red block)", "passed": False}],
                    }
                if len(reds) < 2:
                    # Holding Step 4: Verify 1 Red and 1 Yellow are attached
                    if len(reds) == 0 or len(yellows) == 0:
                        return {
                            "inferred_state": "state_3_first_red" if len(reds) > 0 else "state_2_green2blue",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: Please attach First Yellow block (tail) next to Red block",
                            "part_counts": part_counts,
                            "spatial_checks": [{"rule": "Yellow Tail Attached", "passed": False}],
                        }
                    touches_red = any(are_adjacent(y["bbox"], r["bbox"], max_gap=38) for y in yellows for r in reds)
                    touches_green = any(are_adjacent(y["bbox"], g["bbox"], max_gap=38) for y in yellows for g in greens) if greens else True
                    if touches_red or touches_green:
                        return {
                            "inferred_state": "state_4_yellowred",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 4 Complete (First Yellow block connected next to Red)",
                            "part_counts": part_counts,
                            "spatial_checks": [{"rule": "Yellow-Red Joint", "passed": True}],
                        }
                    else:
                        return {
                            "inferred_state": "state_4_yellowred",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "INCORRECT ATTACHMENT: First Yellow block (tail) must be attached to Green beam next to Red block",
                            "part_counts": part_counts,
                            "spatial_checks": [{"rule": "Yellow-Red Joint", "passed": False}],
                        }
                else:
                    # Step 5 candidate: Second Red block stacked (Torso)
                    gbox = greens[0]["bbox"] if greens else None
                    reds_stacked = any(are_stacked(reds[i]["bbox"], reds[j]["bbox"], gbox=gbox, max_gap=48) for i in range(len(reds)) for j in range(i + 1, len(reds))) if len(reds) >= 2 else False
                    spatial_checks.append({"rule": "Red Torso Stacked", "passed": reds_stacked})
                    if reds_stacked:
                        return {
                            "inferred_state": "state_5_bothred",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 5 Complete (Both Red blocks stacked)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_5_bothred",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "INCORRECT ATTACHMENT: Second Red block must be stacked directly on top of First Red block (forming the torso)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }

            # Step 5: Both Reds stacked, waiting for 2nd Yellow block (Step 6)
            elif curr_idx == 5:
                full_reds = [r for r in reds if r.get("area", r["bbox"][2] * r["bbox"][3]) >= 3500]
                if len(full_reds) >= 3 and len(yellows) < 2:
                    return {
                        "inferred_state": "state_5_bothred",
                        "confidence": 0.92,
                        "is_valid": False,
                        "diagnostic": "INCORRECT ORDER: Third Red introduced before Second Yellow! Step 6 requires Yellow block.",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Correct Next Part (2nd Yellow block)", "passed": False}],
                    }
                if len(yellows) < 2:
                    # Holding Step 5: Verify red stack is intact and 1 yellow is present
                    if len(reds) < 2 or len(yellows) < 1:
                        return {
                            "inferred_state": "state_4_yellowred",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: Please stack Second Red block onto First Red block (torso)",
                            "part_counts": part_counts,
                            "spatial_checks": [{"rule": "Red Torso Stacked", "passed": False}],
                        }
                    gbox = greens[0]["bbox"] if greens else None
                    reds_stacked = any(are_stacked(reds[i]["bbox"], reds[j]["bbox"], gbox=gbox, max_gap=48) for i in range(len(reds)) for j in range(i + 1, len(reds))) if len(reds) >= 2 else False
                    spatial_checks.append({"rule": "Red Torso Stacked", "passed": reds_stacked})
                    if reds_stacked:
                        return {
                            "inferred_state": "state_5_bothred",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 5 Complete (Both Red blocks stacked)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_5_bothred",
                            "confidence": 0.90,

                            "is_valid": False,
                            "diagnostic": "INCORRECT ATTACHMENT: Second Red block must be stacked directly on top of First Red block (forming the torso)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                else:
                    # Step 6 candidate: Second Yellow block attached to neck
                    neck_attached = any(are_adjacent(y["bbox"], r["bbox"], max_gap=48) for y in yellows for r in reds)
                    spatial_checks.append({"rule": "Yellow Neck Attached to Torso", "passed": neck_attached})
                    if neck_attached:
                        return {
                            "inferred_state": "state_6_yellowafter2red",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 6 Complete (Second Yellow block attached to neck)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_6_yellowafter2red",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "INCORRECT ATTACHMENT: Second Yellow block must be attached to the top of the Red torso stack to form the neck",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }

            # Step 6: Second Yellow attached, waiting for 3rd Red block (Step 7)
            elif curr_idx == 6:
                if len(reds) < 3:
                    # Holding Step 6: Verify neck is attached to torso and parts are present
                    if len(reds) < 2 or len(yellows) < 2:
                        return {
                            "inferred_state": "state_5_bothred",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: Please attach Second Yellow block to top of torso (neck)",
                            "part_counts": part_counts,
                            "spatial_checks": [{"rule": "Neck Assembly Intact", "passed": False}],
                        }
                    neck_attached = any(are_adjacent(y["bbox"], r["bbox"], max_gap=48) for y in yellows for r in reds)
                    spatial_checks.append({"rule": "Neck Assembly Intact", "passed": neck_attached})
                    if neck_attached:
                        return {
                            "inferred_state": "state_6_yellowafter2red",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 6 Complete (Second Yellow block attached to neck)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_6_yellowafter2red",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "INCORRECT ATTACHMENT: Second Yellow block must be attached to the top of the Red torso stack to form the neck",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                else:
                    # Step 7 candidate: 3rd Red attached to head
                    # The neck yellow block (farthest from feet) must connect to BOTH torso red and head red (>= 2 red connections)
                    if len(blues) >= 1 and len(yellows) >= 2:
                        feet_center = np.mean([(b["bbox"][0] + b["bbox"][2] // 2, b["bbox"][1] + b["bbox"][3] // 2) for b in blues], axis=0)
                        sorted_y = sorted(yellows, key=lambda y: np.hypot(y["bbox"][0] + y["bbox"][2] // 2 - feet_center[0],
                                                                          y["bbox"][1] + y["bbox"][3] // 2 - feet_center[1]))
                        neck_yellow = sorted_y[-1]
                        reds_on_neck = [r for r in reds if are_adjacent(neck_yellow["bbox"], r["bbox"], max_gap=50)]
                        head_attached = len(reds_on_neck) >= 2
                    else:
                        head_attached = any(len([r for r in reds if are_adjacent(y["bbox"], r["bbox"], max_gap=50)]) >= 2 for y in yellows)

                    spatial_checks.append({"rule": "Red Head Attached to Neck", "passed": head_attached})
                    if head_attached:
                        return {
                            "inferred_state": "state_7_finalred",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 7 Complete (Third Red block attached to head)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_7_finalred",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "INCORRECT ATTACHMENT: Third Red block must be attached to the Yellow neck to form the head",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }

            # Step 7: Third Red attached, waiting for 3rd Yellow block (Step 8 - Complete)
            elif curr_idx == 7:
                is_step8 = (len(yellows) >= 3)

                if not is_step8:
                    # Holding Step 7: Verify head is attached to neck
                    if len(reds) < 3 or len(yellows) < 2:
                        return {
                            "inferred_state": "state_6_yellowafter2red",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: Please attach Third Red block to neck (head)",
                            "part_counts": part_counts,
                            "spatial_checks": [{"rule": "Head Assembly Intact", "passed": False}],
                        }
                    if len(blues) >= 1 and len(yellows) >= 2:
                        feet_center = np.mean([(b["bbox"][0] + b["bbox"][2] // 2, b["bbox"][1] + b["bbox"][3] // 2) for b in blues], axis=0)
                        sorted_y = sorted(yellows, key=lambda y: np.hypot(y["bbox"][0] + y["bbox"][2] // 2 - feet_center[0],
                                                                          y["bbox"][1] + y["bbox"][3] // 2 - feet_center[1]))
                        neck_yellow = sorted_y[-1]
                        reds_on_neck = [r for r in reds if are_adjacent(neck_yellow["bbox"], r["bbox"], max_gap=50)]
                        head_attached = len(reds_on_neck) >= 2
                    else:
                        head_attached = any(len([r for r in reds if are_adjacent(y["bbox"], r["bbox"], max_gap=50)]) >= 2 for y in yellows)
                    spatial_checks.append({"rule": "Head Assembly Intact", "passed": head_attached})
                    if head_attached:
                        return {
                            "inferred_state": "state_7_finalred",
                            "confidence": 0.95,
                            "is_valid": True,
                            "diagnostic": "PASS: Step 7 Complete (Third Red block attached to head)",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_7_finalred",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "INCORRECT ATTACHMENT: Third Red block must be attached to the Yellow neck to form the head",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                else:
                    # Step 8 candidate: Complete animal figure
                    crown_attached = all(any(are_adjacent(y["bbox"], r["bbox"], max_gap=50) for r in reds) for y in yellows)
                    connected = all(
                        any(are_adjacent(d["bbox"], other["bbox"], max_gap=50) for other in detections if other != d)
                        for d in detections
                    )
                    spatial_checks.append({"rule": "Crown Attached to Head", "passed": crown_attached})
                    spatial_checks.append({"rule": "All Parts Connected", "passed": connected})
                    if crown_attached and connected:
                        return {
                            "inferred_state": "state_8_complete",
                            "confidence": 0.98,
                            "is_valid": True,
                            "diagnostic": "PASS: Complete 9-part block figure verified!",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    elif not crown_attached:
                        return {
                            "inferred_state": "state_8_complete",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "INCORRECT ATTACHMENT: Final Yellow block must be attached to the Red head to form the crown",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }
                    else:
                        return {
                            "inferred_state": "state_8_complete",
                            "confidence": 0.90,
                            "is_valid": False,
                            "diagnostic": "ASSEMBLING: One or more blocks are detached from assembly",
                            "part_counts": part_counts,
                            "spatial_checks": spatial_checks,
                        }

            # Step 8: Complete
            elif curr_idx == 8:
                crown_attached = all(any(are_adjacent(y["bbox"], r["bbox"], max_gap=50) for r in reds) for y in yellows)
                connected = all(
                    any(are_adjacent(d["bbox"], other["bbox"], max_gap=50) for other in detections if other != d)
                    for d in detections
                )
                if crown_attached and connected:
                    return {
                        "inferred_state": "state_8_complete",
                        "confidence": 0.98,
                        "is_valid": True,
                        "diagnostic": "PASS: Complete 9-part block figure verified!",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Figure Complete", "passed": True}],
                    }
                elif not crown_attached:
                    return {
                        "inferred_state": "state_8_complete",
                        "confidence": 0.90,
                        "is_valid": False,
                        "diagnostic": "INCORRECT ATTACHMENT: Final Yellow block must be attached to the Red head to form the crown",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Figure Complete", "passed": False}],
                    }
                else:
                    return {
                        "inferred_state": "state_8_complete",
                        "confidence": 0.90,
                        "is_valid": False,
                        "diagnostic": "ASSEMBLING: One or more blocks are detached from assembly",
                        "part_counts": part_counts,
                        "spatial_checks": [{"rule": "Figure Complete", "passed": False}],
                    }

        # --------------------------------------------------------------------------------
        # FALLBACK: REVERSE MATCH WHEN TARGET STEP IS NOT SPECIFIED
        # --------------------------------------------------------------------------------
        inferred_state = "state_0_unstarted"
        diagnostic = "Workspace active"
        is_valid = True

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

        # Spatial check on inferred state
        if inferred_state == "state_1_greenblue":
            adj = any(are_adjacent(b["bbox"], greens[0]["bbox"], max_gap=32) for b in blues) if greens and blues else False
            is_valid = adj
            diagnostic = "PASS: Green beam + 1 Blue foot attached" if adj else "ASSEMBLING: Attach Blue foot to Green beam"
        elif inferred_state == "state_2_green2blue":
            att = [b for b in blues if are_adjacent(b["bbox"], greens[0]["bbox"], max_gap=32)] if greens else []
            same_side = True
            if len(att) >= 2 and greens:
                gbox = greens[0]["bbox"]
                is_horiz = gbox[2] >= gbox[3]
                gc = (gbox[0] + gbox[2] // 2, gbox[1] + gbox[3] // 2)
                b1_center = (att[0]["bbox"][0] + att[0]["bbox"][2] // 2, att[0]["bbox"][1] + att[0]["bbox"][3] // 2)
                b2_center = (att[1]["bbox"][0] + att[1]["bbox"][2] // 2, att[1]["bbox"][1] + att[1]["bbox"][3] // 2)
                d1 = b1_center[1] - gc[1] if is_horiz else b1_center[0] - gc[0]
                d2 = b2_center[1] - gc[1] if is_horiz else b2_center[0] - gc[0]
                if (d1 > 15 and d2 < -15) or (d1 < -15 and d2 > 15):
                    same_side = False
            is_valid = len(att) >= 2 and same_side
            if not same_side:
                diagnostic = "INCORRECT ASSEMBLY: Both Blue feet must be on the SAME side of the Green beam to form legs!"
            elif is_valid:
                diagnostic = "PASS: Both Blue feet securely attached (2-legged base)"
            else:
                diagnostic = "ASSEMBLING: Please attach 2nd Blue foot to Green beam"
        elif inferred_state == "state_3_first_red":
            is_valid = any(are_adjacent(r["bbox"], greens[0]["bbox"], max_gap=35) for r in reds) if greens else True
            diagnostic = "PASS: First Red block attached to Green beam" if is_valid else "ASSEMBLING: Please attach Red block onto Green beam"
        elif inferred_state == "state_4_yellowred":
            touches_red = any(are_adjacent(y["bbox"], r["bbox"], max_gap=38) for y in yellows for r in reds)
            is_valid = touches_red
            diagnostic = "PASS: First Yellow block connected next to Red block" if is_valid else "INCORRECT ATTACHMENT: First Yellow block must be attached next to Red block"
        elif inferred_state == "state_5_bothred":
            gbox = greens[0]["bbox"] if greens else None
            reds_stacked = are_stacked(reds[0]["bbox"], reds[1]["bbox"], gbox=gbox, max_gap=48) if len(reds) >= 2 else False
            is_valid = reds_stacked
            diagnostic = "PASS: Both Red blocks stacked" if is_valid else "INCORRECT ATTACHMENT: Second Red block must be stacked on First Red block"
        elif inferred_state == "state_6_yellowafter2red":
            yellows_separate = not (len(yellows) >= 2 and are_adjacent(yellows[0]["bbox"], yellows[1]["bbox"], max_gap=25))
            neck_attached = any(are_adjacent(y["bbox"], r["bbox"], max_gap=48) for y in yellows for r in reds)
            is_valid = yellows_separate and neck_attached
            diagnostic = "PASS: Second Yellow block attached to neck" if is_valid else "INCORRECT ASSEMBLY: Neck must attach to Red torso and Yellow blocks must remain separate"
        elif inferred_state == "state_7_finalred":
            yellows_separate = not (len(yellows) >= 2 and are_adjacent(yellows[0]["bbox"], yellows[1]["bbox"], max_gap=25))
            head_attached = any(are_adjacent(r["bbox"], y["bbox"], max_gap=50) for r in reds for y in yellows)
            is_valid = yellows_separate and head_attached
            diagnostic = "PASS: Third Red block attached to head" if is_valid else "INCORRECT ASSEMBLY: Head must attach to Yellow neck and Yellow blocks must remain separate"
        elif inferred_state == "state_8_complete":
            yellows_separate = True
            if len(yellows) >= 2:
                for i in range(len(yellows)):
                    for j in range(i + 1, len(yellows)):
                        if are_adjacent(yellows[i]["bbox"], yellows[j]["bbox"], max_gap=25):
                            yellows_separate = False
                            break
            crown_attached = all(any(are_adjacent(y["bbox"], r["bbox"], max_gap=50) for r in reds) for y in yellows)
            connected = all(any(are_adjacent(d["bbox"], other["bbox"], max_gap=50) for other in detections if other != d) for d in detections)
            is_valid = yellows_separate and crown_attached and connected
            diagnostic = "PASS: Complete 9-part block figure verified!" if is_valid else "INCORRECT ASSEMBLY: Figure parts are not attached in the correct animal geometry"

        conf = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0
        return {
            "inferred_state": inferred_state,
            "confidence": conf,
            "is_valid": is_valid,
            "diagnostic": diagnostic,
            "part_counts": part_counts,
            "spatial_checks": spatial_checks,
        }
