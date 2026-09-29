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
from block_detector import BlockDetector, YOLO_DET_MODEL_PATH
from assembly_graph import AssemblyGraph
from block_config import ASSEMBLY_STATES, STEP_TITLES, NEXT_REQUIRED_PART

YOLO_CLS_MODEL_PATH = "runs_classify/block_states/weights/best.pt"


class ComponentDetector:
    def __init__(
        self,
        det_model_path=YOLO_DET_MODEL_PATH,
        cls_model_path=YOLO_CLS_MODEL_PATH,
        conf_threshold=0.40,
    ):
        self.block_detector = BlockDetector(model_path=det_model_path, conf_threshold=conf_threshold)
        self.assembly_graph = AssemblyGraph()

        self.cls_model = None
        if os.path.exists(cls_model_path):
            try:
                from ultralytics import YOLO
                self.cls_model = YOLO(cls_model_path)
                print(f"[ComponentDetector] Loaded trained state classifier from '{cls_model_path}'")
            except Exception as e:
                print(f"[ComponentDetector] Could not load state classifier ({e}).")

    def analyze(self, img_or_path, current_step_index=0):
        """
        Analyzes a single frame:
          - Detects individual block parts and counts
          - Identifies incoming part held in hand
          - Evaluates spatial assembly graph
          - Corroborates with whole-assembly classifier
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

        # 1. Detect all blocks in frame
        detections = self.block_detector.detect(img)

        # 2. Identify incoming object (held separately in hand)
        incoming = self.block_detector.identify_incoming_object(detections)

        # If incoming part is isolated, exclude it from the base assembly graph evaluation
        assembly_detections = detections
        if incoming is not None and len(detections) > 1:
            assembly_detections = [d for d in detections if d != incoming]

        # 3. Spatial Assembly Graph Evaluation
        target_state = ASSEMBLY_STATES[current_step_index] if current_step_index < len(ASSEMBLY_STATES) else None
        graph_eval = self.assembly_graph.evaluate(assembly_detections, current_target_step=target_state)

        inferred_state = graph_eval["inferred_state"]
        conf = graph_eval["confidence"]
        is_valid = graph_eval["is_valid"]
        diagnostic = graph_eval["diagnostic"]

        # 4. Macro State Classifier Corroboration (95.1% SOTA YOLO11s)
        cls_pred = None
        cls_conf = 0.0
        if self.cls_model is not None:
            res_cls = self.cls_model.predict(img, verbose=False)[0]
            cls_pred = res_cls.names[res_cls.probs.top1]
            cls_conf = float(res_cls.probs.top1conf)

            # Prioritize the 95.1% accurate deep classifier for assembly stage
            inferred_state = cls_pred
            conf = cls_conf

            # If both agree, boost confidence
            if graph_eval["inferred_state"] == cls_pred:
                conf = min(0.99, max(conf, (conf + graph_eval["confidence"]) / 2.0))

        # 5. Incoming Object Validation against Next Required Step
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
            "part_counts": graph_eval["part_counts"],
            "spatial_checks": graph_eval["spatial_checks"],
            "signals": {
                "cls_pred": cls_pred,
                "cls_conf": cls_conf,
                "graph_state": graph_eval["inferred_state"],
            },
        }
