"""
Central configuration for Block Assembly Quality Inspection System.
Defines video mapping, block classes, assembly states, and spatial graph constraints.
Matches the physical 9-part block giraffe/animal assembly progression.
"""

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "DATASET")
EXTRACTED_DIR = os.path.join(PROJECT_ROOT, "extracted_frames")
DET_DATASET_DIR = os.path.join(PROJECT_ROOT, "yolo_dataset_det")
CLS_DATASET_DIR = os.path.join(PROJECT_ROOT, "yolo_dataset_cls")

# Block Classes for Object Detection
BLOCK_CLASSES = [
    "blue_block",    # 0 (2x2 base feet)
    "green_block",   # 1 (long rectangular beam / body spine)
    "red_block",     # 2 (2x2 body stack & head extension)
    "yellow_block",  # 3 (2x2 front connector, neck & crown)
]

BLOCK_COLORS_BGR = {
    "blue_block": (235, 130, 40),    # Cyan/Blue
    "green_block": (40, 200, 40),    # Bright Green
    "red_block": (40, 40, 235),      # Bright Red
    "yellow_block": (40, 220, 240),  # Yellow
}

# Sequential Assembly States (9 stages total, 0 to 8)
ASSEMBLY_STATES = [
    "state_0_unstarted",
    "state_1_greenblue",
    "state_2_green2blue",
    "state_3_first_red",
    "state_4_yellowred",
    "state_5_bothred",
    "state_6_yellowafter2red",
    "state_7_finalred",
    "state_8_complete",
]

# Human-readable step titles for HUD
STEP_TITLES = {
    "state_0_unstarted": "0. Unstarted / Presenting Parts",
    "state_1_greenblue": "1. Green + 1 Blue Base",
    "state_2_green2blue": "2. Green + 2 Blue Feet",
    "state_3_first_red": "3. First Red Block",
    "state_4_yellowred": "4. First Yellow Block",
    "state_5_bothred": "5. Second Red Stack",
    "state_6_yellowafter2red": "6. Second Yellow Block",
    "state_7_finalred": "7. Third Red Block",
    "state_8_complete": "8. Complete 9-Part Assembly",
}

# Mapping of dataset videos to parts or states
VIDEO_MAPPING = {
    "blue_block.mp4": {
        "type": "part",
        "target": "blue_block",
        "desc": "Single Blue Block in hand (base foot)"
    },
    "green_block.mp4": {
        "type": "part",
        "target": "green_block",
        "desc": "Single Green Beam in hand (main body spine)"
    },
    "red_block.mp4": {
        "type": "part",
        "target": "red_block",
        "desc": "Single Red Block in hand"
    },
    "yellow_block.mp4": {
        "type": "part",
        "target": "yellow_block",
        "desc": "Single Yellow Block in hand"
    },
    "state1_greenblue.mp4": {
        "type": "state",
        "target": "state_1_greenblue",
        "desc": "Step 1: Green beam attached to 1st Blue base block"
    },
    "state2_green2blue.mp4": {
        "type": "state",
        "target": "state_2_green2blue",
        "desc": "Step 2: 2nd Blue foot attached to Green beam (2-legged base)"
    },
    "state3_first_red.mp4": {
        "type": "state",
        "target": "state_3_first_red",
        "desc": "Step 3: 1st Red block attached onto Green beam"
    },
    "state4_yellowred.mp4": {
        "type": "state",
        "target": "state_4_yellowred",
        "desc": "Step 4: 1st Yellow block attached next to Red block"
    },
    "state5_bothred.mp4": {
        "type": "state",
        "target": "state_5_bothred",
        "desc": "Step 5: 2nd Red block stacked on top of 1st Red block"
    },
    "state6_yellowafter2red.mp4": {
        "type": "state",
        "target": "state_6_yellowafter2red",
        "desc": "Step 6: 2nd Yellow block attached at top of stack"
    },
    "state7_finalred.mp4": {
        "type": "state",
        "target": "state_7_finalred",
        "desc": "Step 7: 3rd Red block attached to front/head"
    },
    "state8_complete.mp4": {
        "type": "state",
        "target": "state_8_complete",
        "desc": "Step 8: Complete 9-part block animal assembly"
    },
}

# Expected parts count per state
EXPECTED_PARTS_PER_STATE = {
    "state_0_unstarted": {"min_total": 0, "parts": {}},
    "state_1_greenblue": {"min_total": 2, "parts": {"green_block": 1, "blue_block": 1}},
    "state_2_green2blue": {"min_total": 3, "parts": {"green_block": 1, "blue_block": 2}},
    "state_3_first_red": {"min_total": 4, "parts": {"green_block": 1, "blue_block": 2, "red_block": 1}},
    "state_4_yellowred": {"min_total": 5, "parts": {"green_block": 1, "blue_block": 2, "red_block": 1, "yellow_block": 1}},
    "state_5_bothred": {"min_total": 6, "parts": {"green_block": 1, "blue_block": 2, "red_block": 2, "yellow_block": 1}},
    "state_6_yellowafter2red": {"min_total": 7, "parts": {"green_block": 1, "blue_block": 2, "red_block": 2, "yellow_block": 2}},
    "state_7_finalred": {"min_total": 8, "parts": {"green_block": 1, "blue_block": 2, "red_block": 3, "yellow_block": 2}},
    "state_8_complete": {"min_total": 9, "parts": {"green_block": 1, "blue_block": 2, "red_block": 3, "yellow_block": 3}},
}

# Next required incoming part color per step
NEXT_REQUIRED_PART = {
    0: "blue_block",    # or green_block
    1: "blue_block",    # 2nd blue foot
    2: "red_block",     # 1st red block
    3: "yellow_block",  # 1st yellow block
    4: "red_block",     # 2nd red block (stacked)
    5: "yellow_block",  # 2nd yellow block
    6: "red_block",     # 3rd red block (head extension)
    7: "yellow_block",  # 3rd yellow block (final completion)
}
