# Block Assembly Quality Inspection System

An automated Computer Vision & Deep Learning quality inspection system for multi-part toy block assembly. It combines real-time **Object Detection** (localizing individual blocks and verifying parts in hand) with **Assembly Graph Spatial Verification** (checking joints, alignments, and connections), **Whole-Structure State Classification**, and **Temporal Consensus Smoothing** to strictly enforce correct assembly order and catch manufacturing/assembly defects.

---

## Architecture: Dual-Perception Perception + Spatial Graph

```
Raw Camera / Video Stream (Webcam / DroidCam / Video)
                     │
                     ▼
   ┌───────────────────────────────────┐
   │        ComponentDetector          │
   │  (Dual-Perception Mutual Engine)  │
   └───────────────────────────────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────────────────┐
│  Block Detector  │    │      Macro State Model       │
│  (YOLOv8 Detect) │    │      (YOLOv8 Classifier)     │
│  - Blue blocks   │    │  - Classifies Stage 0 to 8   │
│  - Red blocks    │    │    from whole visual layout  │
│  - Green beam    │    └──────────────────────────────┘
│  - Yellow blocks │                     │
└──────────────────┘                     │
         │                               │
         ├───────────────────────────────┤
         ▼                               ▼
┌──────────────────────────────────────────────┐
│       Assembly Graph & Part Corroboration    │
│  - Checks required part counts per stage     │
│  - Validates physical adjacency & joints     │
│  - Tracks incoming part in hand              │
│    (Alerts: "VALID" vs "WRONG PART")         │
│  - Cross-corroborates Detector + Classifier  │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│         AssemblyStateMachine                 │
│  - 10-frame Rolling Temporal Consensus      │
│  - Strict Sequential Progression             │
│  - Latches RED "SEQUENCE REJECTED" on skips  │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│           Industrial Inspection HUD          │
│  - PASS (Green) / HOLD (Slate) / REJECT (Red)│
│  - Live 9-Step Sequential Checklist          │
│  - Bounding Boxes & Incoming Object Badges   │
│  - DroidCam (USB & WiFi IP) + Webcam Support │
└──────────────────────────────────────────────┘
```

---

## Assembly Stages (9 Stages Total)

| Step | State Name | Video Source | Required Incoming Part | Cumulative Parts | Description |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **0** | `state_0_unstarted` | *(Workspace presentation)* | — | None | Workspace ready, presenting parts |
| **1** | `state_1_greenblue` | `state1_greenblue.mp4` | 🟦 **Blue** | 1 Green, 1 Blue | Green beam attached to 1st Blue base foot |
| **2** | `state_2_green2blue`| `state2_green2blue.mp4`| 🟦 **Blue** | 1 Green, 2 Blue | Second Blue foot attached (2-legged base) |
| **3** | `state_3_first_red` | `state3_first_red.mp4` | 🟥 **Red** | 1 Green, 2 Blue, 1 Red | First Red block attached onto Green beam |
| **4** | `state_4_yellowred` | `state4_yellowred.mp4` | 🟨 **Yellow** | 1 Green, 2 Blue, 1 Red, 1 Yellow | First Yellow block attached next to Red block |
| **5** | `state_5_bothred` | `state5_bothred.mp4` | 🟥 **Red** | 1 Green, 2 Blue, 2 Red, 1 Yellow | Second Red block stacked on first Red block |
| **6** | `state_6_yellowafter2red` | `state6_yellowafter2red.mp4` | 🟨 **Yellow** | 1 Green, 2 Blue, 2 Red, 2 Yellow | Second Yellow block attached at top of stack |
| **7** | `state_7_finalred` | `state7_finalred.mp4` | 🟥 **Red** | 1 Green, 2 Blue, 3 Red, 2 Yellow | Third Red block attached to front/head |
| **8** | `state_8_complete` | `state8_complete.mp4` | 🟨 **Yellow** | 1 Green, 2 Blue, 3 Red, 3 Yellow | Final Yellow block completes 9-part animal figure |

---

## How to Run

### 1. Unified Interactive Menu
```bash
python main.py
```

### 2. Step-by-Step CLI Commands

#### Step A: Extract Frames from Videos
Extracts frames from all 12 dataset videos in `DATASET/` with Laplacian blur filtering and an 80/20 train/val temporal split:
```bash
python extract_frames.py
```

#### Step B: Generate Datasets & Labels
Generates:
- `yolo_dataset_det/`: Object detection dataset with auto-generated YOLO bounding box labels
- `yolo_dataset_cls/`: State classification dataset split into stage folders
```bash
python prepare_dataset.py
```

#### Step C: Train Models
Train the YOLOv8 Object Detector:
```bash
python train_detector.py --epochs 60 --model yolov8s.pt
```

Train the YOLOv8 State Classifier:
```bash
python train_classifier.py --epochs 60 --model yolov8s-cls.pt
```

#### Step D: Run Benchmark Evaluation
Evaluates accuracy on held-out validation frames:
```bash
python evaluate.py
```

#### Step E: Live Inspection HUD
Launch the real-time HUD with webcam:
```bash
python live_demo.py
```

Or connect via external DroidCam:
```bash
# Using DroidCam via USB/PC Client:
python live_demo.py --camera 1

# Using DroidCam via direct WiFi IP URL:
python live_demo.py --camera http://<PHONE_IP>:4747/video
```

Or test on one of the recorded dataset videos:
```bash
python live_demo.py --video "DATASET/state8_complete.mp4"
```

Or inspect a single image:
```bash
python live_demo.py --image "demo_output.jpg"
```

### Live Controls:
- **`r`**: Reset sequence tracker back to Step 0.
- **`q`**: Quit the inspection window.
