# Block Assembly Quality Inspection System

An automated Computer Vision & Deep Learning quality inspection system for multi-part toy block assembly. It combines real-time **Object Detection** (localizing individual blocks and verifying incoming parts in hand) with **Green-Anchored Assembly Cropping**, **Macro State Classification**, **Temporal Consensus Smoothing**, and a **FastAPI Performance Dashboard Backend** to strictly enforce correct assembly order and catch assembly defects.

---

## Architecture: Dual-Perception + Anchor Crop Pipeline

```
Raw Camera / Video Stream (Webcam / DroidCam / Video)
                     │
                     ├─────────────────────────────────────────┐
                     ▼                                         ▼
      ┌─────────────────────────────┐           ┌─────────────────────────────┐
      │      Block Detector         │           │        Block Cropper        │
      │      (YOLOv8/11 Detect)     │           │   (Green-Anchored Filter)   │
      │  - Blue base feet           │           │  - Finds green spine anchor │
      │  - Red stack & head         │           │  - Extracts focused 384px   │
      │  - Yellow connectors/crown  │           │    assembly crop ROI        │
      │  - Tracks part in hand      │           └──────────────┬──────────────┘
      └──────────────┬──────────────┘                          │
                     │                                         ▼
                     │                          ┌─────────────────────────────┐
                     │                          │    Macro State Classifier   │
                     │                          │   (YOLOv8/11-cls @ 384px)   │
                     │                          │  - Classifies Stage 0 to 8  │
                     │                          │    from clean focused crop  │
                     │                          └──────────────┬──────────────┘
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          ▼
                         ┌─────────────────────────────────┐
                         │       ComponentDetector         │
                         │  - Dual-Perception Fusion       │
                         │  - Validates part in hand       │
                         │    ("VALID" vs "WRONG PART")    │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │      AssemblyStateMachine       │
                         │  - 10-frame Rolling Consensus   │
                         │  - Hand-Stillness Protection    │
                         │  - Enforces 0 -> 8 Sequence     │
                         └────────────────┬────────────────┘
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
┌─────────────────────────────────┐               ┌─────────────────────────────────┐
│     Industrial Inspection HUD   │               │   FastAPI Performance Dashboard │
│  - PASS / ASSEMBLING / REJECT   │   (Optional   │   - PostgreSQL / Supabase sync  │
│  - Live 9-Step Checklist        │  Background   │   - Operator cycle tracking     │
│  - Real-time FPS & Latency      │   Streaming)  │   - Real-time event analytics   │
│  - Picture-in-Picture Crop Inset│ ────────────> │   - Defect rate reporting       │
└─────────────────────────────────┘               └─────────────────────────────────┘
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

## Quick Setup

### 1. Prerequisites
- Python 3.10 to 3.12 (or 3.14)
- Webcam, DroidCam phone camera, or recorded video files

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Option A: Interactive Launcher Menu
```bash
python main.py
```

### Option B: Live Inspection HUD
Launch the real-time HUD (auto-detects working camera):
```bash
python live_demo.py
```

Connect via external USB or DroidCam:
```bash
# Using DroidCam via USB/PC Client (index 1 or 2):
python live_demo.py --camera 1

# Using DroidCam via direct WiFi IP URL:
python live_demo.py --camera http://<PHONE_IP>:4747/video
```

Test on a recorded video:
```bash
python live_demo.py --video "DATASET/state8_complete.mp4"
```

Inspect a single image:
```bash
python live_demo.py --image "demo_output.jpg"
```

#### Live HUD Controls:
- **`r` / Space**: Reset sequence tracker back to Step 0.
- **`s`**: Save an inspection snapshot and pristine raw image to `snapshots/`.
- **`q` / Esc**: Quit.

---

## Connect to the Performance Dashboard Backend

To stream live inspection events, cycle start/end, and pass/fail statistics to the FastAPI backend:

1. **Start the FastAPI backend** in a separate terminal:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

2. **Launch the Live HUD with the dashboard bridge**:
   ```bash
   python live_demo.py --dashboard http://localhost:8000 --operator OP001
   ```

---

## How to Retrain the Cropped State Classifier (GPU Laptop)

1. Generate or verify the cropped training dataset:
   ```bash
   python prepare_cropped_dataset.py
   ```

2. Train the state classifier (runs in ~6–8 mins on an RTX 4050):
   ```bash
   python train_classifier.py --epochs 60 --model yolo11s-cls.pt --batch 16
   ```

3. Benchmark validation accuracy and latency:
   ```bash
   python evaluate.py
   ```
