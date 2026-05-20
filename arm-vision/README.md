# arm-vision — Webcam Hand-Control for the Robot Arm

A standalone service that tracks a hand via webcam and drives the existing
4-servo robot arm over the legacy `key:value` WebSocket protocol.  It connects
to `ws://<host>:3000/` as a plain WebSocket client; the `robot-control/`
server is untouched.

---

## Requirements

| Item | Value |
|------|-------|
| Python | 3.13 (shared venv `C:\Users\Lenovo\torch-env`) |
| PyTorch | 2.8.0+cu129 — **already installed**, supports RTX 5050 / Blackwell sm_120 |
| Other deps | `ultralytics`, `opencv-python`, `websocket-client`, `mediapipe`, `pytest`, `websockets` |

Install the non-torch deps (torch is already present — do **not** reinstall it):

```
C:\Users\Lenovo\torch-env\Scripts\python.exe -m pip install -r requirements.txt
```

---

## GPU Smoke Test

Run this first.  It verifies torch sees CUDA, the RTX 5050 is reachable at
sm_120, and a YOLO inference pass completes on CUDA:

```
C:\Users\Lenovo\torch-env\Scripts\python.exe smoke_gpu.py
```

Expected last line: `PASS: GPU stack is ready.`

---

## Quick Start (no robot, no trained model needed)

Validate the full pipeline today on CPU with the MediaPipe backend:

```
cd arm-vision
C:\Users\Lenovo\torch-env\Scripts\python.exe -m armvision.main --backend mediapipe-cpu --dry-run
```

Move your hand in front of the webcam, press **SPACE** to engage, and watch
`base:NN  updown:NN  arm:NN  gripper:NN` print to the terminal.  Press **q**
to quit.

---

## Tracker Backends

Select with `--backend {yolo-gpu,mediapipe-cpu}` (default: `yolo-gpu`).

### `mediapipe-cpu` — works now

Uses the MediaPipe Tasks `HandLandmarker` (the legacy `mp.solutions` API is
unavailable on Python 3.13).  Auto-downloads `hand_landmarker.task` (~7 MB)
on first run.  Provides real 21-keypoint hand tracking on CPU.

### `yolo-gpu` — requires a trained hand model

Runs YOLO11-pose on CUDA for maximum throughput.  **There is no official
pretrained YOLO hand-pose model.**  The file `yolo11n-pose.pt` already present
is a 17-keypoint *body* model; it will not detect hands.  Use `--backend
mediapipe-cpu` until a hand model is trained (see below).

---

## Training the YOLO Hand Model

```python
from ultralytics import YOLO
YOLO("yolo11n-pose.pt").train(
    data="hand-keypoints.yaml",
    epochs=80,
    imgsz=640,
    device=0,
)
```

Or from the command line:

```
yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=80 imgsz=640 device=0
```

- Downloads the 26,768-image hand-keypoints dataset automatically.
- When training completes, set `YOLO_HAND_MODEL` in `armvision/tracker.py` to
  the resulting `runs/hand-pose/weights/best.pt`.
- The model must output 21 keypoints in MediaPipe order; remap indices in
  `YoloTracker.process` if the dataset uses a different ordering.

---

## CLI Reference

```
python -m armvision.main [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--backend {yolo-gpu,mediapipe-cpu}` | `yolo-gpu` | Tracker backend |
| `--camera N` | `0` | Camera index (OpenCV) |
| `--ws URL` | `ws://localhost:3000/` | WebSocket URL of the robot server |
| `--dry-run` | off | Print angles to stdout instead of sending; no robot needed |

---

## Key Bindings (preview window)

| Key | Action |
|-----|--------|
| SPACE | Toggle clutch — engage (sending) / idle (paused) |
| r | Emergency RESUME |
| q or ESC | Emergency STOP + quit |

Nothing is sent to the robot while the clutch is idle.

---

## Control Mapping (hand → servos)

| Servo | Hand feature | Range |
|-------|-------------|-------|
| `base` | Hand center X | left→0°, right→180° |
| `updown` | Hand center Y (inverted) | top→180°, bottom→0° |
| `arm` | Apparent hand size (wrist↔middle-MCP distance) | far→0°, near→180° |
| `gripper` | Thumb-index pinch distance (scale-invariant) | together→0° (closed), apart→180° (open) |

Only the centered ~60% of the frame ("control box") maps to the full 0–180°
range, so the operator does not need to reach the frame edges.

---

## Running the Tests

From the `arm-vision` directory:

```
C:\Users\Lenovo\torch-env\Scripts\python.exe -m pytest -v
```

Expected: **21 passed** (mapper 6, smoother 6, safety 6, robot_client 3).
The suite is CPU-only and does not require a camera or a running robot server.
