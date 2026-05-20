# Webcam Arm Control — Design Spec

**Date:** 2026-05-20
**Status:** Approved, ready for implementation planning
**Author:** Naufal Reky Ardhana

## Goal

Add a third way to control the existing 4-servo robot arm: by moving a hand in
front of a webcam. A new standalone Python service tracks the operator's hand,
converts its position into servo angles, and sends them to the existing robot
over the unchanged `key:value` WebSocket protocol.

Hand tracking runs **GPU-accelerated** on the target machine (NVIDIA RTX 5050
Laptop GPU, Blackwell `sm_120`, 8 GB) using an Ultralytics YOLO11-pose
hand-keypoint model on PyTorch. The tracker is pluggable behind a fixed
interface, with a MediaPipe CPU backend kept as a fallback (see *Inference
backend*).

## Context

The legacy `robot-control/` stack is treated as a **fixed interface**, not
modified:

- A Node.js WebSocket relay (`control-server/server.js`, port 3000) forwards
  `key:value` text messages between web clients and an ESP32.
- The ESP32 drives 4 servos: `base` (rotation), `updown` (arm up/down), `arm`
  (forward/back), `gripper`. Each accepts `0`–`180`; home is `90`.
- Control messages: `base:90`, `updown:120`, `arm:45`, `gripper:0`,
  `emergency:STOP`, `emergency:RESUME`.
- A client identifies as a web client by sending any non-`ESP32` message first.

The new vision service connects to this relay **as another web client** and
emits the same messages. Nothing in `robot-control/` changes.

## Research basis

Literature (via academic/scholar search, re-verified 2026-05-20 against the
primary sources) converges on a few points that shape this design:

- Single RGB webcam + a real-time hand-landmark model is the proven low-cost
  standard. MediaPipe is the reference implementation, but is **CPU-only on
  Windows** (its GPU delegate is Ubuntu-only). To use the available GPU on the
  target machine we use an Ultralytics **YOLO11-pose** hand-keypoint model,
  which outputs the same 21-landmark topology (its dataset was in fact annotated
  *with* MediaPipe) and is the model that actually benefits from the GPU. See
  *Inference backend*.
- **2D control outperforms 3D reconstruction** from a single camera — monocular
  depth is too noisy (IEEE 2024, *Real-Time Human Pose Estimation as a
  Cost-Effective Solution for the Teleoperation of a 6-Axis Cobot Arm*). Hence
  Approach B uses 2D position + apparent hand size, not 3D joint reconstruction.
  This finding is model-agnostic and holds for the YOLO tracker.
- **Smoothing (EMA/Kalman) + rate-limiting is mandatory**, or the arm jitters
  (MDPI Biomimetics 2026; IJMERR — 86 ms latency, 12.4 mm RMSE on low-cost
  servos). EMA measurably reduces frame-to-frame jitter in the literature.
- Closest reference: *Interactive Teleoperation of an Articulated Robotic Arm
  Using Vision-Based Human Hand Tracking* (MDPI Biomimetics, Vol. 11, Issue 2,
  Feb 2026) — 5 joints + gripper, single laptop webcam, calibration-based
  mapping, and the identical stabilization trio used here (short temporal
  smoothing + update-rate limiting + minimum-change thresholds), over a
  lightweight network link to an embedded controller.

## Architecture

```
webcam ──▶ OpenCV capture ──▶ YOLO11-pose hand kpts ──▶ angle mapper
                              (GPU; MediaPipe-CPU fallback)    │
                                                      smoothing + clutch + safety
                                                               │
                                                          WebSocket client
                                                               │
                                            ws://<host>:3000  (same key:value protocol)
                                                               ▼
                                    [ legacy Node relay ] ──▶ ESP32 ──▶ 4 servos
```

New code lives in a sibling directory `arm-vision/`. The legacy `robot-control/`
is untouched.

## Components

Each module has one responsibility and a clean interface so it can be tested
independently.

| Module | Responsibility | Interface |
|---|---|---|
| `capture.py` | Webcam access (OpenCV) | `read() -> frame` |
| `tracker.py` | Pluggable hand-landmark backend (YOLO11-pose GPU / MediaPipe CPU) | `process(frame) -> HandResult \| None` |
| `mapper.py` | Hand features → servo angles | `to_angles(HandResult) -> dict` |
| `smoother.py` | Deadzone + EMA + rate-limit | `smooth(raw_angles) -> angles` |
| `safety.py` | Clutch state, hold-on-loss, e-stop | state machine |
| `robot_client.py` | WebSocket client + reconnect | `send(key, value)` |
| `main.py` | Wire modules, debug preview window | run loop |

`HandResult` holds normalized landmarks (21 points, MediaPipe topology),
handedness (if available), and a confidence score. Both backends populate this
same structure, so everything downstream of `tracker.py` is backend-agnostic.

Per-frame flow: `capture → tracker → mapper → smoother → safety → robot_client`.

## Inference backend

`tracker.py` is a thin strategy chosen at startup via `--backend`:

| Backend | Model | Device | When |
|---|---|---|---|
| `yolo-gpu` (default) | Ultralytics YOLO11-pose, `hand-keypoints` (21 kpts) | CUDA (RTX 5050) | Primary path |
| `mediapipe-cpu` | MediaPipe Hands | CPU | Fallback if the GPU stack breaks |

**Why YOLO11-pose on GPU.** MediaPipe's GPU delegate is Ubuntu-only, so on
Windows it cannot use the RTX 5050. YOLO11-pose runs on PyTorch with full CUDA
support and is the model that actually benefits from the GPU (≈100+ FPS on this
card vs. slower-than-MediaPipe on CPU). Its `hand-keypoints` model emits the
same 21-landmark layout as MediaPipe, so `mapper.py` is unchanged across
backends.

**Stack + setup (Blackwell `sm_120` is bleeding-edge):**

- Stable PyTorch does **not** yet support `sm_120`. Install **PyTorch nightly
  with the cu128 (CUDA 12.8) wheel**, then `ultralytics`, in that order.
- Verify after install: `torch.cuda.is_available()` is `True` and the device
  reports the RTX 5050 — a silent CPU fallback would tank FPS unnoticed. `main`
  logs the resolved device on startup and warns if `yolo-gpu` fell back to CPU.
- Pin a known-good nightly date in the dependency file; nightly builds drift.
- First run downloads model weights.

**Accuracy knob:** start with `yolo11n-pose` (nano). If landmarks are jittery,
move to `yolo11s-pose` — the GPU has ample headroom. (Keypoint *indices* between
the Ultralytics hand topology and MediaPipe are verified during implementation;
they are expected to match since the dataset was annotated with MediaPipe.)

## Mapping (Approach B — primary: hand-position driving)

The tracker returns 21 normalized landmarks (x, y ∈ 0..1) regardless of backend.
The mapper computes:

| Servo | Source | Mapping |
|---|---|---|
| `base` | hand center X | x 0.0 → 0° (left), x 1.0 → 180° (right) |
| `updown` | hand center Y | y 0.0 → 180° (top), y 1.0 → 0° (bottom) — inverted |
| `arm` | hand size (wrist↔middle-MCP distance) | near → 180°, far → 0° |
| `gripper` | pinch (thumb-tip ↔ index-tip distance) | wide → 180° (open), together → 0° (closed) |

Design decisions baked in:

- **Control box, not full frame.** Map a centered sub-region (default: middle
  60% of the frame) to 0–180, so the operator need not reach frame edges and
  edge-distortion is avoided. Region is a config constant.
- **`arm` (forward/back via hand size) is the noisy axis** (per IEEE 2024
  finding). It gets the heaviest smoothing and is the first candidate to pin to a
  fixed value if unreliable in practice.

## Smoothing + safety

```
raw angle ─▶ deadzone (ignore <2° wiggle) ─▶ EMA (α≈0.3) ─▶ rate-limit (≤Δ8°/frame) ─▶ output
```

- **Per-servo tuning, not global.** `smoother.py` already operates per angle, so
  each servo gets its own `(α, rate-limit)`. The `arm` axis (hand size — the
  noisy one) gets heavier smoothing (α≈0.15–0.2) and a tighter rate-limit;
  `base`/`gripper` can use α≈0.3–0.4 for snappier response.
- **FPS-aware α.** A fixed α behaves differently at 100 FPS (GPU) vs 30 FPS
  (CPU fallback). Either pin the loop to a target FPS, or derive α from the
  measured frame interval `dt`, so smoothing feel is stable across backends.

- **Clutch (spacebar):** toggles engaged/idle. While idle, frames are still
  processed and previewed, but **no `key:value` is sent** — the arm holds its
  last commanded position.
- **Tracking loss** (no hand, or confidence below threshold for N consecutive
  frames): freeze — stop sending, hold last position. On hand return, re-engage
  smoothly from held angles (no snap).
- **Change-gated sends:** only transmit a servo when its smoothed value changed
  (mirrors the legacy joystick loop) — keeps the wire quiet.
- **Emergency:** a key (`q`/`esc`) sends `emergency:STOP`; another sends
  `emergency:RESUME`. Hard kill, independent of the clutch.

## Error handling

- **Camera open failure:** clear message, clean exit.
- **WebSocket down:** auto-reconnect every 3 s while tracking continues locally
  (preview still works); resume sends on reconnect. (Same 3s retry as legacy
  web client.)
- **Malformed / missing landmarks:** treated as tracking loss (freeze), never a
  crash.

## Testing

- `mapper.py`, `smoother.py`, `safety.py` are pure functions → unit-tested with
  synthetic landmark inputs (no camera or robot needed).
- `robot_client.py` → tested against a tiny mock WebSocket echo server.
- `capture.py` / `tracker.py` → manual smoke test via the live preview window.
  The tracker smoke test also asserts the resolved device (CUDA for `yolo-gpu`)
  and prints measured FPS, so a silent CPU fallback is caught immediately.
- `--dry-run` flag prints angles to console instead of sending, validating the
  full pipeline (either backend) with no ESP32 attached.

## Alternative / future mode: Approach A — Joint Mimicry

Documented for future use; **not** part of the initial implementation.

Instead of tracking only the hand, use a **full-body pose model** to track the
whole arm and have the robot copy the operator's joint angles. On this machine
that means a GPU pose backend (e.g. YOLO11-pose body keypoints, or RTMPose);
MediaPipe Pose remains an option but is CPU-only on Windows.

| Servo | Source (operator joint) |
|---|---|
| `base` | shoulder yaw (arm swung left/right) |
| `updown` | shoulder pitch (arm raised/lowered) |
| `arm` | elbow bend (open/closed) |
| `gripper` | hand pinch (open/closed) |

**Trade-offs vs. Approach B:**

- Higher "wow" factor — the robot visibly mirrors the operator's posture.
- Requires shoulder + elbow + wrist all clearly visible to the webcam.
- 3 of 4 angles depend on estimating limb bend/depth from a flat image, which is
  noisier on a single camera (the reason Approach B was chosen first).
- Slots into the existing module split by adding an alternative strategy in
  `mapper.py` (and adding a Pose backend in `tracker.py` — the same `--backend`
  seam used for YOLO/MediaPipe); smoothing, safety, and the robot client are
  unchanged. The GPU also pays off most here, where heavier pose models run.

References for this mode: IJMERR *Pose Estimation-Driven Control of Humanoid
Upper Arms*; IEEE *Real-Time Upper Body Motion Tracking* (MediaPipe Pose +
kinematics, 3-DOF, 94% accuracy).

## Out of scope

- No changes to `robot-control/` (Node relay, ESP32 firmware, web UI).
- No 3D joint reconstruction (deliberately avoided per research).
- No accuracy/RMSE instrumentation (can be added later if the goal shifts to a
  measured write-up).
