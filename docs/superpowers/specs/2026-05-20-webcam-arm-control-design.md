# Webcam Arm Control — Design Spec

**Date:** 2026-05-20
**Status:** Approved, ready for implementation planning
**Author:** Naufal Reky Ardhana

## Goal

Add a third way to control the existing 4-servo robot arm: by moving a hand in
front of a webcam. A new standalone Python service tracks the operator's hand,
converts its position into servo angles, and sends them to the existing robot
over the unchanged `key:value` WebSocket protocol.

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

Literature (via Perplexity academic/scholar search) converges on a few points
that shape this design:

- Single RGB webcam + **MediaPipe** is the proven low-cost standard
  (runs on an average laptop CPU, no GPU).
- **2D control outperforms 3D reconstruction** from a single camera — monocular
  depth is too noisy (IEEE 2024, "Cost-Effective Solution for Teleoperation of a
  6-Axis Cobot"). Hence Approach B uses 2D position + apparent hand size, not 3D
  joint reconstruction.
- **Smoothing (EMA/Kalman) + rate-limiting is mandatory**, or the arm jitters
  (MDPI Biomimetics 2025; IJMERR — 86 ms latency, 12.4 mm RMSE on low-cost
  servos).
- Closest reference: *Interactive Teleoperation of an Articulated Robotic Arm
  Using Vision-Based Human Hand Tracking* (MDPI Biomimetics 2025) — 5 joints +
  gripper, single laptop webcam, calibration-based mapping, temporal smoothing,
  lightweight network link to an embedded controller.

## Architecture

```
webcam ──▶ OpenCV capture ──▶ MediaPipe Hands ──▶ angle mapper
                                                       │
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
| `tracker.py` | MediaPipe Hands wrapper | `process(frame) -> HandResult \| None` |
| `mapper.py` | Hand features → servo angles | `to_angles(HandResult) -> dict` |
| `smoother.py` | Deadzone + EMA + rate-limit | `smooth(raw_angles) -> angles` |
| `safety.py` | Clutch state, hold-on-loss, e-stop | state machine |
| `robot_client.py` | WebSocket client + reconnect | `send(key, value)` |
| `main.py` | Wire modules, debug preview window | run loop |

`HandResult` holds normalized landmarks, handedness, and a confidence score.

Per-frame flow: `capture → tracker → mapper → smoother → safety → robot_client`.

## Mapping (Approach B — primary: hand-position driving)

MediaPipe returns normalized landmarks (x, y ∈ 0..1). The mapper computes:

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
- `--dry-run` flag prints angles to console instead of sending, validating the
  full pipeline with no ESP32 attached.

## Alternative / future mode: Approach A — Joint Mimicry

Documented for future use; **not** part of the initial implementation.

Instead of tracking only the hand, use **MediaPipe Pose** to track the whole arm
and have the robot copy the operator's joint angles:

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
  `mapper.py` (and swapping MediaPipe Hands for Pose in `tracker.py`); smoothing,
  safety, and the robot client are unchanged.

References for this mode: IJMERR *Pose Estimation-Driven Control of Humanoid
Upper Arms*; IEEE *Real-Time Upper Body Motion Tracking* (MediaPipe Pose +
kinematics, 3-DOF, 94% accuracy).

## Out of scope

- No changes to `robot-control/` (Node relay, ESP32 firmware, web UI).
- No 3D joint reconstruction (deliberately avoided per research).
- No accuracy/RMSE instrumentation (can be added later if the goal shifts to a
  measured write-up).
