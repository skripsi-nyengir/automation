# Webcam Arm Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python service (`arm-vision/`) that tracks a hand via webcam and drives the existing 4-servo robot arm over the unchanged `key:value` WebSocket protocol.

**Architecture:** Per-frame pipeline `capture → tracker → mapper → smoother → safety → robot_client`. The tracker is a pluggable backend (GPU YOLO11-pose primary, MediaPipe-CPU fallback) behind a fixed `process(frame) -> HandResult | None` interface. Pure modules (mapper, smoother, safety) are unit-tested with synthetic inputs; the WebSocket client is tested against a mock echo server. The legacy `robot-control/` stack is never modified.

**Tech Stack:** Python 3.13, PyTorch nightly cu128 (Blackwell sm_120), Ultralytics YOLO11-pose, OpenCV, `websocket-client`, MediaPipe (optional fallback), pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-20-webcam-arm-control-design.md`

**Legacy protocol facts (do not change the legacy stack):**
- Connect to `ws://<host>:3000/` (server is `robot-control/control-server/server.js`).
- Server marks a connection as a "web client" on its **first non-`ESP32` message** (`server.js:24-41`). Send a benign handshake first.
- ESP32 ignores unknown servo names without crashing (`robot_arm.ino:221-224`), so `vision:hello` is a safe handshake.
- Commands: `base:<0-180>`, `updown:<0-180>`, `arm:<0-180>`, `gripper:<0-180>`, `emergency:STOP`, `emergency:RESUME`.
- Server may push `esp32status:connected|disconnected` messages; the client should tolerate/ignore them.

---

## File Structure

```
arm-vision/
  requirements.txt          # pinned deps + install order notes
  smoke_gpu.py              # standalone GPU/YOLO fail-fast check (Task 1)
  armvision/
    __init__.py
    config.py               # all tunable constants (control box, ranges, smoothing, safety)
    types.py                # HandResult dataclass + landmark index constants
    mapper.py               # HandResult -> servo angles (pure)
    smoother.py             # deadzone + per-servo EMA + rate-limit (pure) + ema_alpha helper
    safety.py               # clutch / tracking-loss state machine + change-gate (pure)
    robot_client.py         # websocket-client wrapper + reconnect + handshake
    tracker.py              # backend factory + YOLO + MediaPipe backends
    capture.py              # OpenCV webcam wrapper
    main.py                 # CLI, wire pipeline, preview window, key handling
  tests/
    test_mapper.py
    test_smoother.py
    test_safety.py
    test_robot_client.py
```

---

## Task 1: GPU smoke test (fail fast before building anything)

**Goal:** Prove the bleeding-edge stack works on this machine (RTX 5050, sm_120) before investing in the pipeline. If this fails, stop and resolve the environment first.

**Files:**
- Create: `arm-vision/requirements.txt`
- Create: `arm-vision/smoke_gpu.py`

- [ ] **Step 1: Create the project directory and requirements file**

Create `arm-vision/requirements.txt`:

```text
# INSTALL ORDER MATTERS — PyTorch nightly (cu128) FIRST, then the rest.
#
#   python -m venv .venv
#   .venv\Scripts\activate            (Windows PowerShell: .venv\Scripts\Activate.ps1)
#   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
#   pip install -r requirements.txt
#
# Stable torch does NOT support Blackwell sm_120 yet; the nightly cu128 wheel does.
# Pin a known-good nightly once verified (e.g. torch==2.10.0.devYYYYMMDD+cu128).

ultralytics
opencv-python
websocket-client
pytest
# Optional CPU fallback backend. May lack a Python 3.13 wheel; install only if needed:
#   pip install mediapipe
```

- [ ] **Step 2: Write the smoke test script**

Create `arm-vision/smoke_gpu.py`:

```python
"""Fail-fast GPU check. Run before building the pipeline.

Verifies: PyTorch sees CUDA, the device is the RTX 5050 (Blackwell sm_120),
and a YOLO11-pose model actually runs a forward pass on CUDA.
Exits non-zero with a clear message on any failure.
"""
import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch not installed. Install PyTorch nightly cu128 first.")
        return 1

    print(f"torch version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False. "
              "Check NVIDIA driver + that you installed the cu128 nightly wheel.")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)  # (12, 0) == sm_120 for Blackwell
    print(f"CUDA device: {name}  capability: sm_{cap[0]}{cap[1]}")
    if cap[0] < 12:
        print(f"WARN: expected sm_120 (Blackwell), got sm_{cap[0]}{cap[1]}. Continuing.")

    # Prove a model runs on CUDA. yolo11n-pose.pt is the generic (body) pose
    # model — used here only to confirm CUDA inference works end to end.
    try:
        import numpy as np
        from ultralytics import YOLO
    except ImportError as e:
        print(f"FAIL: ultralytics/numpy not installed: {e}")
        return 1

    model = YOLO("yolo11n-pose.pt")  # auto-downloads on first run
    dummy = np.zeros((640, 640, 3), dtype="uint8")
    results = model.predict(dummy, device="cuda", verbose=False)
    dev = results[0].boxes.data.device if results[0].boxes is not None else "cuda"
    print(f"YOLO ran on device: {dev}")
    print("PASS: GPU stack is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Set up the environment and install PyTorch nightly**

Run (Windows PowerShell, from `arm-vision/`):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

Expected: installs complete without error. If the nightly wheel fails to resolve, note the exact error — this is the fail-fast gate.

- [ ] **Step 4: Run the smoke test**

Run: `python smoke_gpu.py`
Expected output ends with `PASS: GPU stack is ready.` and prints `CUDA device: NVIDIA GeForce RTX 5050 Laptop GPU  capability: sm_120`.

**If it fails:** stop here. Resolve the environment (driver, nightly wheel, CUDA) before continuing. Do not proceed to Task 2 on a broken GPU stack — the `mediapipe-cpu` backend exists as a fallback, but the user chose the GPU path.

- [ ] **Step 5: Pin the working torch version and commit**

After PASS, capture the resolved version: `pip show torch` → copy the version into `requirements.txt` as a comment (e.g. `# verified: torch==2.10.0.devYYYYMMDD+cu128`).

```bash
git add arm-vision/requirements.txt arm-vision/smoke_gpu.py
git commit -m "feat(arm-vision): GPU smoke test + pinned torch nightly cu128"
```

---

## Task 2: Project package skeleton + types

**Files:**
- Create: `arm-vision/armvision/__init__.py` (empty)
- Create: `arm-vision/armvision/types.py`
- Create: `arm-vision/tests/__init__.py` (empty)

- [ ] **Step 1: Create the empty package markers**

Create `arm-vision/armvision/__init__.py` with a single line:

```python
"""Webcam-driven control service for the 4-servo robot arm."""
```

Create `arm-vision/tests/__init__.py` as an empty file.

- [ ] **Step 2: Write `types.py` with the HandResult contract**

Create `arm-vision/armvision/types.py`:

```python
"""Shared types. HandResult is the single contract between tracker and the
rest of the pipeline; every backend must populate it identically."""
from __future__ import annotations

from dataclasses import dataclass

# MediaPipe-topology landmark indices (YOLO hand-keypoints uses the same order).
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9

NUM_LANDMARKS = 21


@dataclass(frozen=True)
class HandResult:
    """One detected hand for one frame.

    landmarks: 21 (x, y) pairs, each normalized to 0..1 in image space
               (x = left→right, y = top→bottom).
    confidence: detection/landmark confidence 0..1.
    handedness: "Left"/"Right"/None (not required by the mapper).
    """
    landmarks: tuple[tuple[float, float], ...]
    confidence: float
    handedness: str | None = None

    def point(self, index: int) -> tuple[float, float]:
        return self.landmarks[index]
```

- [ ] **Step 3: Commit**

```bash
git add arm-vision/armvision/__init__.py arm-vision/armvision/types.py arm-vision/tests/__init__.py
git commit -m "feat(arm-vision): package skeleton + HandResult type"
```

---

## Task 3: Config constants

**Files:**
- Create: `arm-vision/armvision/config.py`

- [ ] **Step 1: Write `config.py`**

Create `arm-vision/armvision/config.py`:

```python
"""All tunable constants in one place. Imported by mapper/smoother/safety/main."""
from __future__ import annotations

from dataclasses import dataclass, field

SERVOS = ("base", "updown", "arm", "gripper")
HOME_ANGLE = 90


@dataclass(frozen=True)
class MapperConfig:
    # Control box: map this centered sub-region of the frame to 0..180 so the
    # operator need not reach the frame edges. 0.6 == middle 60%.
    box_fraction: float = 0.60
    # Apparent hand size (wrist↔middle-MCP distance, normalized) -> arm axis.
    size_min: float = 0.10  # far / small hand  -> arm 0°
    size_max: float = 0.35  # near / big hand   -> arm 180°
    # Pinch distance (thumb-tip↔index-tip, normalized by hand size) -> gripper.
    pinch_min: float = 0.15  # fingers together -> gripper 0° (closed)
    pinch_max: float = 0.80  # fingers apart    -> gripper 180° (open)


@dataclass(frozen=True)
class ServoSmoothing:
    tau: float          # EMA time constant (s); larger = heavier smoothing
    max_delta: float    # max change per frame (deg)
    deadzone: float     # ignore raw changes smaller than this (deg)


@dataclass(frozen=True)
class SmootherConfig:
    # arm is the noisy axis (hand-size) -> heaviest smoothing, tightest rate-limit.
    per_servo: dict[str, ServoSmoothing] = field(default_factory=lambda: {
        "base":    ServoSmoothing(tau=0.08, max_delta=8.0, deadzone=2.0),
        "updown":  ServoSmoothing(tau=0.08, max_delta=8.0, deadzone=2.0),
        "arm":     ServoSmoothing(tau=0.20, max_delta=4.0, deadzone=2.0),
        "gripper": ServoSmoothing(tau=0.06, max_delta=10.0, deadzone=2.0),
    })


@dataclass(frozen=True)
class SafetyConfig:
    confidence_threshold: float = 0.5
    loss_frames: int = 5  # consecutive bad frames before freezing


@dataclass(frozen=True)
class AppConfig:
    mapper: MapperConfig = field(default_factory=MapperConfig)
    smoother: SmootherConfig = field(default_factory=SmootherConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ws_url: str = "ws://localhost:3000/"
    handshake: str = "vision:hello"
```

- [ ] **Step 2: Commit**

```bash
git add arm-vision/armvision/config.py
git commit -m "feat(arm-vision): central config constants"
```

---

## Task 4: Mapper (pure, TDD)

**Files:**
- Create: `arm-vision/tests/test_mapper.py`
- Create: `arm-vision/armvision/mapper.py`

- [ ] **Step 1: Write the failing tests**

Create `arm-vision/tests/test_mapper.py`:

```python
import math

from armvision.config import MapperConfig
from armvision.mapper import to_angles
from armvision.types import HandResult, NUM_LANDMARKS


def make_hand(center=(0.5, 0.5), size=0.225, pinch=0.475):
    """Build a synthetic HandResult with a known center, hand size and pinch.

    Places wrist and middle-MCP `size` apart vertically (so hand-size distance
    == size), and thumb/index tips `pinch`*size apart so pinch ratio == pinch.
    All other landmarks sit at center.
    """
    cx, cy = center
    pts = [(cx, cy)] * NUM_LANDMARKS
    pts[0] = (cx, cy + size / 2)   # WRIST
    pts[9] = (cx, cy - size / 2)   # MIDDLE_MCP -> distance == size
    sep = pinch * size
    pts[4] = (cx - sep / 2, cy)    # THUMB_TIP
    pts[8] = (cx + sep / 2, cy)    # INDEX_TIP -> distance == pinch*size
    return HandResult(landmarks=tuple(pts), confidence=1.0)


def test_centered_hand_gives_mid_base_and_updown():
    angles = to_angles(make_hand(center=(0.5, 0.5)), MapperConfig())
    assert angles["base"] == 90
    assert angles["updown"] == 90


def test_base_increases_left_to_right():
    cfg = MapperConfig()
    left = to_angles(make_hand(center=(0.2, 0.5)), cfg)["base"]
    right = to_angles(make_hand(center=(0.8, 0.5)), cfg)["base"]
    assert left == 0      # at/under box low edge
    assert right == 180   # at/over box high edge


def test_updown_is_inverted():
    cfg = MapperConfig()
    top = to_angles(make_hand(center=(0.5, 0.2)), cfg)["updown"]
    bottom = to_angles(make_hand(center=(0.5, 0.8)), cfg)["updown"]
    assert top == 180     # top of frame -> max
    assert bottom == 0    # bottom of frame -> min


def test_arm_from_hand_size():
    cfg = MapperConfig()
    near = to_angles(make_hand(size=cfg.size_max), cfg)["arm"]
    far = to_angles(make_hand(size=cfg.size_min), cfg)["arm"]
    assert near == 180
    assert far == 0


def test_gripper_from_pinch():
    cfg = MapperConfig()
    wide = to_angles(make_hand(pinch=cfg.pinch_max), cfg)["gripper"]
    closed = to_angles(make_hand(pinch=cfg.pinch_min), cfg)["gripper"]
    assert wide == 180
    assert closed == 0


def test_all_angles_clamped_0_180():
    cfg = MapperConfig()
    angles = to_angles(make_hand(center=(2.0, -1.0), size=99, pinch=99), cfg)
    for v in angles.values():
        assert 0 <= v <= 180
        assert isinstance(v, int)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd arm-vision && python -m pytest tests/test_mapper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'armvision.mapper'`.

- [ ] **Step 3: Implement `mapper.py`**

Create `arm-vision/armvision/mapper.py`:

```python
"""Pure mapping from hand features to servo angles (0..180 ints)."""
from __future__ import annotations

import math

from .config import MapperConfig
from .types import HandResult, WRIST, MIDDLE_MCP, THUMB_TIP, INDEX_TIP, NUM_LANDMARKS


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _norm(value: float, lo: float, hi: float) -> float:
    """Linear map value in [lo, hi] -> [0, 1], clamped."""
    if hi == lo:
        return 0.0
    t = (value - lo) / (hi - lo)
    return max(0.0, min(1.0, t))


def _angle(t: float) -> int:
    return int(round(max(0.0, min(1.0, t)) * 180))


def _center(hand: HandResult) -> tuple[float, float]:
    xs = sum(p[0] for p in hand.landmarks) / NUM_LANDMARKS
    ys = sum(p[1] for p in hand.landmarks) / NUM_LANDMARKS
    return xs, ys


def to_angles(hand: HandResult, cfg: MapperConfig) -> dict[str, int]:
    cx, cy = _center(hand)

    # Control box: middle `box_fraction` of the frame maps to full 0..180.
    margin = (1.0 - cfg.box_fraction) / 2.0
    lo, hi = margin, 1.0 - margin

    base = _angle(_norm(cx, lo, hi))
    updown = _angle(1.0 - _norm(cy, lo, hi))  # inverted: top -> 180

    size = _dist(hand.point(WRIST), hand.point(MIDDLE_MCP))
    arm = _angle(_norm(size, cfg.size_min, cfg.size_max))

    # Pinch normalized by hand size so it is scale-invariant.
    pinch_raw = _dist(hand.point(THUMB_TIP), hand.point(INDEX_TIP))
    pinch_ratio = pinch_raw / size if size > 1e-6 else 0.0
    gripper = _angle(_norm(pinch_ratio, cfg.pinch_min, cfg.pinch_max))

    return {"base": base, "updown": updown, "arm": arm, "gripper": gripper}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd arm-vision && python -m pytest tests/test_mapper.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add arm-vision/armvision/mapper.py arm-vision/tests/test_mapper.py
git commit -m "feat(arm-vision): hand-feature -> servo-angle mapper"
```

---

## Task 5: Smoother (pure, TDD)

**Files:**
- Create: `arm-vision/tests/test_smoother.py`
- Create: `arm-vision/armvision/smoother.py`

- [ ] **Step 1: Write the failing tests**

Create `arm-vision/tests/test_smoother.py`:

```python
import math

from armvision.config import SmootherConfig, ServoSmoothing
from armvision.smoother import Smoother, ema_alpha


def test_ema_alpha_formula():
    # alpha = 1 - exp(-dt/tau); tau == dt -> 1 - 1/e
    assert math.isclose(ema_alpha(0.1, 0.1), 1 - math.exp(-1), rel_tol=1e-9)


def test_ema_alpha_clamped_to_unit_interval():
    assert ema_alpha(0.0, 0.1) == 1.0      # tau 0 -> no smoothing
    assert 0.0 <= ema_alpha(1.0, 0.0) <= 1.0


def single_cfg(tau=0.1, max_delta=8.0, deadzone=2.0):
    return SmootherConfig(per_servo={
        "base": ServoSmoothing(tau=tau, max_delta=max_delta, deadzone=deadzone),
    })


def test_first_frame_passes_through():
    s = Smoother(single_cfg())
    out = s.smooth({"base": 100.0}, dt=0.033)
    assert out["base"] == 100.0


def test_deadzone_ignores_small_wiggle():
    s = Smoother(single_cfg(deadzone=2.0))
    s.smooth({"base": 100.0}, dt=0.033)          # establish 100
    out = s.smooth({"base": 101.0}, dt=0.033)    # 1° < deadzone -> held
    assert out["base"] == 100.0


def test_rate_limit_caps_per_frame_change():
    s = Smoother(single_cfg(tau=0.0001, max_delta=8.0))  # alpha~1, EMA ~ raw
    s.smooth({"base": 100.0}, dt=0.033)
    out = s.smooth({"base": 200.0}, dt=0.033)
    assert out["base"] == 108.0  # capped to +8 from 100


def test_ema_pulls_toward_target():
    s = Smoother(single_cfg(tau=0.1, max_delta=999.0, deadzone=0.0))
    s.smooth({"base": 0.0}, dt=0.1)
    out = s.smooth({"base": 100.0}, dt=0.1)
    # alpha = 1-1/e ≈ 0.632 -> 63.2
    assert math.isclose(out["base"], 100 * (1 - math.exp(-1)), rel_tol=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd arm-vision && python -m pytest tests/test_smoother.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'armvision.smoother'`.

- [ ] **Step 3: Implement `smoother.py`**

Create `arm-vision/armvision/smoother.py`:

```python
"""Per-servo deadzone + FPS-aware EMA + rate-limit. Pure (state held per instance)."""
from __future__ import annotations

import math

from .config import SmootherConfig


def ema_alpha(tau: float, dt: float) -> float:
    """FPS-aware EMA factor: alpha = 1 - exp(-dt/tau), clamped to [0, 1].

    tau <= 0 means 'no smoothing' (alpha = 1)."""
    if tau <= 0.0:
        return 1.0
    if dt <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.exp(-dt / tau)))


class Smoother:
    def __init__(self, cfg: SmootherConfig):
        self._cfg = cfg
        self._last: dict[str, float] = {}

    def smooth(self, raw_angles: dict[str, float], dt: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for servo, raw in raw_angles.items():
            params = self._cfg.per_servo[servo]
            prev = self._last.get(servo)

            if prev is None:
                out[servo] = raw            # first frame: adopt as-is
                self._last[servo] = raw
                continue

            # Deadzone: ignore tiny raw wiggle.
            if abs(raw - prev) < params.deadzone:
                out[servo] = prev
                continue

            # EMA toward target.
            alpha = ema_alpha(params.tau, dt)
            target = prev + alpha * (raw - prev)

            # Rate-limit per frame.
            delta = target - prev
            if delta > params.max_delta:
                target = prev + params.max_delta
            elif delta < -params.max_delta:
                target = prev - params.max_delta

            out[servo] = target
            self._last[servo] = target
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd arm-vision && python -m pytest tests/test_smoother.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add arm-vision/armvision/smoother.py arm-vision/tests/test_smoother.py
git commit -m "feat(arm-vision): per-servo fps-aware smoother"
```

---

## Task 6: Safety state machine + change-gate (pure, TDD)

**Files:**
- Create: `arm-vision/tests/test_safety.py`
- Create: `arm-vision/armvision/safety.py`

- [ ] **Step 1: Write the failing tests**

Create `arm-vision/tests/test_safety.py`:

```python
from armvision.config import SafetyConfig
from armvision.safety import SafetyController, changed_servos


def test_changed_servos_returns_only_diffs():
    prev = {"base": 90, "updown": 90}
    new = {"base": 95, "updown": 90}
    assert changed_servos(prev, new) == {"base": 95}


def test_changed_servos_all_new_when_no_prev():
    assert changed_servos({}, {"base": 90}) == {"base": 90}


def cfg():
    return SafetyConfig(confidence_threshold=0.5, loss_frames=3)


def test_idle_clutch_blocks_sending():
    sc = SafetyController(cfg())  # starts idle
    d = sc.step(angles={"base": 100}, hand_present=True, confidence=0.9)
    assert d.send is False


def test_engaged_with_good_hand_sends():
    sc = SafetyController(cfg())
    sc.toggle_clutch()  # engage
    d = sc.step(angles={"base": 100}, hand_present=True, confidence=0.9)
    assert d.send is True
    assert d.angles == {"base": 100}


def test_low_confidence_counts_as_loss_after_n_frames():
    sc = SafetyController(cfg())
    sc.toggle_clutch()
    sc.step(angles={"base": 100}, hand_present=True, confidence=0.9)  # ok
    for _ in range(3):  # loss_frames == 3
        d = sc.step(angles={"base": 120}, hand_present=False, confidence=0.0)
    assert d.send is False  # frozen, holds last


def test_reengages_after_loss_without_snap():
    sc = SafetyController(cfg())
    sc.toggle_clutch()
    sc.step(angles={"base": 100}, hand_present=True, confidence=0.9)
    for _ in range(3):
        sc.step(angles={"base": 999}, hand_present=False, confidence=0.0)
    d = sc.step(angles={"base": 105}, hand_present=True, confidence=0.9)
    assert d.send is True
    assert d.angles == {"base": 105}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd arm-vision && python -m pytest tests/test_safety.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'armvision.safety'`.

- [ ] **Step 3: Implement `safety.py`**

Create `arm-vision/armvision/safety.py`:

```python
"""Clutch + tracking-loss state machine and change-gating. Pure logic only;
no I/O. Emergency stop is handled in main (key handler), independent of this."""
from __future__ import annotations

from dataclasses import dataclass

from .config import SafetyConfig


def changed_servos(prev: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    """Return only the servos whose value differs from prev (change-gating)."""
    return {k: v for k, v in new.items() if prev.get(k) != v}


@dataclass(frozen=True)
class SafetyDecision:
    send: bool                 # may we transmit this frame?
    angles: dict[str, float]   # angles to use (held value when not sending)


class SafetyController:
    def __init__(self, cfg: SafetyConfig):
        self._cfg = cfg
        self._engaged = False           # clutch starts idle
        self._lost_count = 0
        self._held: dict[str, float] = {}

    @property
    def engaged(self) -> bool:
        return self._engaged

    def toggle_clutch(self) -> None:
        self._engaged = not self._engaged

    def step(self, angles: dict[str, float], hand_present: bool,
             confidence: float) -> SafetyDecision:
        good = hand_present and confidence >= self._cfg.confidence_threshold

        if good:
            self._lost_count = 0
        else:
            self._lost_count += 1

        tracking_ok = self._lost_count < self._cfg.loss_frames

        if self._engaged and good and tracking_ok:
            self._held = dict(angles)
            return SafetyDecision(send=True, angles=dict(angles))

        # Idle, or tracking lost: hold last commanded position, send nothing.
        return SafetyDecision(send=False, angles=dict(self._held))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd arm-vision && python -m pytest tests/test_safety.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add arm-vision/armvision/safety.py arm-vision/tests/test_safety.py
git commit -m "feat(arm-vision): clutch/tracking-loss safety + change-gate"
```

---

## Task 7: Robot WebSocket client (TDD against a mock server)

**Files:**
- Create: `arm-vision/tests/test_robot_client.py`
- Create: `arm-vision/armvision/robot_client.py`

- [ ] **Step 1: Write the failing tests**

Create `arm-vision/tests/test_robot_client.py`. The mock server uses the stdlib-free `websocket-client`'s companion is not a server, so we run a tiny server thread with the `websockets` library if present; otherwise use a raw socket. To avoid extra deps, this mock uses Python's `socket`/`http`-free approach via `websocket-client` is client-only — so we implement a minimal server with the `websockets` library, which is a common transitive dep. If `websockets` is unavailable, the test skips with a clear message.

```python
import threading
import time

import pytest

from armvision.robot_client import RobotClient

websockets = pytest.importorskip(
    "websockets", reason="pip install websockets to run robot_client tests")
import asyncio


class MockServer:
    """Minimal WebSocket echo/record server on a background asyncio loop."""
    def __init__(self):
        self.received: list[str] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.port = None
        self._ready = threading.Event()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())
        self._loop.run_forever()

    async def _serve(self):
        async def handler(ws):
            async for msg in ws:
                self.received.append(msg)
        self._server = await websockets.serve(handler, "localhost", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        self._ready.set()

    def start(self):
        self._thread.start()
        assert self._ready.wait(timeout=5)

    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)


@pytest.fixture
def server():
    s = MockServer()
    s.start()
    yield s
    s.stop()


def wait_for(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_sends_handshake_on_connect(server):
    client = RobotClient(f"ws://localhost:{server.port}/", handshake="vision:hello")
    client.connect()
    assert wait_for(lambda: "vision:hello" in server.received)
    client.close()


def test_send_formats_key_value(server):
    client = RobotClient(f"ws://localhost:{server.port}/", handshake="vision:hello")
    client.connect()
    assert wait_for(lambda: client.is_connected)
    client.send("base", 120)
    assert wait_for(lambda: "base:120" in server.received)
    client.close()


def test_send_before_connect_is_dropped_not_crash():
    client = RobotClient("ws://localhost:59999/", handshake="vision:hello")
    client.send("base", 90)  # not connected -> no exception
    assert client.is_connected is False
```

- [ ] **Step 2: Install the test dependency and run to verify failure**

Run: `cd arm-vision && pip install websockets && python -m pytest tests/test_robot_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'armvision.robot_client'`.

- [ ] **Step 3: Implement `robot_client.py`**

Create `arm-vision/armvision/robot_client.py`:

```python
"""WebSocket client to the legacy relay. Runs in a background thread with
auto-reconnect. Registers as a 'web client' by sending a non-ESP32 handshake
on connect (see server.js). Tolerates inbound esp32status messages."""
from __future__ import annotations

import threading

import websocket  # from the `websocket-client` package


class RobotClient:
    def __init__(self, url: str, handshake: str = "vision:hello",
                 reconnect_seconds: float = 3.0):
        self._url = url
        self._handshake = handshake
        self._reconnect = reconnect_seconds
        self._app: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._connected = False
        self._stop = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            self._app = websocket.WebSocketApp(
                self._url,
                on_open=self._on_open,
                on_close=self._on_close,
                on_error=self._on_error,
                on_message=self._on_message,
            )
            # run_forever returns on disconnect; loop to reconnect.
            self._app.run_forever()
            self._connected = False
            if self._stop:
                break
            # Match the legacy web client's 3s retry.
            for _ in range(int(self._reconnect * 10)):
                if self._stop:
                    return
                threading.Event().wait(0.1)

    def _on_open(self, _app) -> None:
        self._connected = True
        try:
            self._app.send(self._handshake)  # register as web client
        except Exception:
            pass

    def _on_close(self, _app, *_args) -> None:
        self._connected = False

    def _on_error(self, _app, _err) -> None:
        self._connected = False

    def _on_message(self, _app, _msg) -> None:
        pass  # esp32status / echoes are ignored

    def send(self, key: str, value) -> None:
        if not self._connected or self._app is None:
            return
        try:
            self._app.send(f"{key}:{value}")
        except Exception:
            self._connected = False

    def close(self) -> None:
        self._stop = True
        if self._app is not None:
            try:
                self._app.close()
            except Exception:
                pass
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd arm-vision && python -m pytest tests/test_robot_client.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Add `websockets` to requirements and commit**

Append to `arm-vision/requirements.txt`:

```text
websockets  # test-only: mock WebSocket server for robot_client tests
```

```bash
git add arm-vision/armvision/robot_client.py arm-vision/tests/test_robot_client.py arm-vision/requirements.txt
git commit -m "feat(arm-vision): websocket robot client with reconnect + handshake"
```

---

## Task 8: Capture (OpenCV webcam wrapper)

**Files:**
- Create: `arm-vision/armvision/capture.py`

This wraps the camera; verified by manual smoke test in Task 10 (no unit test — it needs hardware).

- [ ] **Step 1: Implement `capture.py`**

Create `arm-vision/armvision/capture.py`:

```python
"""OpenCV webcam wrapper. read() -> BGR frame (numpy array)."""
from __future__ import annotations

import cv2


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index: int = 0):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open webcam at index {index}. "
                "Check it is connected and not in use by another app.")

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            raise CameraError("Failed to read frame from webcam.")
        return frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
```

- [ ] **Step 2: Commit**

```bash
git add arm-vision/armvision/capture.py
git commit -m "feat(arm-vision): OpenCV camera wrapper"
```

---

## Task 9: Tracker backends (YOLO GPU + MediaPipe CPU)

**Files:**
- Create: `arm-vision/armvision/tracker.py`

Backends are verified by manual smoke test (Task 10), not unit tests — they need a camera/model. Both return the identical `HandResult`.

> **RISK — read before implementing:** Ultralytics ships generic `yolo11n-pose.pt` (COCO body, 17 keypoints), but a **pretrained 21-keypoint hand** model is not guaranteed to be published. Two outcomes:
> 1. A hand-keypoints checkpoint is available (community/HF or Ultralytics) → set `YOLO_HAND_MODEL` to it.
> 2. None available → train one once: `yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640` (dataset auto-downloads; Ultralytics docs: *Hand Keypoints Dataset*), then point `YOLO_HAND_MODEL` at the resulting `best.pt`.
> The model must output 21 keypoints in MediaPipe order. If indices differ, remap inside `_to_hand_result` — keep the rest of the pipeline untouched.

- [ ] **Step 1: Implement `tracker.py`**

Create `arm-vision/armvision/tracker.py`:

```python
"""Pluggable hand-landmark backends behind a fixed interface.

    tracker = make_tracker("yolo-gpu")     # or "mediapipe-cpu"
    result = tracker.process(frame)        # -> HandResult | None
    print(tracker.device)                  # for the startup log
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import HandResult, NUM_LANDMARKS

# Point at a 21-keypoint hand model (see RISK note in the plan).
YOLO_HAND_MODEL = "yolo11n-pose.pt"


class Tracker(Protocol):
    device: str
    def process(self, frame) -> HandResult | None: ...


class YoloTracker:
    def __init__(self, model_path: str = YOLO_HAND_MODEL, conf: float = 0.5):
        import torch
        from ultralytics import YOLO
        self._cuda = torch.cuda.is_available()
        self.device = "cuda" if self._cuda else "cpu"
        self._conf = conf
        self._model = YOLO(model_path)

    def process(self, frame) -> HandResult | None:
        h, w = frame.shape[:2]
        results = self._model.predict(
            frame, device=self.device, conf=self._conf, verbose=False)
        r = results[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None
        kpts = r.keypoints.xy[0].cpu().numpy()  # (K, 2) pixel coords
        if kpts.shape[0] < NUM_LANDMARKS:
            return None
        conf = 1.0
        if r.boxes is not None and len(r.boxes) > 0:
            conf = float(r.boxes.conf[0].cpu().item())
        landmarks = tuple((float(x / w), float(y / h)) for x, y in kpts[:NUM_LANDMARKS])
        return HandResult(landmarks=landmarks, confidence=conf)


class MediaPipeTracker:
    """CPU fallback. MediaPipe is imported lazily so its absence (e.g. no
    Python 3.13 wheel) never breaks the YOLO path."""
    def __init__(self, conf: float = 0.5):
        import mediapipe as mp
        self.device = "cpu"
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=conf,
            min_tracking_confidence=conf)

    def process(self, frame) -> HandResult | None:
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0].landmark
        landmarks = tuple((p.x, p.y) for p in lm[:NUM_LANDMARKS])
        conf = 1.0
        handed = None
        if res.multi_handedness:
            cl = res.multi_handedness[0].classification[0]
            conf = float(cl.score)
            handed = cl.label
        return HandResult(landmarks=landmarks, confidence=conf, handedness=handed)


def make_tracker(backend: str) -> Tracker:
    if backend == "yolo-gpu":
        return YoloTracker()
    if backend == "mediapipe-cpu":
        return MediaPipeTracker()
    raise ValueError(f"Unknown backend: {backend!r} (use 'yolo-gpu' or 'mediapipe-cpu')")
```

- [ ] **Step 2: Commit**

```bash
git add arm-vision/armvision/tracker.py
git commit -m "feat(arm-vision): pluggable YOLO/MediaPipe tracker backends"
```

---

## Task 10: Main wiring, CLI, preview, key handling, dry-run

**Files:**
- Create: `arm-vision/armvision/main.py`

- [ ] **Step 1: Implement `main.py`**

Create `arm-vision/armvision/main.py`:

```python
"""Entry point: wire the pipeline, draw a preview, handle keys.

Keys:  SPACE = toggle clutch (engage/idle)   q / ESC = emergency STOP + quit
       r = emergency RESUME

Flags: --backend {yolo-gpu,mediapipe-cpu}   --camera N   --ws URL
       --dry-run  (print angles instead of sending; no robot needed)
"""
from __future__ import annotations

import argparse
import time

import cv2

from .capture import Camera, CameraError
from .config import AppConfig, SERVOS, HOME_ANGLE
from .mapper import to_angles
from .robot_client import RobotClient
from .safety import SafetyController, changed_servos
from .smoother import Smoother
from .tracker import make_tracker


def parse_args():
    p = argparse.ArgumentParser(description="Webcam hand control for the robot arm.")
    p.add_argument("--backend", default="yolo-gpu",
                   choices=["yolo-gpu", "mediapipe-cpu"])
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--ws", default=None, help="WebSocket URL (overrides config)")
    p.add_argument("--dry-run", action="store_true",
                   help="print angles instead of sending to the robot")
    return p.parse_args()


def draw_overlay(frame, angles, engaged, sending, device, fps):
    lines = [
        f"backend dev: {device}   fps: {fps:4.1f}",
        f"clutch: {'ENGAGED' if engaged else 'idle'}   "
        f"sending: {'yes' if sending else 'no'}",
        "  ".join(f"{s}:{int(angles.get(s, HOME_ANGLE))}" for s in SERVOS),
        "SPACE clutch | r resume | q/ESC stop+quit",
    ]
    y = 24
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        y += 26


def main():
    args = parse_args()
    cfg = AppConfig()
    ws_url = args.ws or cfg.ws_url

    try:
        cam = Camera(args.camera)
    except CameraError as e:
        print(f"ERROR: {e}")
        return 1

    tracker = make_tracker(args.backend)
    print(f"[startup] backend={args.backend} device={tracker.device}")
    if args.backend == "yolo-gpu" and tracker.device != "cuda":
        print("WARNING: yolo-gpu fell back to CPU — FPS will be low. "
              "Re-run the GPU smoke test (smoke_gpu.py).")

    smoother = Smoother(cfg.smoother)
    safety = SafetyController(cfg.safety)

    client = None
    if not args.dry_run:
        client = RobotClient(ws_url, handshake=cfg.handshake)
        client.connect()
        print(f"[startup] connecting to {ws_url}")

    last_sent: dict[str, int] = {}
    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            frame = cam.read()
            frame = cv2.flip(frame, 1)  # mirror so movement feels natural
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            result = tracker.process(frame)
            hand_present = result is not None
            confidence = result.confidence if result else 0.0

            if hand_present:
                raw = to_angles(result, cfg.mapper)
                smoothed = smoother.smooth(
                    {k: float(v) for k, v in raw.items()}, dt)
            else:
                smoothed = {}

            decision = safety.step(
                angles=smoothed, hand_present=hand_present, confidence=confidence)

            int_angles = {k: int(round(v)) for k, v in decision.angles.items()}
            if decision.send:
                for servo, value in changed_servos(last_sent, int_angles).items():
                    if args.dry_run:
                        print(f"{servo}:{value}")
                    elif client:
                        client.send(servo, value)
                last_sent = dict(int_angles)

            draw_overlay(frame, int_angles or last_sent, safety.engaged,
                         decision.send, tracker.device, fps)
            cv2.imshow("arm-vision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                safety.toggle_clutch()
            elif key == ord("r"):
                if client:
                    client.send("emergency", "RESUME")
                print("[key] emergency RESUME")
            elif key in (ord("q"), 27):  # q or ESC
                if client:
                    client.send("emergency", "STOP")
                print("[key] emergency STOP + quit")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        if client:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Dry-run smoke test (no robot, no GPU dependency on backend)**

Run: `cd arm-vision && python -m armvision.main --dry-run`
Expected: a preview window opens; move your hand and watch `base:NN` etc. print only when SPACE has engaged the clutch and a value changes. Press `q` to quit.

If `yolo-gpu` cannot load a 21-keypoint hand model yet (see Task 9 RISK), run the same smoke test with `--backend mediapipe-cpu --dry-run` to validate the full pipeline while the hand model is sorted out.

- [ ] **Step 3: Live smoke test against the real relay (optional, needs robot)**

Start the legacy relay (`robot-control/`), power the ESP32, then:
Run: `cd arm-vision && python -m armvision.main --backend yolo-gpu --ws ws://<relay-host>:3000/`
Expected: startup log shows `device=cuda`; engaging the clutch (SPACE) moves the arm; `q` sends emergency STOP.

- [ ] **Step 4: Commit**

```bash
git add arm-vision/armvision/main.py
git commit -m "feat(arm-vision): main pipeline, preview, CLI, dry-run"
```

---

## Task 11: Full test run + README

**Files:**
- Create: `arm-vision/README.md`

- [ ] **Step 1: Run the whole unit suite**

Run: `cd arm-vision && python -m pytest -v`
Expected: all tests in `test_mapper.py`, `test_smoother.py`, `test_safety.py`, `test_robot_client.py` PASS.

- [ ] **Step 2: Write `README.md`**

Create `arm-vision/README.md` documenting: purpose, the install order (PyTorch nightly cu128 first), `python smoke_gpu.py` gate, the `--backend`/`--camera`/`--ws`/`--dry-run` flags, key bindings (SPACE/r/q/ESC), and the Task 9 hand-model RISK note (how to obtain or train the 21-keypoint model). Keep it to the essentials a new operator needs.

- [ ] **Step 3: Commit**

```bash
git add arm-vision/README.md
git commit -m "docs(arm-vision): operator README"
```

---

## Self-review notes (coverage vs spec)

- Architecture pipeline, `arm-vision/` sibling dir, legacy untouched → Tasks 2–10.
- Inference backend (pluggable, YOLO-GPU default, MediaPipe-CPU fallback, device log/warn) → Tasks 9, 10; GPU gate → Task 1.
- Mapping (control box, inverted updown, hand-size arm, pinch gripper, clamping) → Task 4.
- Smoothing (deadzone, per-servo EMA, rate-limit, FPS-aware α) → Task 5.
- Safety (clutch, tracking-loss freeze/re-engage, change-gated sends, emergency) → Tasks 6, 10.
- Error handling (camera failure, WS reconnect, malformed landmarks = loss) → Tasks 7, 8, 9, 10.
- Testing (pure units, mock WS server, manual smoke, `--dry-run`) → Tasks 4–7, 10, 11.
- Out of scope (no 3D, no RMSE, no legacy changes) → respected throughout.
```
