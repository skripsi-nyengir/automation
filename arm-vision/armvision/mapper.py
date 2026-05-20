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
