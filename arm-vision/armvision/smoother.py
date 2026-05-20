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
