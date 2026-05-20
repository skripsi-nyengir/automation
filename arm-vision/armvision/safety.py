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
