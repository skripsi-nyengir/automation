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
