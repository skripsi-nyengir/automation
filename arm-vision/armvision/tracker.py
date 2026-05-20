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
