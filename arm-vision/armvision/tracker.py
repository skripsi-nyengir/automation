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
    """CPU fallback using the MediaPipe Tasks HandLandmarker.

    The legacy mp.solutions.hands API is unavailable on Python 3.13, so this
    uses mp.tasks. The hand_landmarker.task model bundle (21 landmarks, same
    topology) is downloaded once on first use if not present. Lazy imports keep
    MediaPipe's absence from ever breaking the YOLO path."""

    MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

    def __init__(self, conf: float = 0.5, model_path: str = "hand_landmarker.task"):
        import os
        import urllib.request

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.exists(model_path):
            print(f"[tracker] downloading MediaPipe hand model to {model_path} ...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)

        self.device = "cpu"
        self._mp = mp
        base = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=1,
            min_hand_detection_confidence=conf,
            min_hand_presence_confidence=conf,
            min_tracking_confidence=conf,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame) -> HandResult | None:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        landmarks = tuple((p.x, p.y) for p in lm[:NUM_LANDMARKS])
        conf = 1.0
        handed = None
        if result.handedness and result.handedness[0]:
            cat = result.handedness[0][0]
            conf = float(cat.score)
            handed = cat.category_name
        return HandResult(landmarks=landmarks, confidence=conf, handedness=handed)


def make_tracker(backend: str) -> Tracker:
    if backend == "yolo-gpu":
        return YoloTracker()
    if backend == "mediapipe-cpu":
        return MediaPipeTracker()
    raise ValueError(f"Unknown backend: {backend!r} (use 'yolo-gpu' or 'mediapipe-cpu')")
