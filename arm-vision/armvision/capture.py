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
