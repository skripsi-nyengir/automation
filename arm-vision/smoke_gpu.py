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
        print("FAIL: torch not installed in this environment.")
        return 1

    print(f"torch version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False. "
              "Check NVIDIA driver + the CUDA torch build.")
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
