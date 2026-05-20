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
