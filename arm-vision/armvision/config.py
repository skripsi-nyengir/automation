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
