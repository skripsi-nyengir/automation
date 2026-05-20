import math

from armvision.config import MapperConfig
from armvision.mapper import to_angles
from armvision.types import HandResult, NUM_LANDMARKS


def make_hand(center=(0.5, 0.5), size=0.225, pinch=0.475):
    """Build a synthetic HandResult with a known center, hand size and pinch.

    Places wrist and middle-MCP `size` apart vertically (so hand-size distance
    == size), and thumb/index tips `pinch`*size apart so pinch ratio == pinch.
    All other landmarks sit at center.
    """
    cx, cy = center
    pts = [(cx, cy)] * NUM_LANDMARKS
    pts[0] = (cx, cy + size / 2)   # WRIST
    pts[9] = (cx, cy - size / 2)   # MIDDLE_MCP -> distance == size
    sep = pinch * size
    pts[4] = (cx - sep / 2, cy)    # THUMB_TIP
    pts[8] = (cx + sep / 2, cy)    # INDEX_TIP -> distance == pinch*size
    return HandResult(landmarks=tuple(pts), confidence=1.0)


def test_centered_hand_gives_mid_base_and_updown():
    angles = to_angles(make_hand(center=(0.5, 0.5)), MapperConfig())
    assert angles["base"] == 90
    assert angles["updown"] == 90


def test_base_increases_left_to_right():
    cfg = MapperConfig()
    left = to_angles(make_hand(center=(0.2, 0.5)), cfg)["base"]
    right = to_angles(make_hand(center=(0.8, 0.5)), cfg)["base"]
    assert left == 0      # at/under box low edge
    assert right == 180   # at/over box high edge


def test_updown_is_inverted():
    cfg = MapperConfig()
    top = to_angles(make_hand(center=(0.5, 0.2)), cfg)["updown"]
    bottom = to_angles(make_hand(center=(0.5, 0.8)), cfg)["updown"]
    assert top == 180     # top of frame -> max
    assert bottom == 0    # bottom of frame -> min


def test_arm_from_hand_size():
    cfg = MapperConfig()
    near = to_angles(make_hand(size=cfg.size_max), cfg)["arm"]
    far = to_angles(make_hand(size=cfg.size_min), cfg)["arm"]
    assert near == 180
    assert far == 0


def test_gripper_from_pinch():
    cfg = MapperConfig()
    wide = to_angles(make_hand(pinch=cfg.pinch_max), cfg)["gripper"]
    closed = to_angles(make_hand(pinch=cfg.pinch_min), cfg)["gripper"]
    assert wide == 180
    assert closed == 0


def test_all_angles_clamped_0_180():
    cfg = MapperConfig()
    angles = to_angles(make_hand(center=(2.0, -1.0), size=99, pinch=99), cfg)
    for v in angles.values():
        assert 0 <= v <= 180
        assert isinstance(v, int)
