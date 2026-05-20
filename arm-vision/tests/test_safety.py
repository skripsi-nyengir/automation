from armvision.config import SafetyConfig
from armvision.safety import SafetyController, changed_servos


def test_changed_servos_returns_only_diffs():
    prev = {"base": 90, "updown": 90}
    new = {"base": 95, "updown": 90}
    assert changed_servos(prev, new) == {"base": 95}


def test_changed_servos_all_new_when_no_prev():
    assert changed_servos({}, {"base": 90}) == {"base": 90}


def cfg():
    return SafetyConfig(confidence_threshold=0.5, loss_frames=3)


def test_idle_clutch_blocks_sending():
    sc = SafetyController(cfg())  # starts idle
    d = sc.step(angles={"base": 100}, hand_present=True, confidence=0.9)
    assert d.send is False


def test_engaged_with_good_hand_sends():
    sc = SafetyController(cfg())
    sc.toggle_clutch()  # engage
    d = sc.step(angles={"base": 100}, hand_present=True, confidence=0.9)
    assert d.send is True
    assert d.angles == {"base": 100}


def test_low_confidence_counts_as_loss_after_n_frames():
    sc = SafetyController(cfg())
    sc.toggle_clutch()
    sc.step(angles={"base": 100}, hand_present=True, confidence=0.9)  # ok
    for _ in range(3):  # loss_frames == 3
        d = sc.step(angles={"base": 120}, hand_present=False, confidence=0.0)
    assert d.send is False  # frozen, holds last


def test_reengages_after_loss_without_snap():
    sc = SafetyController(cfg())
    sc.toggle_clutch()
    sc.step(angles={"base": 100}, hand_present=True, confidence=0.9)
    for _ in range(3):
        sc.step(angles={"base": 999}, hand_present=False, confidence=0.0)
    d = sc.step(angles={"base": 105}, hand_present=True, confidence=0.9)
    assert d.send is True
    assert d.angles == {"base": 105}
