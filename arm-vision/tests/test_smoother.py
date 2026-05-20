import math

from armvision.config import SmootherConfig, ServoSmoothing
from armvision.smoother import Smoother, ema_alpha


def test_ema_alpha_formula():
    # alpha = 1 - exp(-dt/tau); tau == dt -> 1 - 1/e
    assert math.isclose(ema_alpha(0.1, 0.1), 1 - math.exp(-1), rel_tol=1e-9)


def test_ema_alpha_clamped_to_unit_interval():
    assert ema_alpha(0.0, 0.1) == 1.0      # tau 0 -> no smoothing
    assert 0.0 <= ema_alpha(1.0, 0.0) <= 1.0


def single_cfg(tau=0.1, max_delta=8.0, deadzone=2.0):
    return SmootherConfig(per_servo={
        "base": ServoSmoothing(tau=tau, max_delta=max_delta, deadzone=deadzone),
    })


def test_first_frame_passes_through():
    s = Smoother(single_cfg())
    out = s.smooth({"base": 100.0}, dt=0.033)
    assert out["base"] == 100.0


def test_deadzone_ignores_small_wiggle():
    s = Smoother(single_cfg(deadzone=2.0))
    s.smooth({"base": 100.0}, dt=0.033)          # establish 100
    out = s.smooth({"base": 101.0}, dt=0.033)    # 1° < deadzone -> held
    assert out["base"] == 100.0


def test_rate_limit_caps_per_frame_change():
    s = Smoother(single_cfg(tau=0.0001, max_delta=8.0))  # alpha~1, EMA ~ raw
    s.smooth({"base": 100.0}, dt=0.033)
    out = s.smooth({"base": 200.0}, dt=0.033)
    assert out["base"] == 108.0  # capped to +8 from 100


def test_ema_pulls_toward_target():
    s = Smoother(single_cfg(tau=0.1, max_delta=999.0, deadzone=0.0))
    s.smooth({"base": 0.0}, dt=0.1)
    out = s.smooth({"base": 100.0}, dt=0.1)
    # alpha = 1-1/e ≈ 0.632 -> 63.2
    assert math.isclose(out["base"], 100 * (1 - math.exp(-1)), rel_tol=1e-6)
