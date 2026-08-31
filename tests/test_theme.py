"""Theme tests.

The colour work is computable, so it is computed rather than eyeballed. These
tests are what stop a future palette edit from quietly making a shaded table cell
unreadable.
"""

import numpy as np
import pytest

from ecognomy.viewer import theme
from ecognomy.viewer.theme import _RAMPS, shade

INK = {"light": "#0b0b0b", "dark": "#ffffff"}
WCAG_BODY_TEXT = 4.5


def _relative_luminance(hex_colour: str) -> float:
    channels = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255.0
    linear = np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])


def _contrast(fg: str, bg: str) -> float:
    a, b = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("mode", sorted(_RAMPS))
@pytest.mark.parametrize("hue", ("green", "blue", "orange"))
def test_every_ramp_step_keeps_text_readable(mode, hue):
    """A shaded cell still has a number in it, so every step must carry ink."""
    for step in _RAMPS[mode][hue]:
        ratio = _contrast(INK[mode], step)
        assert ratio >= WCAG_BODY_TEXT, f"{mode}/{hue} {step} is only {ratio:.2f}:1"


@pytest.mark.parametrize("mode", sorted(_RAMPS))
@pytest.mark.parametrize("hue", ("green", "blue", "orange"))
def test_ramps_are_monotone_in_lightness(mode, hue):
    """Sequential means one hue running light to dark, with no reversals — a
    ramp that doubles back encodes magnitude ambiguously."""
    lums = [_relative_luminance(c) for c in _RAMPS[mode][hue]]
    ordered = sorted(lums, reverse=(mode == "light"))
    assert lums == pytest.approx(ordered, abs=1e-9)


def test_shade_is_monotone_and_bounded():
    steps = _RAMPS[theme.MODE]["green"]
    seen = [shade(v, 1.0, "green") for v in np.linspace(0.01, 1.0, 40)]
    indices = [steps.index(c) for c in seen]
    assert indices == sorted(indices), "darker must mean larger, always"
    assert shade(5.0, 1.0, "green") == steps[-1], "values above the ceiling clamp"


def test_shade_returns_nothing_for_absent_values():
    """Zero is not a small amount, it is an absence — it should not read as the
    palest tint of 'present'."""
    assert shade(0.0, 1.0, "green") == "transparent"
    assert shade(float("nan"), 1.0, "green") == "transparent"
    assert shade(1.0, 0.0, "green") == "transparent"


def test_categorical_colours_never_cycle():
    """A ninth series must not silently reuse slot 1's hue."""
    n = len(theme.P["series"])
    assert theme.series_color(0) != theme.series_color(n)
    assert theme.series_color(n) == theme.P["text_muted"]
