from __future__ import annotations

import numpy as np
import pytest

from bropilot.utils.imaging import (
    board_roi,
    content_mask,
    dilate,
    estimate_polarity,
    ink_delta,
    ink_mask,
    ink_ratio,
    volatility_map,
)

H, W = 240, 320


def blank(dark: bool = False) -> np.ndarray:
    value = 20 if dark else 235
    return np.full((H, W, 3), value, dtype=np.uint8)


def write(frame: np.ndarray, rows: int, dark_ink: bool = True) -> np.ndarray:
    out = frame.copy()
    ink = 25 if dark_ink else 240
    for r in range(rows):
        y = 30 + r * 18
        out[y : y + 3, 40 : 40 + 200] = ink
    return out


class TestPolarity:
    def test_dark_ink_on_light_canvas(self):
        assert estimate_polarity(write(blank(), 4)[:, :, 0]) == 1

    def test_light_ink_on_dark_canvas(self):
        frame = write(blank(dark=True), 4, dark_ink=False)
        assert estimate_polarity(frame[:, :, 0]) == -1


class TestInkMask:
    def test_blank_frame_has_almost_no_ink(self):
        assert ink_ratio(ink_mask(blank())) < 0.02

    def test_writing_increases_ink(self):
        few = ink_ratio(ink_mask(write(blank(), 2)))
        many = ink_ratio(ink_mask(write(blank(), 8)))
        assert many > few > 0.0

    def test_polarity_invariance(self):
        light = ink_ratio(ink_mask(write(blank(), 6)))
        dark = ink_ratio(ink_mask(write(blank(dark=True), 6, dark_ink=False)))
        assert light == pytest.approx(dark, abs=0.01)


class TestInkDelta:
    def test_writing_is_addition_without_removal(self):
        prev = dilate(ink_mask(write(blank(), 3)))
        cur = dilate(ink_mask(write(blank(), 6)))
        d = ink_delta(prev, cur)
        assert d.added > 0.005
        assert d.removed < 0.002, "adding lines must not register as erasure"

    def test_erasing_is_removal(self):
        prev = dilate(ink_mask(write(blank(), 8)))
        cur = dilate(ink_mask(blank()))
        d = ink_delta(prev, cur)
        assert d.removed > 0.005
        assert d.removed > d.added

    def test_slide_change_shows_both_directions(self):
        prev = ink_mask(write(blank(), 6))
        other = blank()
        other[100:200, 40:280] = 20
        d = ink_delta(dilate(prev), dilate(ink_mask(other)))
        assert d.added > 0 and d.removed > 0
        assert d.churn > d.added

    def test_identical_frames_have_zero_delta(self):
        m = ink_mask(write(blank(), 5))
        d = ink_delta(m, m)
        assert d.added == 0.0 and d.removed == 0.0


class TestVolatility:
    def test_constantly_changing_region_is_flagged(self):
        masks = []
        rng = np.random.default_rng(0)
        for i in range(20):
            frame = write(blank(), 5)
            frame[10:60, 240:310] = rng.integers(0, 255, (50, 70, 3), dtype=np.uint8)
            masks.append(ink_mask(frame))

        vol = volatility_map(masks)
        assert vol[10:60, 240:310].mean() > vol[30:120, 40:240].mean()

        keep = content_mask(masks, quantile=0.9)
        assert keep[30:45, 60:200].mean() > keep[20:55, 250:305].mean()


class TestRoi:
    def test_roi_tracks_where_ink_is(self):
        masks = []
        for _ in range(10):
            frame = blank()
            frame[80:160, 100:220] = 25
            masks.append(ink_mask(frame))
        roi = board_roi(masks, keep=0.98, pad=5)
        assert roi.y0 < 90 and roi.y1 > 150
        assert roi.x0 < 110 and roi.x1 > 210
        assert roi.height < H and roi.width < W

    def test_roi_degrades_to_full_frame_when_empty(self):
        roi = board_roi([ink_mask(blank()) for _ in range(3)])
        assert roi.width > 0 and roi.height > 0
