from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Array = np.ndarray


def to_gray(frame: Array) -> Array:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def estimate_polarity(gray: Array) -> int:
    lo, hi = np.percentile(gray, [2, 98])
    mid = 0.5 * (lo + hi)
    return 1 if float(np.median(gray)) >= mid else -1


def ink_mask(
    frame: Array,
    *,
    block: int = 35,
    offset: int = 12,
    denoise: bool = True,
) -> Array:
    gray = to_gray(frame)
    if estimate_polarity(gray) < 0:
        gray = 255 - gray

    block = block if block % 2 == 1 else block + 1
    mask = cv2.adaptiveThreshold(
        gray,
        maxValue=1,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block,
        C=offset,
    ).astype(bool)

    if denoise:
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
        ).astype(bool)
    return mask


def ink_ratio(mask: Array) -> float:
    return float(mask.mean())


def volatility_map(masks, smooth: int = 9) -> Array:
    n = len(masks)
    first = masks[0]
    if n < 2:
        return np.zeros(first.shape, dtype=np.float32)
    counts = np.zeros(first.shape, dtype=np.uint32)
    prev = first
    for i in range(1, n):
        cur = masks[i]
        counts += prev != cur
        prev = cur
    changes = (counts / float(n - 1)).astype(np.float32)
    if smooth > 1:
        k = smooth if smooth % 2 == 1 else smooth + 1
        changes = cv2.GaussianBlur(changes, (k, k), 0)
    return changes


class PackedMasks:
    def __init__(self, masks=()) -> None:
        self._data: list[Array] = []
        self.shape: tuple[int, int] | None = None
        for m in masks:
            self.append(m)

    def append(self, mask: Array) -> None:
        if self.shape is None:
            self.shape = mask.shape
        elif mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} != {self.shape}")
        self._data.append(np.packbits(mask, axis=None))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> Array:
        h, w = self.shape
        return np.unpackbits(self._data[i], count=h * w).reshape(h, w).astype(bool)

    def __iter__(self):
        for i in range(len(self._data)):
            yield self[i]

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self._data)


def content_mask(masks: list[Array], quantile: float = 0.985) -> Array:
    vol = volatility_map(masks)
    if float(vol.max()) <= 0:
        return np.ones_like(vol, dtype=bool)
    threshold = float(np.quantile(vol, quantile))
    keep = vol <= max(threshold, 1e-6)
    keep = cv2.morphologyEx(
        keep.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)
    ).astype(bool)
    return keep


@dataclass(frozen=True)
class Roi:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, frame: Array) -> Array:
        return frame[self.y0 : self.y1, self.x0 : self.x1]

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


def _profile_bounds(profile: Array, keep: float, pad: int, limit: int) -> tuple[int, int]:
    total = float(profile.sum())
    if total <= 0:
        return 0, limit
    cum = np.cumsum(profile) / total
    lo = int(np.searchsorted(cum, (1.0 - keep) / 2.0))
    hi = int(np.searchsorted(cum, 1.0 - (1.0 - keep) / 2.0))
    lo = max(0, lo - pad)
    hi = min(limit, hi + pad + 1)
    return (0, limit) if hi - lo < 8 else (lo, hi)


def board_roi(
    masks: list[Array],
    valid: Array | None = None,
    *,
    keep: float = 0.99,
    pad: int = 12,
) -> Roi:
    accum = np.zeros(masks[0].shape, dtype=np.float32)
    for m in masks:
        accum += m
    if valid is not None:
        accum *= valid
    h, w = accum.shape
    y0, y1 = _profile_bounds(accum.sum(axis=1), keep, pad, h)
    x0, x1 = _profile_bounds(accum.sum(axis=0), keep, pad, w)
    return Roi(x0=x0, y0=y0, x1=x1, y1=y1)


@dataclass(frozen=True)
class InkDelta:
    added: float
    removed: float
    coverage: float

    @property
    def net(self) -> float:
        return self.added - self.removed

    @property
    def churn(self) -> float:
        return self.added + self.removed


def ink_delta(prev: Array, cur: Array, valid: Array | None = None) -> InkDelta:
    if valid is not None:
        prev = prev & valid
        cur = cur & valid
    area = float(prev.size)
    added = float(np.count_nonzero(cur & ~prev)) / area
    removed = float(np.count_nonzero(prev & ~cur)) / area
    return InkDelta(added=added, removed=removed, coverage=float(cur.mean()))


def dilate(mask: Array, iterations: int = 1) -> Array:
    if iterations <= 0:
        return mask
    return cv2.dilate(
        mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=iterations
    ).astype(bool)
