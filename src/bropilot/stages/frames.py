from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..schema import BoardState, Lecture, Span
from ..utils.common import get_logger, to_timecode
from ..utils.imaging import (
    PackedMasks,
    Roi,
    board_roi,
    content_mask,
    dilate,
    ink_delta,
    ink_mask,
)
from .base import Context, Stage

log = get_logger(__name__)


def _iter_frames(path: str, step_seconds: float, scale_width: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps * step_seconds)))
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    if scale_width and w > scale_width:
                        new_h = int(round(h * scale_width / w))
                        frame = cv2.resize(
                            frame, (scale_width, new_h), interpolation=cv2.INTER_AREA
                        )
                    yield idx / fps, frame
            idx += 1
    finally:
        cap.release()


class BoardTrackingStage(Stage):
    name = "boards"
    config_keys = ("vision",)

    def apply(self, ctx: Context) -> Lecture:
        cfg = self.cfg.vision
        doc = ctx.doc
        if not doc.video_path:
            raise RuntimeError("ingest stage must run before board tracking")

        frames_dir = ctx.subdir("frames")
        for stale in frames_dir.glob("board_*.png"):
            stale.unlink()

        timestamps: list[float] = []
        masks = PackedMasks()
        raw_frames: dict[int, np.ndarray] = {}
        keep_full = bool(cfg.keep_frames_in_memory)

        log.info("sampling frames every %.1fs", cfg.sample_step)
        for i, (ts, frame) in enumerate(
            _iter_frames(doc.video_path, cfg.sample_step, cfg.scale_width)
        ):
            if i and i % 300 == 0:
                log.info("boards progress: %s / %s sampled", to_timecode(ts), to_timecode(doc.duration))
            timestamps.append(ts)
            masks.append(ink_mask(frame, block=cfg.threshold_block, offset=cfg.threshold_offset))
            if keep_full:
                raw_frames[i] = frame

        if len(masks) < 3:
            raise RuntimeError("too few frames sampled — is the video readable?")
        log.info("sampled %d frames", len(masks))

        valid = content_mask(masks, quantile=cfg.static_quantile)
        log.info("stable content region covers %.0f%% of the frame", 100 * valid.mean())

        roi = board_roi(masks, valid, keep=cfg.roi_keep, pad=cfg.roi_pad)
        log.info("board ROI: %s", roi.as_tuple())
        valid_roi = roi.crop(valid)
        masks = PackedMasks(roi.crop(m) & valid_roi for m in masks)

        cuts = self._find_cuts(masks, cfg)
        log.info("detected %d board states", len(cuts))

        boards = self._materialise(
            doc, cuts, masks, timestamps, roi, frames_dir, raw_frames, cfg
        )
        doc.boards = boards
        doc.meta["vision"] = {
            "sampled_frames": len(masks),
            "roi": roi.as_tuple(),
            "states": len(boards),
            "stable_fraction": round(float(valid.mean()), 4),
            "mask_memory_mb": round(masks.nbytes / 2**20, 1),
        }
        return doc

    def _find_cuts(self, masks: list[np.ndarray], cfg) -> list[tuple[int, int]]:
        boundaries = [0]
        prev = dilate(masks[0], cfg.dilate_iters)

        for i in range(1, len(masks)):
            cur = dilate(masks[i], cfg.dilate_iters)
            d = ink_delta(prev, cur)
            erased = d.removed > cfg.tau_erase
            appeared = d.added > cfg.tau_jump
            if erased or appeared:
                boundaries.append(i)
            prev = cur

        boundaries.append(len(masks))
        spans = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

        min_len = max(1, int(round(cfg.min_state_seconds / cfg.sample_step)))
        merged: list[tuple[int, int]] = []
        for span in spans:
            if merged and span[1] - span[0] < min_len:
                merged[-1] = (merged[-1][0], span[1])
            else:
                merged.append(span)
        return merged

    def _materialise(
        self,
        doc: Lecture,
        cuts: list[tuple[int, int]],
        masks: list[np.ndarray],
        timestamps: list[float],
        roi: Roi,
        frames_dir: Path,
        cached: dict[int, np.ndarray],
        cfg,
    ) -> list[BoardState]:
        cap = None if cached else cv2.VideoCapture(doc.video_path)
        boards: list[BoardState] = []
        bid = 0
        try:
            for lo, hi in cuts:
                coverage = [float(masks[i].mean()) for i in range(lo, hi)]
                best_local = int(np.argmax(np.array(coverage) + 1e-9 * np.arange(hi - lo)))
                best = lo + best_local
                if coverage[best_local] < cfg.min_ink_ratio:
                    continue

                key_ts = timestamps[best]
                frame = cached.get(best)
                if frame is None:
                    frame = self._seek(cap, key_ts)
                if frame is None:
                    continue

                crop = self._to_roi(frame, roi, cfg.scale_width)
                out = frames_dir / f"board_{bid:03d}_{int(key_ts):06d}.png"
                cv2.imwrite(str(out), crop, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                end_ts = timestamps[hi] if hi < len(timestamps) else doc.duration
                boards.append(
                    BoardState(
                        bid=bid,
                        span=Span(start=timestamps[lo], end=max(end_ts, timestamps[lo])),
                        key_ts=key_ts,
                        frame_path=str(out),
                        ink_ratio=round(coverage[best_local], 5),
                    )
                )
                log.debug("state %d @ %s (ink %.3f)", bid, to_timecode(key_ts), coverage[best_local])
                bid += 1
        finally:
            if cap is not None:
                cap.release()
        return boards

    @staticmethod
    def _seek(cap, ts: float):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None

    @staticmethod
    def _to_roi(frame: np.ndarray, roi: Roi, scale_width: int) -> np.ndarray:
        h, w = frame.shape[:2]
        factor = (w / scale_width) if (scale_width and w > scale_width) else 1.0
        x0 = max(0, int(roi.x0 * factor))
        y0 = max(0, int(roi.y0 * factor))
        x1 = min(w, int(round(roi.x1 * factor)))
        y1 = min(h, int(round(roi.y1 * factor)))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return frame
        return frame[y0:y1, x0:x1]
