from __future__ import annotations

import numpy as np

from ..schema import Lecture, Segment, Span
from ..utils.common import get_logger, to_timecode
from .base import Context, Stage

log = get_logger(__name__)


def _encode(texts: list[str], cfg) -> np.ndarray:
    if cfg.encoder == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(cfg.encoder_model)
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        sublinear_tf=True,
        max_features=cfg.tfidf_max_features,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vec.fit_transform(texts).astype(np.float32).toarray()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def cohesion_depth(vectors: np.ndarray, window: int) -> np.ndarray:
    n = len(vectors)
    if n < 2:
        return np.zeros(max(0, n - 1), dtype=np.float32)

    cohesion = np.zeros(n - 1, dtype=np.float32)
    for i in range(n - 1):
        left = vectors[max(0, i + 1 - window) : i + 1].mean(axis=0)
        right = vectors[i + 1 : min(n, i + 1 + window)].mean(axis=0)
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        cohesion[i] = float(left @ right / denom) if denom > 0 else 0.0

    depth = np.zeros_like(cohesion)
    for i in range(len(cohesion)):
        left_peak = cohesion[i]
        j = i
        while j > 0 and cohesion[j - 1] >= cohesion[j]:
            j -= 1
            left_peak = cohesion[j]
        right_peak = cohesion[i]
        j = i
        while j < len(cohesion) - 1 and cohesion[j + 1] >= cohesion[j]:
            j += 1
            right_peak = cohesion[j]
        depth[i] = (left_peak - cohesion[i]) + (right_peak - cohesion[i])

    span = float(depth.max() - depth.min())
    return (depth - depth.min()) / span if span > 1e-8 else np.zeros_like(depth)


class SegmentationStage(Stage):
    name = "segmentation"
    config_keys = ("segmentation",)

    def apply(self, ctx: Context) -> Lecture:
        cfg = self.cfg.segmentation
        doc = ctx.doc
        utterances = doc.utterances
        if len(utterances) < 4:
            doc.segments = [self._whole_lecture(doc)]
            return doc

        vectors = _encode([u.text for u in utterances], cfg)
        depth = cohesion_depth(vectors, cfg.cohesion_window)

        visual = self._visual_prior(doc, cfg)

        score = cfg.w_visual * visual + cfg.w_lexical * depth
        boundaries = self._select(score, utterances, cfg)
        log.info(
            "segmentation: %d gaps -> %d boundaries (%d board states)",
            len(score), len(boundaries), len(doc.boards),
        )

        doc.segments = self._build(doc, boundaries, score)
        doc.meta["segmentation"] = {
            "segments": len(doc.segments),
            "encoder": cfg.encoder,
            "mean_seconds": round(
                float(np.mean([s.span.duration for s in doc.segments])), 1
            ),
        }
        for s in doc.segments:
            log.debug("segment %d: %s–%s", s.sid,
                      to_timecode(s.span.start), to_timecode(s.span.end))
        return doc

    def _visual_prior(self, doc: Lecture, cfg) -> np.ndarray:
        gaps = np.array(
            [0.5 * (doc.utterances[i].span.end + doc.utterances[i + 1].span.start)
             for i in range(len(doc.utterances) - 1)],
            dtype=np.float32,
        )
        prior = np.zeros_like(gaps)
        if not doc.boards:
            return prior
        onsets = np.array([b.span.start for b in doc.boards], dtype=np.float32)
        sigma = float(cfg.visual_sigma)
        for onset in onsets:
            prior = np.maximum(prior, np.exp(-0.5 * ((gaps - onset) / sigma) ** 2))
        return prior

    def _select(self, score: np.ndarray, utterances, cfg) -> list[int]:
        min_gap = float(cfg.min_seconds)
        chosen: list[int] = []
        order = np.argsort(-score)

        def time_at(i: int) -> float:
            return utterances[i].span.end

        for idx in order:
            if score[idx] < cfg.min_score:
                break
            t = time_at(int(idx))
            if all(abs(t - time_at(c)) >= min_gap for c in chosen):
                chosen.append(int(idx))

        chosen.sort()

        result: list[int] = []
        prev_time = utterances[0].span.start
        for idx in chosen + [len(utterances) - 1]:
            while time_at(idx) - prev_time > cfg.max_seconds:
                window = [
                    j for j in range(len(score))
                    if prev_time + min_gap <= time_at(j) <= prev_time + cfg.max_seconds
                ]
                if not window:
                    break
                best = max(window, key=lambda j: score[j])
                result.append(best)
                prev_time = time_at(best)
            if idx in chosen:
                result.append(idx)
                prev_time = time_at(idx)
        return sorted(set(result))

    def _build(self, doc: Lecture, boundaries: list[int], score: np.ndarray) -> list[Segment]:
        edges = [0] + [b + 1 for b in boundaries] + [len(doc.utterances)]
        edges = sorted(set(e for e in edges if 0 <= e <= len(doc.utterances)))

        segments: list[Segment] = []
        for sid, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if hi <= lo:
                continue
            chunk = doc.utterances[lo:hi]
            span = Span(start=chunk[0].span.start, end=chunk[-1].span.end)
            segments.append(
                Segment(
                    sid=sid,
                    span=span,
                    utterance_ids=[u.uid for u in chunk],
                    board_ids=[b.bid for b in doc.boards_in(span)],
                    boundary_score=float(score[hi - 2]) if 1 < hi <= len(score) + 1 else 0.0,
                )
            )
        return segments

    @staticmethod
    def _whole_lecture(doc: Lecture) -> Segment:
        span = Span(start=0.0, end=max(doc.duration, 1.0))
        return Segment(
            sid=0,
            span=span,
            utterance_ids=[u.uid for u in doc.utterances],
            board_ids=[b.bid for b in doc.boards],
        )
