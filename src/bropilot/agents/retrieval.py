from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import Lecture
from ..utils.common import get_logger, to_timecode

log = get_logger(__name__)


@dataclass
class Chunk:
    cid: int
    sid: int
    start: float
    title: str
    text: str
    board: str | None = None

    @property
    def tc(self) -> str:
        return to_timecode(self.start)

    def searchable(self) -> str:
        parts = [self.title, self.text]
        if self.board:
            parts.append(self.board)
        return "\n".join(p for p in parts if p)

    def as_prompt_dict(self) -> dict:
        return {"tc": self.tc, "title": self.title, "text": self.text, "board": self.board}


def build_chunks(doc: Lecture, target_words: int = 220, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    by_id = {u.uid: u for u in doc.utterances}
    cid = 0

    for seg in doc.segments:
        boards = [b for b in doc.boards if b.bid in seg.board_ids and b.latex]
        board_text = "\n\n".join(b.latex for b in boards) or None

        utterances = [by_id[i] for i in seg.utterance_ids if i in by_id]
        window: list = []
        words = 0
        for utt in utterances:
            window.append(utt)
            words += utt.words
            if words >= target_words:
                chunks.append(
                    Chunk(
                        cid=cid, sid=seg.sid,
                        start=window[0].span.start,
                        title=seg.title or f"Фрагмент {to_timecode(seg.span.start)}",
                        text=" ".join(u.text.strip() for u in window),
                        board=board_text,
                    )
                )
                cid += 1
                carry, carried = [], 0
                for u in reversed(window):
                    if carried >= overlap:
                        break
                    carry.insert(0, u)
                    carried += u.words
                window, words = carry, carried

        if window:
            chunks.append(
                Chunk(
                    cid=cid, sid=seg.sid,
                    start=window[0].span.start,
                    title=seg.title or f"Фрагмент {to_timecode(seg.span.start)}",
                    text=" ".join(u.text.strip() for u in window),
                    board=board_text,
                )
            )
            cid += 1

    log.debug("built %d chunks from %d segments", len(chunks), len(doc.segments))
    return chunks


class Retriever:
    def __init__(self, doc: Lecture, cfg) -> None:
        self.doc = doc
        self.cfg = cfg
        self.chunks = build_chunks(doc, cfg.chunk_words, cfg.chunk_overlap)
        if not self.chunks:
            raise RuntimeError("nothing to index — run the pipeline first")

        corpus = [c.searchable() for c in self.chunks]
        self._fit_sparse(corpus)
        self._dense = self._fit_dense(corpus) if cfg.use_dense else None

    def _fit_sparse(self, corpus: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_features=60_000
        )
        matrix = self._vectorizer.fit_transform(corpus).astype(np.float32).toarray()
        self._sparse = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)

    def _fit_dense(self, corpus: list[str]) -> np.ndarray | None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.warning("sentence-transformers unavailable — sparse retrieval only")
            return None
        self._encoder = SentenceTransformer(self.cfg.dense_model)
        return np.asarray(
            self._encoder.encode(corpus, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def search(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        k = k or self.cfg.top_k
        q_sparse = self._vectorizer.transform([query]).astype(np.float32).toarray()[0]
        norm = np.linalg.norm(q_sparse)
        q_sparse = q_sparse / norm if norm > 0 else q_sparse
        score = self._sparse @ q_sparse

        if self._dense is not None:
            q_dense = np.asarray(
                self._encoder.encode([query], normalize_embeddings=True,
                                     show_progress_bar=False)[0],
                dtype=np.float32,
            )
            score = (1 - self.cfg.dense_weight) * score + self.cfg.dense_weight * (
                self._dense @ q_dense
            )

        order = np.argsort(-score)[: max(k, 1)]
        picked = [(self.chunks[int(i)], float(score[int(i)])) for i in order]
        return [(c, s) for c, s in picked if s > self.cfg.min_score] or picked[:1]

    def neighbours(self, chunk: Chunk, radius: int = 1) -> list[Chunk]:
        lo = max(0, chunk.cid - radius)
        hi = min(len(self.chunks), chunk.cid + radius + 1)
        return self.chunks[lo:hi]
