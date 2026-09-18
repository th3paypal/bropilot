from __future__ import annotations

from pathlib import Path

from ..schema import BoardState, Lecture, Span
from ..utils.common import get_logger, resolve_artifact, to_timecode, truncate_words
from ..utils.llm import LlmClient, encode_image
from ..utils.prompts import prompt_digest, render
from .base import Context, Stage

log = get_logger(__name__)

PROMPT = "board_read"


class BoardReadingStage(Stage):
    name = "board_reading"
    config_keys = ("ocr", "llm", "course")

    def cache_salt(self) -> dict:
        return {"prompt": prompt_digest(PROMPT)}

    def apply(self, ctx: Context) -> Lecture:
        cfg = self.cfg.ocr
        doc = ctx.doc
        if not doc.boards:
            log.warning("no board states to read — skipping")
            return doc

        frames = {b.bid: resolve_artifact(b.frame_path, ctx.workdir, "frames")
                  for b in doc.boards}
        present = [b for b in doc.boards if frames[b.bid] is not None]
        if not present:
            raise RuntimeError(
                f"не найден ни один кадр доски из {len(doc.boards)} "
                f"(например, {doc.boards[0].frame_path}). Папку прогона, видимо, "
                f"перенесли без подпапки frames/. Пересоберите кадры: "
                f"`bropilot run <источник> --force`, либо удалите {ctx.cache_dir}."
            )
        if len(present) < len(doc.boards):
            log.warning("%d of %d board frames are missing — they will be skipped",
                        len(doc.boards) - len(present), len(doc.boards))

        client = LlmClient(
            model=self.cfg.llm.vision_model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            concurrency=self.cfg.llm.concurrency,
            thinking=self.cfg.llm.get("thinking", "disabled"),
        )

        log.info("reading %d board states with %s", len(present), self.cfg.llm.vision_model)
        results = client.map(
            lambda b: self._read_one(client, doc, b, frames[b.bid], cfg),
            present,
            desc="board_reading",
        )

        readable = 0
        for board, payload in zip(present, results):
            board.frame_path = str(frames[board.bid])
            if not payload:
                continue
            board.latex = (payload.get("latex") or "").strip() or None
            board.caption = (payload.get("caption") or "").strip() or None
            board.kind = payload.get("kind") if payload.get("kind") in {
                "handwriting", "slide", "mixed"
            } else "unknown"
            try:
                board.confidence = float(payload.get("confidence", 0.0))
            except (TypeError, ValueError):
                board.confidence = None
            if board.latex:
                readable += 1

        doc.boards = [
            b for b in doc.boards
            if b.latex or (b.confidence or 0) >= cfg.keep_confidence
        ]

        doc.meta["board_reading"] = {
            "read": readable,
            "kept": len(doc.boards),
            "frames_missing": len(frames) - len(present),
            **client.report(),
        }
        log.info("read %d/%d states, kept %d", readable, len(results), len(doc.boards))
        return doc

    def _read_one(self, client: LlmClient, doc: Lecture, board: BoardState,
                  frame: Path, cfg) -> dict | None:
        context = self._speech_around(doc, board, cfg.context_seconds, cfg.context_words)
        prompt = render(
            PROMPT,
            timecode=to_timecode(board.key_ts),
            context=context,
            course=self.cfg.course.get("name"),
        )
        blocks = [encode_image(frame), {"type": "text", "text": prompt}]
        payload = client.complete_json(blocks, system=cfg.system, prefill="{")
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _speech_around(doc: Lecture, board: BoardState, window: float, max_words: int) -> str:
        span = Span(
            start=max(0.0, board.span.start - window),
            end=board.span.end + window * 0.25,
        )
        text = " ".join(u.text.strip() for u in doc.utterances_in(span, min_overlap=0.3))
        return truncate_words(text, max_words)
