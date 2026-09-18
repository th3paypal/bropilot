from __future__ import annotations

from ..schema import Lecture, QuizItem
from ..utils.common import get_logger, to_timecode, truncate_words
from ..utils.llm import LlmClient
from ..utils.prompts import render
from .retrieval import build_chunks

log = get_logger(__name__)

DIFFICULTY_MIX = "половина recall, треть apply, остальное analyse"


class QuizAgent:
    def __init__(self, doc: Lecture, cfg) -> None:
        self.doc = doc
        self.cfg = cfg
        self.client = LlmClient(
            model=cfg.llm.text_model,
            max_tokens=cfg.quiz.max_tokens,
            temperature=cfg.quiz.temperature,
            concurrency=cfg.llm.concurrency,
            thinking=cfg.llm.get("thinking", "disabled"),
        )

    def generate(self, n_questions: int = 10) -> list[QuizItem]:
        groups = self._groups()
        if not groups:
            raise RuntimeError("no segments to build a quiz from")

        per_group = max(1, round(n_questions / len(groups)))
        log.info("generating %d questions across %d groups", n_questions, len(groups))

        batches = self.client.map(
            lambda g: self._ask_group(g, per_group), groups, desc="quiz"
        )

        items: list[QuizItem] = []
        for batch in batches:
            for raw in batch or []:
                item = self._validate(raw)
                if item:
                    items.append(item)

        items.sort(key=lambda i: i.ref_ts)
        return items[:n_questions]

    def _groups(self) -> list[list]:
        chunks = build_chunks(self.doc, self.cfg.retrieval.chunk_words,
                              self.cfg.retrieval.chunk_overlap)
        size = max(1, self.cfg.quiz.chunks_per_group)
        return [chunks[i : i + size] for i in range(0, len(chunks), size)]

    def _ask_group(self, group: list, n: int) -> list | None:
        prompt = render(
            "quiz",
            n=n,
            mix=DIFFICULTY_MIX,
            course=self.cfg.course.get("name"),
            chunks=[
                {
                    "tc": c.tc,
                    "title": c.title,
                    "text": truncate_words(c.text, self.cfg.quiz.max_words_per_chunk),
                    "board": c.board,
                }
                for c in group
            ],
        )
        payload = self.client.complete_json(prompt, system=self.cfg.quiz.system, prefill="[")
        return payload if isinstance(payload, list) else None

    def _validate(self, raw) -> QuizItem | None:
        if not isinstance(raw, dict):
            return None
        options = raw.get("options")
        if not isinstance(options, list) or len(options) != 4:
            return None
        options = [str(o).strip() for o in options]
        if len(set(options)) != 4 or any(not o for o in options):
            return None
        try:
            answer = int(raw["answer_index"])
            ref_ts = float(raw.get("ref_ts", 0.0))
        except (KeyError, TypeError, ValueError):
            return None
        if not 0 <= answer < 4:
            return None
        ref_ts = min(max(0.0, ref_ts), max(self.doc.duration, 0.0) or ref_ts)

        difficulty = raw.get("difficulty")
        if difficulty not in {"recall", "apply", "analyse"}:
            difficulty = "recall"

        return QuizItem(
            question=str(raw.get("question", "")).strip(),
            options=options,
            answer_index=answer,
            explanation=str(raw.get("explanation", "")).strip(),
            ref_ts=ref_ts,
            difficulty=difficulty,
        )


def format_quiz(items: list[QuizItem], with_answers: bool = False) -> str:
    lines: list[str] = []
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. [{to_timecode(item.ref_ts)}] ({item.difficulty}) {item.question}")
        for j, option in enumerate(item.options):
            lines.append(f"   {chr(ord('А') + j)}) {option}")
        if with_answers:
            lines.append(
                f"   -> {chr(ord('А') + item.answer_index)}. {item.explanation}"
            )
        lines.append("")
    return "\n".join(lines)
