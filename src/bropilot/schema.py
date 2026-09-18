from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Span(Base):
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)

    @field_validator("end")
    @classmethod
    def _ordered(cls, v: float, info) -> float:
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"end={v} precedes start={start}")
        return v

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlap(self, other: "Span") -> float:
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))

    def iou(self, other: "Span") -> float:
        inter = self.overlap(other)
        union = self.duration + other.duration - inter
        return inter / union if union > 0 else 0.0


class Utterance(Base):
    uid: int
    span: Span
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    @property
    def words(self) -> int:
        return len(self.text.split())


BoardKind = Literal["handwriting", "slide", "mixed", "unknown"]


class BoardState(Base):
    bid: int
    span: Span
    key_ts: float
    frame_path: str
    ink_ratio: float = 0.0
    kind: BoardKind = "unknown"
    latex: str | None = None
    caption: str | None = None
    confidence: float | None = None


class Segment(Base):
    sid: int
    span: Span
    title: str = ""
    utterance_ids: list[int] = Field(default_factory=list)
    board_ids: list[int] = Field(default_factory=list)
    boundary_score: float = 0.0
    body: str | None = None

    def transcript(self, doc: "Lecture") -> str:
        by_id = {u.uid: u for u in doc.utterances}
        return " ".join(by_id[i].text.strip() for i in self.utterance_ids if i in by_id)


class QuizItem(Base):
    question: str
    options: list[str]
    answer_index: int
    explanation: str
    ref_ts: float
    difficulty: Literal["recall", "apply", "analyse"] = "recall"


class Lecture(Base):
    source: str
    title: str = ""
    duration: float = 0.0
    language: str = "ru"
    audio_path: str | None = None
    video_path: str | None = None
    utterances: list[Utterance] = Field(default_factory=list)
    boards: list[BoardState] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    quiz: list[QuizItem] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def utterances_in(self, span: Span, min_overlap: float = 0.5) -> list[Utterance]:
        out = []
        for u in self.utterances:
            if u.span.duration <= 0:
                continue
            if u.span.overlap(span) / u.span.duration >= min_overlap:
                out.append(u)
        return out

    def boards_in(self, span: Span) -> list[BoardState]:
        return [b for b in self.boards if span.start <= b.key_ts < span.end]

    def full_text(self) -> str:
        return " ".join(u.text.strip() for u in self.utterances)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Lecture":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def merge_spans(spans: Sequence[Span]) -> Span:
    if not spans:
        raise ValueError("cannot merge an empty sequence of spans")
    return Span(start=min(s.start for s in spans), end=max(s.end for s in spans))
