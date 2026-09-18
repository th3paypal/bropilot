from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schema import Lecture
from ..utils.common import from_timecode, get_logger, to_timecode
from ..utils.llm import LlmClient
from ..utils.prompts import render
from .retrieval import Chunk, Retriever

log = get_logger(__name__)

_CITATION = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")


@dataclass
class Answer:
    text: str
    citations: list[float] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    sources: list[tuple[str, float]] = field(default_factory=list)

    def render(self) -> str:
        lines = [self.text.strip()]
        if self.sources:
            lines.append("\nИсточники в записи: " + ", ".join(tc for tc, _ in self.sources))
        if self.unsupported:
            lines.append(
                "\n[!] Метки, которых нет среди найденных фрагментов: "
                + ", ".join(self.unsupported)
            )
        return "\n".join(lines)


class QaAgent:
    def __init__(self, doc: Lecture, cfg) -> None:
        self.doc = doc
        self.cfg = cfg
        self.retriever = Retriever(doc, cfg.retrieval)
        self.client = LlmClient(
            model=cfg.llm.text_model,
            max_tokens=cfg.qa.max_tokens,
            temperature=cfg.qa.temperature,
            concurrency=1,
            thinking=cfg.llm.get("thinking", "disabled"),
        )

    def ask(self, question: str) -> Answer:
        hits = self.retriever.search(question)
        context = self._expand(hits)
        log.debug("retrieved %d chunks for %r", len(context), question[:60])

        prompt = render(
            "qa",
            question=question,
            chunks=[c.as_prompt_dict() for c in context],
            course=self.cfg.course.get("name"),
        )
        text = self.client.complete(prompt, system=self.cfg.qa.system)
        return self._verify(text, context, hits)

    def _expand(self, hits: list[tuple[Chunk, float]]) -> list[Chunk]:
        collected: dict[int, Chunk] = {}
        for chunk, _ in hits:
            for neighbour in self.retriever.neighbours(chunk, self.cfg.retrieval.neighbour_radius):
                collected[neighbour.cid] = neighbour
        ordered = sorted(collected.values(), key=lambda c: c.start)
        return ordered[: self.cfg.retrieval.max_context_chunks]

    def _verify(self, text: str, context: list[Chunk], hits) -> Answer:
        cited = _CITATION.findall(text)
        windows = [(c.start, c.start + self.cfg.retrieval.citation_tolerance) for c in context]

        valid: list[float] = []
        bogus: list[str] = []
        for tc in cited:
            seconds = from_timecode(tc)
            if any(lo - self.cfg.retrieval.citation_tolerance <= seconds <= hi for lo, hi in windows):
                valid.append(seconds)
            else:
                bogus.append(tc)
        if bogus:
            log.warning("answer cited timecodes outside the retrieved context: %s", bogus)

        return Answer(
            text=text.strip(),
            citations=sorted(set(valid)),
            unsupported=bogus,
            sources=[(to_timecode(c.start), round(s, 3)) for c, s in hits],
        )
