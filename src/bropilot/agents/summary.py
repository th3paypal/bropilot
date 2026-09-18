from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..schema import Lecture
from ..utils.common import get_logger, stable_hash, to_timecode, truncate_words
from ..utils.latex_md import latex_to_markdown
from ..utils.llm import LlmClient

log = get_logger(__name__)

LENGTHS = {
    "коротко": "5–7 пунктов, по одному предложению",
    "абзац": "один связный абзац на 120–180 слов",
    "подробно": "подробный пересказ по разделам, 400–600 слов, с ключевыми формулами",
}

SYSTEM = (
    "Ты пересказываешь конкретную лекцию по её конспекту. Используешь только "
    "материал конспекта, ничего не добавляешь от себя. Пишешь по-русски, формулы — "
    "в LaTeX внутри $...$."
)

PROMPT = """Ниже конспект лекции{course} по разделам; у каждого раздела метка времени начала.

{sections}

Задание: {task}
Формат: {length}.
После каждого пункта или утверждения ставь метку времени раздела в квадратных скобках, например [12:30].
Если по запрошенной теме в конспекте ничего нет — так и напиши.
Верни только текст пересказа в Markdown, без вступлений."""


def _sections_text(doc: Lecture, max_words: int) -> str:
    parts = []
    budget = max(200, max_words // max(1, len(doc.segments)))
    for seg in doc.segments:
        body = latex_to_markdown(seg.body or "", heading_level=4, timecode_fmt="")
        if not body.strip():
            body = seg.transcript(doc)
        parts.append(
            f"### [{to_timecode(seg.span.start)}] {seg.title or 'Раздел'}\n"
            + truncate_words(body, budget)
        )
    return "\n\n".join(parts)


def summarize(
    doc: Lecture,
    cfg,
    *,
    focus: str | None = None,
    length: str = "коротко",
    cache_dir: Path | None = None,
    max_words: int = 12000,
    on_usage: Callable[[dict], None] | None = None,
) -> str:
    key = stable_hash("summary", focus or "", length, [s.body for s in doc.segments])
    cache = Path(cache_dir) / "summaries.json" if cache_dir else None
    store: dict = {}
    if cache and cache.exists():
        store = json.loads(cache.read_text(encoding="utf-8"))
        if key in store:
            return store[key]["text"]

    task = (f"объясни тему «{focus}» так, как она изложена в этой лекции"
            if focus else "перескажи всю лекцию: главные идеи, определения и результаты")
    course = cfg.course.get("name")
    prompt = PROMPT.format(
        course=f" курса «{course}»" if course else "",
        sections=_sections_text(doc, max_words),
        task=task,
        length=LENGTHS.get(length, LENGTHS["коротко"]),
    )
    client = LlmClient(
        model=cfg.llm.text_model,
        max_tokens=4000,
        temperature=0.2,
        concurrency=1,
        thinking=cfg.llm.get("thinking", "disabled"),
    )
    text = client.complete(prompt, system=SYSTEM).strip()
    if on_usage is not None:
        on_usage(client.report())

    if cache:
        store[key] = {"focus": focus, "length": length, "text": text}
        cache.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    return text
