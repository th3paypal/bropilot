from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from jinja2 import Environment

from ..exporting import compile_pdf
from ..schema import Lecture, Segment
from ..utils.common import (
    from_timecode,
    get_logger,
    to_timecode,
    truncate_words,
    video_link_prefix,
)
from ..utils.latex_guard import repair_or_escape
from ..utils.llm import LlmClient
from ..utils.prompts import prompt_digest, render
from .base import Context, Stage

log = get_logger(__name__)

PROMPT = "compose_section"
TEMPLATE = Path(__file__).resolve().parents[3] / "templates" / "lecture.tex.j2"

_BARE_TS = re.compile(r"\\ts\{(\d{1,2}:\d{2}(?::\d{2})?)\}")

DISCLAIMER = (
    "Конспект собран автоматически из видеозаписи: распознавание речи и записей "
    "с доски выполнено моделями и может содержать ошибки. Пометка "
    "\\bropilotunsure{X} означает неуверенное прочтение символа, "
    "\\bropilotfigure{...} — рисунок, который не был перенесён в текст. "
    "Метки на полях указывают время в исходной записи."
)


_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in text)


def template_env() -> Environment:
    return Environment(
        block_start_string="<%", block_end_string="%>",
        comment_start_string="<#", comment_end_string="#>",
        keep_trailing_newline=True,
        autoescape=False,
    )


def render_document(**kwargs) -> str:
    template = template_env().from_string(TEMPLATE.read_text(encoding="utf-8"))
    return template.render(**kwargs)


def with_seconds(body: str) -> str:
    return _BARE_TS.sub(
        lambda m: f"\\ts[{int(from_timecode(m.group(1)))}]{{{m.group(1)}}}", body
    )


def assemble_tex(doc: Lecture, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_document(
        title=latex_escape(doc.title or "Конспект лекции"),
        source=doc.source.replace("\n", " "),
        video_prefix=video_link_prefix(doc.source) or "",
        date=dt.date.today().strftime("%d.%m.%Y"),
        built_at=dt.datetime.now().isoformat(timespec="seconds"),
        disclaimer=DISCLAIMER,
        sections=[with_seconds(s.body) for s in doc.segments if s.body],
        appendix=None,
    )
    tex_path = out_dir / "notes.tex"
    tex_path.write_text(rendered, encoding="utf-8")
    log.info("wrote %s (%d sections)", tex_path, len(doc.segments))
    return tex_path


class ComposeStage(Stage):
    name = "compose"
    config_keys = ("compose", "llm", "course")

    def cache_salt(self) -> dict:
        return {
            "prompt": prompt_digest(PROMPT),
            "template": TEMPLATE.read_text(encoding="utf-8")[:4000],
        }

    def apply(self, ctx: Context) -> Lecture:
        cfg = self.cfg.compose
        doc = ctx.doc
        if not doc.segments:
            raise RuntimeError("segmentation stage must run before compose")

        client = LlmClient(
            model=self.cfg.llm.text_model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            concurrency=self.cfg.llm.concurrency,
            thinking=self.cfg.llm.get("thinking", "disabled"),
        )

        wave = max(1, int(cfg.wave_size))
        written: list[str] = []
        problems = 0

        for start in range(0, len(doc.segments), wave):
            batch = doc.segments[start : start + wave]
            results = client.map(
                lambda seg: self._write_section(client, doc, seg, list(written), cfg),
                batch,
                desc="compose",
            )
            for seg, payload in zip(batch, results):
                if not payload:
                    seg.title = seg.title or f"Фрагмент {to_timecode(seg.span.start)}"
                    seg.body = self._fallback_body(seg)
                    problems += 1
                    continue
                seg.title = (payload.get("title") or "").strip() or f"Раздел {seg.sid + 1}"
                body, report = repair_or_escape(payload.get("latex") or "")
                if not report.ok:
                    problems += 1
                    log.warning("segment %d: %s", seg.sid, "; ".join(report.errors))
                seg.body = body
                written.append(seg.title)

        tex_path = self._assemble(ctx, doc)
        build = compile_pdf(tex_path) if cfg.compile_pdf else None
        if build is not None and not build.ok:
            log.warning("PDF not built: %s", build.error)

        doc.meta["compose"] = {
            "sections": len(doc.segments),
            "sections_with_problems": problems,
            "tex": str(tex_path),
            "pdf": str(build.pdf) if build and build.pdf else None,
            "pdf_engine": build.engine if build else None,
            "pdf_error": (build.error or None) if build else "сборка PDF отключена",
            **client.report(),
        }
        return doc

    def _write_section(
        self, client: LlmClient, doc: Lecture, seg: Segment, written: list[str], cfg
    ) -> dict | None:
        boards = [b for b in doc.boards if b.bid in seg.board_ids and b.latex]
        prompt = render(
            PROMPT,
            index=seg.sid + 1,
            total=len(doc.segments),
            start_tc=to_timecode(seg.span.start),
            end_tc=to_timecode(seg.span.end),
            transcript=truncate_words(seg.transcript(doc), cfg.max_transcript_words),
            boards=[
                {
                    "tc": to_timecode(b.key_ts),
                    "latex": b.latex,
                    "caption": b.caption,
                    "confidence": round(b.confidence, 2) if b.confidence else None,
                }
                for b in boards[: cfg.max_boards_per_section]
            ],
            previous_titles=written[-cfg.title_context :] if written else [],
            course=self.cfg.course.get("name"),
        )
        payload = client.complete_json(prompt, system=cfg.system, prefill="{")
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _fallback_body(seg: Segment) -> str:
        return (
            f"\\section{{Фрагмент {to_timecode(seg.span.start)}}}\n"
            "\\begin{bropilotnote}\\small Раздел не удалось сгенерировать "
            "автоматически — см. запись лекции по метке на полях.\\end{bropilotnote}\n"
            f"\\ts{{{to_timecode(seg.span.start)}}}\n"
        )

    def _assemble(self, ctx: Context, doc: Lecture) -> Path:
        return assemble_tex(doc, ctx.subdir("output"))
