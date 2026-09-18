from __future__ import annotations

import re
from collections import Counter

from ..schema import Lecture, Span, Utterance
from ..utils.common import get_logger, to_timecode
from .base import Context, Stage

log = get_logger(__name__)

_SENTENCE_END = re.compile(r"[.!?…]['\"»)\]]?\s*$")
_WORD = re.compile(r"\w+", re.UNICODE)


def looks_degenerate(text: str, *, min_words: int = 6, ratio: float = 0.6) -> bool:
    tokens = _WORD.findall(text.lower())
    if len(tokens) < min_words:
        return False
    most_common = Counter(tokens).most_common(1)[0][1]
    return most_common / len(tokens) >= ratio


def rejection_reason(
    text: str,
    no_speech_prob: float | None,
    avg_logprob: float | None,
    *,
    max_no_speech: float,
    min_avg_logprob: float,
    hard_min_logprob: float | None = None,
) -> str | None:
    if not text.strip():
        return "empty"
    silent = (
        no_speech_prob is not None and no_speech_prob > max_no_speech
        and avg_logprob is not None and avg_logprob < min_avg_logprob
    )
    if silent:
        return "silence"
    if hard_min_logprob is not None and avg_logprob is not None and avg_logprob < hard_min_logprob:
        return "low_confidence"
    if looks_degenerate(text):
        return "repetition"
    return None


class AsrStage(Stage):
    name = "asr"
    config_keys = ("asr",)

    def apply(self, ctx: Context) -> Lecture:
        from faster_whisper import WhisperModel

        cfg = self.cfg.asr
        doc = ctx.doc
        if not doc.audio_path:
            raise RuntimeError("ingest stage must run before asr")

        model = WhisperModel(
            cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            download_root=str(ctx.subdir("models")),
        )
        log.info("transcribing with %s (%s)", cfg.model, cfg.compute_type)

        raw, info = model.transcribe(
            doc.audio_path,
            language=cfg.language or None,
            beam_size=cfg.beam_size,
            vad_filter=cfg.vad_filter,
            vad_parameters={"min_silence_duration_ms": cfg.min_silence_ms},
            condition_on_previous_text=cfg.condition_on_previous_text,
            initial_prompt=cfg.get("initial_prompt") or None,
        )

        kept: list = []
        reasons: Counter = Counter()
        total = float(getattr(info, "duration", 0.0) or doc.duration or 0.0)
        next_report = 60.0
        for seg in raw:
            if seg.end >= next_report:
                log.info("asr progress: %s / %s", to_timecode(seg.end), to_timecode(total))
                next_report = seg.end + 60.0
            reason = rejection_reason(
                seg.text,
                seg.no_speech_prob,
                seg.avg_logprob,
                max_no_speech=cfg.max_no_speech,
                min_avg_logprob=cfg.min_avg_logprob,
                hard_min_logprob=cfg.get("hard_min_logprob"),
            )
            if reason is None:
                kept.append(seg)
            else:
                reasons[reason] += 1
        dropped = sum(reasons.values())

        log.info("kept %d raw segments, discarded %d %s", len(kept), dropped, dict(reasons))
        doc.language = getattr(info, "language", cfg.language) or cfg.language
        doc.utterances = self._regroup(kept, cfg)
        doc.meta["asr"] = {
            "model": cfg.model,
            "raw_segments": len(kept) + dropped,
            "utterances": len(doc.utterances),
            "dropped": dropped,
            "dropped_by_reason": dict(reasons),
            "coverage": round(
                sum(float(seg.end - seg.start) for seg in kept) / max(doc.duration, 1.0), 3
            ),
        }
        return doc

    def _regroup(self, segments, cfg) -> list[Utterance]:
        groups: list[list] = []
        current: list = []

        for seg in segments:
            if not current:
                current = [seg]
                continue

            prev = current[-1]
            gap = seg.start - prev.end
            span_len = seg.end - current[0].start
            text_so_far = " ".join(s.text for s in current)

            hard_break = gap >= cfg.group_pause or span_len >= cfg.group_max_seconds
            soft_break = (
                _SENTENCE_END.search(text_so_far)
                and span_len >= cfg.group_min_seconds
            )
            if hard_break or soft_break:
                groups.append(current)
                current = [seg]
            else:
                current.append(seg)

        if current:
            groups.append(current)

        utterances: list[Utterance] = []
        for uid, group in enumerate(groups):
            text = re.sub(r"\s+", " ", " ".join(s.text.strip() for s in group)).strip()
            logprobs = [s.avg_logprob for s in group if s.avg_logprob is not None]
            utterances.append(
                Utterance(
                    uid=uid,
                    span=Span(start=float(group[0].start), end=float(group[-1].end)),
                    text=text,
                    avg_logprob=sum(logprobs) / len(logprobs) if logprobs else None,
                    no_speech_prob=max(
                        (s.no_speech_prob for s in group if s.no_speech_prob is not None),
                        default=None,
                    ),
                )
            )
        return utterances
