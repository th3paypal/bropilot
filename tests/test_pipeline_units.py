from __future__ import annotations

import numpy as np
import pytest

from bropilot.agents.retrieval import build_chunks
from bropilot.schema import BoardState, Lecture, Segment, Span, Utterance
from bropilot.stages.asr import looks_degenerate
from bropilot.stages.segmentation import cohesion_depth
from bropilot.utils.common import from_timecode, to_timecode
from bropilot.utils.latex_guard import check, repair_or_escape, sanitise
from bropilot.utils.llm import extract_json


class TestTimecodes:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "00:00"), (61, "01:01"), (754.2, "12:34"), (3600, "1:00:00"), (4210, "1:10:10")],
    )
    def test_format(self, seconds, expected):
        assert to_timecode(seconds) == expected

    @pytest.mark.parametrize("tc", ["00:00", "12:34", "1:10:10"])
    def test_roundtrip(self, tc):
        assert to_timecode(from_timecode(tc)) == tc


class TestSpan:
    def test_rejects_reversed_interval(self):
        with pytest.raises(ValueError):
            Span(start=10.0, end=3.0)

    def test_overlap_and_iou(self):
        a, b = Span(start=0, end=10), Span(start=5, end=15)
        assert a.overlap(b) == 5
        assert a.iou(b) == pytest.approx(5 / 15)
        assert a.overlap(Span(start=20, end=30)) == 0


class TestLatexGuard:
    def test_accepts_a_clean_fragment(self):
        frag = r"\section{Регуляризация}\ts{12:34} Пусть $\lambda > 0$. \[ Q(w) = \|Xw - y\|^2 \]"
        assert check(frag).ok

    def test_detects_unbalanced_dollars(self):
        assert not check(r"пусть $x = 1").ok

    def test_detects_unclosed_environment(self):
        assert not check(r"\begin{align} a &= b").ok

    def test_ignores_escaped_specials(self):
        assert check(r"стоимость 100\$ и 50\% от \{A\}").ok

    def test_rejects_preamble_commands(self):
        assert not check(r"\usepackage{amsmath} текст").ok

    def test_sanitise_unwraps_a_full_document(self):
        raw = "```latex\n\\documentclass{article}\\begin{document}\\section{A} x\\end{document}\n```"
        cleaned = sanitise(raw)
        assert cleaned.startswith(r"\section{A}")
        assert "documentclass" not in cleaned

    def test_sanitise_converts_double_dollar_display_math(self):
        assert sanitise("$$x^2$$") == r"\[x^2\]"

    def test_broken_fragment_falls_back_to_a_visible_block(self):
        body, report = repair_or_escape(r"\begin{align} a &= b")
        assert not report.ok
        assert body.startswith(r"\begin{quote}")
        assert check(body).ok, "the fallback itself must compile"


class TestJsonExtraction:
    def test_bare_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_object(self):
        assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}

    def test_object_with_preamble(self):
        assert extract_json('Вот результат:\n{"ok": true}\nготово') == {"ok": True}

    def test_raises_on_garbage(self):
        with pytest.raises(Exception):
            extract_json("никакого json здесь нет")


class TestAsrFilters:
    def test_flags_a_repetition_loop(self):
        assert looks_degenerate("спасибо спасибо спасибо спасибо спасибо спасибо")

    def test_leaves_normal_speech_alone(self):
        assert not looks_degenerate(
            "итак сегодня мы поговорим про градиентный спуск и его модификации"
        )

    def test_short_phrases_are_never_flagged(self):
        assert not looks_degenerate("да да")


class TestCohesionDepth:
    def test_peaks_at_a_vocabulary_shift(self):
        a = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (6, 1))
        b = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (6, 1))
        depth = cohesion_depth(np.vstack([a, b]), window=3)
        assert int(np.argmax(depth)) == 5

    def test_uniform_text_has_no_strong_boundary(self):
        vectors = np.tile(np.array([1.0, 0.0], dtype=np.float32), (10, 1))
        assert float(cohesion_depth(vectors, window=3).max()) == pytest.approx(0.0, abs=1e-6)


def _toy_lecture(n_utterances: int = 30) -> Lecture:
    utterances = [
        Utterance(uid=i, span=Span(start=i * 10.0, end=i * 10.0 + 9.0),
                  text=f"предложение номер {i} про модель и признаки " * 3)
        for i in range(n_utterances)
    ]
    doc = Lecture(source="toy", duration=n_utterances * 10.0, utterances=utterances)
    doc.boards = [
        BoardState(bid=0, span=Span(start=0, end=150), key_ts=20.0,
                   frame_path="/dev/null", latex=r"$y = Xw$"),
    ]
    doc.segments = [
        Segment(sid=0, span=Span(start=0, end=150),
                utterance_ids=list(range(15)), board_ids=[0], title="Часть 1"),
        Segment(sid=1, span=Span(start=150, end=300),
                utterance_ids=list(range(15, 30)), board_ids=[], title="Часть 2"),
    ]
    return doc


class TestChunking:
    def test_windows_respect_the_target_size(self):
        chunks = build_chunks(_toy_lecture(), target_words=100, overlap=20)
        assert len(chunks) > 2
        assert all(len(c.text.split()) < 400 for c in chunks)

    def test_chunks_stay_inside_their_segment(self):
        doc = _toy_lecture()
        for chunk in build_chunks(doc, target_words=100, overlap=20):
            segment = doc.segments[chunk.sid]
            assert segment.span.start <= chunk.start <= segment.span.end

    def test_board_latex_is_attached_to_its_segment(self):
        chunks = build_chunks(_toy_lecture(), target_words=100, overlap=20)
        assert any(c.board and "Xw" in c.board for c in chunks if c.sid == 0)
        assert all(c.board is None for c in chunks if c.sid == 1)

    def test_chunks_are_ordered_and_uniquely_identified(self):
        chunks = build_chunks(_toy_lecture(), target_words=80, overlap=20)
        assert [c.cid for c in chunks] == sorted(c.cid for c in chunks)
        assert len({c.cid for c in chunks}) == len(chunks)


class TestSerialisation:
    def test_lecture_roundtrips_through_disk(self, tmp_path):
        doc = _toy_lecture()
        path = tmp_path / "lecture.json"
        doc.save(path)
        restored = Lecture.load(path)
        assert restored.duration == doc.duration
        assert len(restored.utterances) == len(doc.utterances)
        assert restored.segments[0].title == "Часть 1"
