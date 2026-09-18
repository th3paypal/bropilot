from __future__ import annotations

from types import SimpleNamespace

import pytest

from bropilot.exporting import available_engine, compile_pdf
from bropilot.stages.asr import rejection_reason
from bropilot.stages.compose import latex_escape, render_document, with_seconds
from bropilot.utils import llm as llm_mod
from bropilot.utils.latex_guard import repair_or_escape


def _render(**over):
    kwargs = dict(
        title="Лекция", source="lecture.mp4", video_prefix="",
        date="01.01.2026", built_at="now",
        disclaimer="Тест", sections=[r"\section{Раздел}\ts[60]{01:00} Текст $x^2$."],
        appendix=None,
    )
    kwargs.update(over)
    return render_document(**kwargs)


def test_template_parses_and_keeps_latex_braces():
    tex = _render()
    assert r"\newcommand{\ts}[2][]{%" in tex
    assert r"\underline{#1}" in tex
    assert r"\section{Раздел}" in tex
    assert "\\begin{document}" in tex and tex.rstrip().endswith("\\end{document}")


def test_timecodes_become_deep_links_only_for_video_urls():
    with_url = _render(video_prefix="https://youtu.be/abc?t=")
    assert r"\renewcommand{\bropilotvideo}{https://youtu.be/abc?t=}" in with_url
    assert r"\renewcommand{\bropilotvideo}" not in _render(video_prefix="")


def test_bare_note_command_becomes_an_environment():
    body, report = repair_or_escape(
        "\\begin{definition}Опр.\\end{definition}\n\n"
        "\\note\nЗначок $:=$ — присваивание.\n\nОбычный абзац."
    )
    assert "\\begin{note}" in body and "\\end{note}" in body
    assert "\\note\n" not in body
    assert "Обычный абзац." in body.split("\\end{note}")[1]
    assert report.ok


def test_note_normalisation_leaves_correct_latex_alone():
    ready = "\\begin{note}Уже окружение.\\end{note}"
    assert repair_or_escape(ready)[0] == ready
    assert "\\bropilotfigure{график}" in repair_or_escape("\\bropilotfigure{график}")[0]


def test_with_seconds_adds_the_optional_argument():
    assert with_seconds(r"a \ts{12:34} b") == r"a \ts[754]{12:34} b"
    assert with_seconds(r"\ts{1:02:03}") == r"\ts[3723]{1:02:03}"
    assert with_seconds(r"\ts[754]{12:34}") == r"\ts[754]{12:34}"


def test_template_optional_blocks():
    assert "bropilotnote" in _render(disclaimer="x").split("\\begin{document}")[1]
    body = _render(disclaimer=None).split("\\begin{document}")[1]
    assert "\\begin{bropilotnote}" not in body
    assert "\\appendix" in _render(appendix="Приложение")


def test_latex_escape_title():
    assert latex_escape("ML_1 & 50% #4") == r"ML\_1 \& 50\% \#4"
    assert latex_escape("a\\b") == r"a\textbackslash{}b"


@pytest.mark.skipif(available_engine() is None, reason="no LaTeX toolchain")
def test_template_compiles(tmp_path):
    tex = _render(title=latex_escape("Лекция 4 | ML_1"),
                  video_prefix="https://youtu.be/abc?t=",
                  sections=[
        r"\section{Спуск}\ts[10]{00:10} $w \in \R^d$, $\eps>0$."
        r"\begin{definition}Определение.\end{definition}"
        r"\begin{proof}Док.\end{proof}\bropilotfigure{график}"
    ])
    (tmp_path / "notes.tex").write_text(tex, encoding="utf-8")
    result = compile_pdf(tmp_path / "notes.tex")
    assert result.ok, result.log_tail or result.error
    assert result.pdf.stat().st_size > 1000
    payload = result.pdf.read_bytes()
    assert b"youtu.be/abc?t=10s" in payload or _links_in_streams(payload)


def _links_in_streams(payload: bytes) -> bool:
    import re
    import zlib

    for match in re.finditer(rb"stream\r?\n", payload):
        chunk = payload[match.end(): payload.find(b"endstream", match.end())]
        try:
            if b"youtu.be/abc?t=10s" in zlib.decompress(chunk):
                return True
        except zlib.error:
            continue
    return False


KW = dict(max_no_speech=0.6, min_avg_logprob=-1.0, hard_min_logprob=-1.6)


def test_clean_speech_with_high_nospeech_head_is_kept():
    assert rejection_reason("Градиентный спуск сходится", 0.85, -0.2, **KW) is None


def test_silence_needs_both_signals():
    assert rejection_reason("Спасибо за просмотр", 0.9, -1.3, **KW) == "silence"
    assert rejection_reason("нормальная речь", 0.1, -1.3, **KW) is None


def test_hard_floor_and_repetition():
    assert rejection_reason("что-то", 0.1, -2.0, **KW) == "low_confidence"
    assert rejection_reason("да да да да да да да да", 0.1, -0.3, **KW) == "repetition"
    assert rejection_reason("   ", 0.1, -0.3, **KW) == "empty"


class _FakeMessages:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **params):
        self.calls.append(params)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
        )


def _client(model: str, reply: str, monkeypatch) -> tuple[llm_mod.LlmClient, _FakeMessages]:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    client = llm_mod.LlmClient(model, temperature=0.3, thinking="disabled")
    fake = _FakeMessages(reply)
    client._client = SimpleNamespace(messages=fake)
    return client, fake


def test_modern_model_request_has_no_prefill_or_temperature(monkeypatch):
    client, fake = _client("claude-sonnet-5", '{"title": "T", "latex": "x"}', monkeypatch)
    out = client.complete_json("prompt", prefill="{")
    params = fake.calls[0]
    assert out == {"title": "T", "latex": "x"}
    assert "temperature" not in params
    assert params["messages"][-1]["role"] == "user"
    assert params["thinking"] == {"type": "disabled"}


def test_legacy_model_keeps_prefill_and_temperature(monkeypatch):
    client, fake = _client("claude-haiku-4-5-20251001", '"a": 1}', monkeypatch)
    assert client.complete_json("prompt", prefill="{") == {"a": 1}
    params = fake.calls[0]
    assert params["temperature"] == 0.3
    assert params["messages"][-1] == {"role": "assistant", "content": "{"}


def test_map_raises_when_every_call_fails(monkeypatch):
    client, _ = _client("claude-sonnet-5", "{}", monkeypatch)

    def boom(_):
        raise RuntimeError("401 invalid key")

    with pytest.raises(llm_mod.LlmError, match="all 3 calls failed"):
        client.map(boom, [1, 2, 3], desc="t")
    assert client.map(lambda x: 1 / x, [1, 0], desc="t") == [1.0, None]


def test_packed_masks_roundtrip_and_volatility_matches_stack():
    import numpy as np

    from bropilot.utils.imaging import PackedMasks, volatility_map

    rng = np.random.default_rng(0)
    raw = [rng.random((37, 53)) > 0.7 for _ in range(12)]
    packed = PackedMasks(raw)
    assert len(packed) == 12 and packed.nbytes < sum(m.nbytes for m in raw) / 7
    assert all(np.array_equal(a, b) for a, b in zip(raw, packed))

    stack = np.stack(raw).astype(np.int8)
    ref = np.abs(np.diff(stack, axis=0)).mean(axis=0).astype(np.float32)
    got = volatility_map(packed, smooth=1)
    assert np.allclose(got, ref, atol=1e-6)
