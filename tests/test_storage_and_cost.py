from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bropilot import exporting, jobs, pricing
from bropilot import settings as user_settings
from bropilot.config import load_config
from bropilot.schema import BoardState, Lecture, Segment, Span, Utterance
from bropilot.stages.base import Context
from bropilot.utils.common import resolve_artifact


@pytest.fixture
def home(tmp_path, monkeypatch):
    target = tmp_path / "home"
    monkeypatch.setattr(user_settings, "HOME", target)
    monkeypatch.setattr(user_settings, "SETTINGS_FILE", target / "settings.json")
    monkeypatch.delenv("BROPILOT_RUNS", raising=False)
    return target


def test_settings_round_trip(home, tmp_path):
    saved = user_settings.load()
    assert saved.active_runs_root() == user_settings.DEFAULT_RUNS

    saved.set_runs_root(tmp_path / "lectures")
    saved.export_dir = str(tmp_path / "out")
    user_settings.save(saved)

    again = user_settings.load()
    assert again.active_runs_root() == (tmp_path / "lectures").resolve()
    assert again.export_target() == tmp_path / "out"


def test_previous_root_stays_in_the_library(home, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()

    saved = user_settings.load()
    saved.set_runs_root(first)
    saved.set_runs_root(second)

    roots = saved.library_roots()
    assert roots[0] == second.resolve()
    assert first.resolve() in roots


def test_env_overrides_saved_root(home, tmp_path, monkeypatch):
    saved = user_settings.load()
    saved.set_runs_root(tmp_path / "saved")
    monkeypatch.setenv("BROPILOT_RUNS", str(tmp_path / "forced"))
    assert saved.active_runs_root() == (tmp_path / "forced").resolve()


def test_check_writable_reports_the_problem(tmp_path):
    assert user_settings.check_writable(tmp_path / "new" / "deep") is None
    assert user_settings.check_writable("") is not None
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert user_settings.check_writable(blocker / "sub") is not None


def test_delete_run_removes_everything(tmp_path):
    wd = tmp_path / "abcdef"
    (wd / "frames").mkdir(parents=True)
    (wd / "frames" / "board_000.png").write_bytes(b"x")
    (wd / "lecture.json").write_text("{}", encoding="utf-8")

    jobs.delete_run(wd)
    assert not wd.exists()


def test_delete_run_is_quiet_about_a_missing_folder(tmp_path):
    jobs.delete_run(tmp_path / "nothing-here")


def _finished_job(wd: Path, *, started: float, finished: float) -> None:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / jobs.JOB_FILE).write_text(
        json.dumps({"pid": 1, "source": "x", "started": started, "until": None}),
        encoding="utf-8")
    (wd / jobs.RESULT_FILE).write_text(
        json.dumps({"returncode": 0, "finished": finished}), encoding="utf-8")
    (wd / jobs.LOG_FILE).write_text(
        "10:00:00 | INFO | [ingest] running (a)\n"
        "10:00:10 | INFO | [ingest] done in 9.7s\n"
        "10:00:10 | INFO | [asr] running (b)\n"
        "10:02:00 | INFO | [asr] done in 110.2s\n"
        "10:02:00 | INFO | [boards] running (c)\n",
        encoding="utf-8")


def test_status_reports_how_long_the_run_took(tmp_path):
    wd = tmp_path / "run"
    _finished_job(wd, started=1000.0, finished=1225.5)

    info = jobs.status(wd)
    assert info["elapsed"] == pytest.approx(225.5)
    assert info["stage_seconds"] == {"ingest": 9.7, "asr": 110.2}
    assert info["stages"]["asr"] == "done"
    assert info["current"] == "boards"


def test_status_counts_a_running_job_from_its_start(tmp_path, monkeypatch):
    wd = tmp_path / "run"
    wd.mkdir()
    (wd / jobs.JOB_FILE).write_text(
        json.dumps({"pid": os.getpid(), "source": "x", "started": 1000.0}),
        encoding="utf-8")
    monkeypatch.setattr(jobs, "is_running", lambda _: True)
    monkeypatch.setattr(jobs.time, "time", lambda: 1042.0)

    info = jobs.status(wd)
    assert info["running"] and info["elapsed"] == pytest.approx(42.0)


def test_artifact_is_found_after_the_run_folder_moved(tmp_path):
    wd = tmp_path / "новый-хеш"
    (wd / "frames").mkdir(parents=True)
    frame = wd / "frames" / "board_001_000269.png"
    frame.write_bytes(b"png")

    stale = "runs/старый-хеш/frames/board_001_000269.png"
    assert resolve_artifact(stale, wd, "frames") == frame
    assert resolve_artifact(str(frame), wd, "frames") == frame
    assert resolve_artifact("runs/x/frames/нет.png", wd, "frames") is None
    assert resolve_artifact(None, wd, "frames") is None


def test_board_reading_names_the_cause_instead_of_burning_api_calls(tmp_path, monkeypatch):
    from bropilot.stages import board_reading as stage_mod

    doc = _doc()
    doc.boards = [BoardState(bid=0, span=Span(start=0, end=10), key_ts=5.0,
                             frame_path="runs/пропала/frames/board_000.png")]

    def explode(*_args, **_kwargs):
        raise AssertionError("к модели обращаться не должны")

    monkeypatch.setattr(stage_mod, "LlmClient", explode)
    cfg = load_config("cpu", [f"paths.workdir={tmp_path.as_posix()}"])
    ctx = Context(cfg=cfg, workdir=tmp_path, doc=doc)

    with pytest.raises(RuntimeError) as err:
        stage_mod.BoardReadingStage(cfg).apply(ctx)
    assert "кадр" in str(err.value) and "--force" in str(err.value)


def test_board_reading_repairs_the_path_it_had_to_search_for(tmp_path, monkeypatch):
    from bropilot.stages import board_reading as stage_mod

    (tmp_path / "frames").mkdir()
    frame = tmp_path / "frames" / "board_000.png"
    frame.write_bytes(b"png")

    doc = _doc()
    doc.boards = [BoardState(bid=0, span=Span(start=0, end=10), key_ts=5.0,
                             frame_path="runs/старая-папка/frames/board_000.png")]

    class _Client:
        usage = None

        def __init__(self, **_kw):
            pass

        def map(self, fn, items, desc=""):
            return [{"latex": "$x$", "caption": "", "kind": "handwriting",
                     "confidence": 0.9} for _ in items]

        def report(self):
            return {"model": "fake", "calls": 1, "input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(stage_mod, "LlmClient", _Client)
    cfg = load_config("cpu", [f"paths.workdir={tmp_path.as_posix()}"])
    result = stage_mod.BoardReadingStage(cfg).apply(
        Context(cfg=cfg, workdir=tmp_path, doc=doc))

    assert result.boards[0].frame_path == str(frame)
    assert result.meta["board_reading"]["frames_missing"] == 0


def test_workdir_is_absolute_so_recorded_paths_survive_a_chdir(tmp_path, monkeypatch):
    from bropilot.pipeline import make_context

    monkeypatch.chdir(tmp_path)
    cfg = load_config("cpu", ["paths.workdir=./runs"])
    ctx = make_context(cfg, "lecture.mp4")
    assert ctx.workdir.is_absolute()
    assert ctx.subdir("frames").is_absolute()


def _doc() -> Lecture:
    return Lecture(
        source="https://www.youtube.com/watch?v=abcdefghijk",
        title="Лекция 4: спуск",
        duration=600.0,
        utterances=[Utterance(uid=0, span=Span(start=0, end=5), text="привет")],
        segments=[Segment(sid=0, span=Span(start=0, end=600), title="Спуск",
                          utterance_ids=[0],
                          body=r"\section{Спуск}\ts{01:00} Текст.")],
    )


def test_markdown_export_links_timecodes_into_the_recording(tmp_path):
    payload = exporting.artifact_bytes(_doc(), tmp_path, "md").decode("utf-8")
    assert "https://youtu.be/abcdefghijk?t=60s" in payload
    assert "# Лекция 4: спуск" in payload


def test_save_to_writes_chosen_formats_and_reports_the_rest(tmp_path):
    wd = tmp_path / "run"
    wd.mkdir()
    dest = tmp_path / "куда угодно" / "конспекты"

    written, problems = exporting.save_to(_doc(), wd, dest, ["md", "json", "pdf"])
    names = {p.name for p in written}
    assert names == {"Лекция 4 спуск.md", "Лекция 4 спуск.json"}
    assert all(p.exists() for p in written)
    assert any("PDF" in text for text in problems)


def test_safe_filename_strips_path_separators():
    assert "/" not in exporting.safe_filename("ML/1: спуск")
    assert exporting.safe_filename("") == "конспект"
    assert exporting.safe_filename("  ..  ") == "конспект"


def test_compile_pdf_says_what_is_missing(tmp_path):
    result = exporting.compile_pdf(tmp_path / "нет-такого.tex")
    assert not result.ok and "нет файла" in result.error


def _priced_doc() -> Lecture:
    doc = _doc()
    doc.meta = {
        "timings": {"ingest": 10.0, "asr": 100.0, "boards": 20.0, "compose": 30.0},
        "board_reading": {"model": "claude-sonnet-5", "calls": 4,
                          "input_tokens": 100_000, "output_tokens": 10_000},
        "compose": {"model": "claude-sonnet-5", "calls": 6,
                    "input_tokens": 200_000, "output_tokens": 40_000},
    }
    return doc


def test_cost_matches_the_price_list():
    doc = _priced_doc()
    cost = pricing.run_cost(doc, workdir=None)
    rate_in, rate_out = pricing.model_rate("claude-sonnet-5")

    expected = (300_000 * rate_in + 50_000 * rate_out) / 1e6
    assert cost.cost_usd() == pytest.approx(expected)
    assert cost.calls == 10
    assert cost.input_tokens == 300_000


def test_timings_split_transcription_from_the_rest():
    cost = pricing.run_cost(_priced_doc(), workdir=None)
    assert cost.transcription_seconds == 110.0
    assert cost.processing_seconds == 160.0
    assert cost.realtime_factor == pytest.approx(600.0 / 160.0)


def test_scaling_to_another_length_is_linear():
    cost = pricing.run_cost(_priced_doc(), workdir=None)
    scaled = cost.scaled_to(80)
    assert scaled["cost_usd"] == pytest.approx(cost.cost_usd() * 8)
    assert scaled["calls"] == pytest.approx(80)


def test_unknown_model_costs_nothing_instead_of_guessing():
    doc = _priced_doc()
    doc.meta["compose"]["model"] = "claude-из-будущего"
    cost = pricing.run_cost(doc, workdir=None)
    compose_row = next(r for r in cost.rows if r.stage == "compose")
    assert compose_row.cost_usd() == 0.0


def test_missing_model_falls_back_and_says_so():
    doc = _priced_doc()
    del doc.meta["compose"]["model"]
    cost = pricing.run_cost(doc, workdir=None, default_model="claude-opus-5")
    assert cost.has_assumed_models
    assert "claude-opus-5" in cost.models


def test_agent_usage_is_added_from_the_ledger(tmp_path):
    pricing.record_usage(tmp_path, "qa", {"model": "claude-sonnet-5", "calls": 1,
                                          "input_tokens": 1000, "output_tokens": 200})
    pricing.record_usage(tmp_path, "quiz", {"model": "claude-sonnet-5", "calls": 2,
                                            "input_tokens": 2000, "output_tokens": 500})
    cost = pricing.run_cost(_priced_doc(), tmp_path)
    assert cost.calls == 13
    assert {r.stage for r in cost.rows} >= {"qa", "quiz"}


def test_ledger_survives_a_broken_line(tmp_path):
    (tmp_path / pricing.USAGE_LEDGER).write_text(
        '{"kind":"qa","calls":1,"input_tokens":10,"output_tokens":2}\nне json\n',
        encoding="utf-8")
    rows = pricing.read_ledger(tmp_path)
    assert len(rows) == 1


def test_subscription_prices_are_divided_by_what_they_cover():
    prices = {a.key: a for a in pricing.analog_prices(60)}
    otter = prices["otter_pro"]
    assert otter.usd_per_lecture == pytest.approx(16.99 / 20)
    assert prices["lecture2notes"].usd_per_lecture == 0.0
    assert prices["bropilot_sonnet"].usd_per_lecture is None


def test_every_price_names_its_source():
    prices = pricing.load_pricing()
    for entry in prices.analogs:
        if entry.get("model") == "pay-per-use":
            continue
        assert entry.get("source"), f"{entry.key}: нет ссылки на источник"
        assert entry.get("checked"), f"{entry.key}: нет даты проверки"
    for key, entry in prices.asr.items():
        assert entry.get("source") and entry.get("checked"), key
