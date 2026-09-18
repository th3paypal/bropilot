from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from . import settings as user_settings
from .agents.qa import QaAgent
from .agents.quiz import QuizAgent
from .config import describe, load_config
from .jobs import STAGE_TITLES, delete_run, workdir_for
from .pipeline import STAGE_NAMES, build_pipeline, make_context
from .pricing import analog_prices, asr_prices, format_usd, run_cost
from .schema import Lecture, QuizItem
from .utils.common import format_duration, to_timecode

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVE_STATE = PROJECT_ROOT / ".serve.json"
SERVE_LOG = PROJECT_ROOT / ".serve.log"


def _port_busy(host: str, port: int) -> bool:
    target = "127.0.0.1" if host in {"0.0.0.0", "localhost"} else host
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((target, port)) == 0


def _wait_until_up(proc: "subprocess.Popen", host: str, port: int, seconds: float = 40) -> bool:
    with console.status("[cyan]поднимаю сервер…"):
        for _ in range(int(seconds * 4)):
            if proc.poll() is not None:
                return False
            if _port_busy(host, port):
                return True
            time.sleep(0.25)
    return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _stop_background() -> None:
    if not SERVE_STATE.exists():
        console.print("[yellow]Фоновый интерфейс не запускался[/] "
                      "(или его остановили не этой командой).")
        raise typer.Exit(1)
    state = json.loads(SERVE_STATE.read_text(encoding="utf-8"))
    pid = int(state["pid"])
    if not _alive(pid):
        SERVE_STATE.unlink(missing_ok=True)
        console.print(f"[dim]процесс {pid} уже не работает — состояние очищено[/]")
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(40):
        if not _alive(pid):
            break
        time.sleep(0.25)
    else:
        os.kill(pid, signal.SIGKILL)
    SERVE_STATE.unlink(missing_ok=True)
    console.print(f"[green]✓[/] интерфейс на {state.get('url', '')} остановлен")


HELP = """bropilot — видеолекция в конспект, вопросы и тест.

    bropilot serve                                   веб-интерфейс на localhost
    bropilot run <url|файл> [ключ=значение ...] [--until СТАДИЯ] [--force]
    bropilot ask  <url|файл> "вопрос"
    bropilot quiz <url|файл> -n 12 --answers
    bropilot inspect <url|файл> [--transcript]
    bropilot pdf <url|файл>                          пересобрать конспект в PDF
    bropilot library                                 что уже обработано
    bropilot remove <url|файл>                       удалить лекцию
    bropilot cost [<url|файл>]                       во что обошлась обработка
    bropilot config [--runs ПАПКА] [--profile cpu]

Переопределения конфига пишутся как обычные аргументы: `vision.tau_erase=0.02`.
"""

app = typer.Typer(
    name="bropilot",
    help=HELP,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

_state: dict[str, object] = {"config": "default", "log_level": None}


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False,
                              markup=False, log_time_format="%H:%M:%S")],
        force=True,
    )
    for noisy in ("httpx", "urllib3", "faster_whisper", "matplotlib", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _cfg(overrides: list[str] | None = None, *, force: bool = False):
    items = list(overrides or [])
    if force:
        items.append("force=true")
    for item in items:
        if "=" not in item:
            console.print(f"[red]Не похоже на переопределение конфига:[/] {item!r} "
                          "— ожидается вид [bold]ключ=значение[/].")
            raise typer.Exit(2)
    try:
        cfg = load_config(str(_state["config"]), items)
    except FileNotFoundError as exc:
        console.print(f"[red]Конфиг не найден:[/] {exc}")
        raise typer.Exit(2) from exc
    _setup_logging(str(_state["log_level"] or cfg.logging.level))
    return cfg


def _runs_root(cfg) -> Path:
    configured = Path(str(cfg.paths.workdir))
    if configured != Path("./runs"):
        return configured
    return user_settings.load().active_runs_root()


def _workdir(cfg, source: str) -> Path:
    return workdir_for(_runs_root(cfg), source)


def _load_doc(cfg, source: str) -> Lecture:
    path = _workdir(cfg, source) / "lecture.json"
    if not path.exists():
        console.print(Panel(
            "Для этого источника нет обработанной лекции.\n"
            f"Сначала выполни: [bold cyan]bropilot run {source!r}[/]",
            title="[yellow]нечего показывать", border_style="yellow"))
        raise typer.Exit(1)
    return Lecture.load(path)


@app.callback()
def _global_options(
    config: Annotated[str, typer.Option("--config", "-c", help="имя конфига в configs/")] = "default",
    log_level: Annotated[Optional[str], typer.Option("--log-level", help="DEBUG | INFO | WARNING")] = None,
) -> None:
    _state["config"] = config
    _state["log_level"] = log_level


@app.command(help="Прогнать пайплайн: видео → речь → доска → разделы → конспект.")
def run(
    source: Annotated[str, typer.Argument(help="ссылка на видео или путь к файлу")],
    overrides: Annotated[Optional[list[str]], typer.Argument(
        help="переопределения конфига, например vision.tau_erase=0.02")] = None,
    until: Annotated[Optional[str], typer.Option("--until", "-u",
        help=f"остановиться после стадии: {', '.join(STAGE_NAMES)}")] = None,
    skip: Annotated[Optional[list[str]], typer.Option("--skip", "-s",
        help="пропустить стадию (можно несколько раз)")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="игнорировать кеш стадий")] = False,
) -> None:
    for name in filter(None, [until, *(skip or [])]):
        if name not in STAGE_NAMES:
            console.print(f"[red]Неизвестная стадия[/] {name!r}. Есть: {', '.join(STAGE_NAMES)}")
            raise typer.Exit(2)

    cfg = _cfg(overrides, force=force)
    ctx = make_context(cfg, source)
    pipeline = build_pipeline(cfg, set(skip or []))
    stages = pipeline.stages
    if until:
        stages = stages[: [s.name for s in stages].index(until) + 1]

    console.print(Panel(
        f"[bold]{source}[/]\nпрофиль [cyan]{_state['config']}[/] · "
        f"стадий {len(stages)} · рабочая папка [dim]{ctx.workdir}[/]",
        title="[bold]bropilot", border_style="cyan"))

    started = time.perf_counter()
    timings: dict[str, float] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=24),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("подготовка", total=len(stages))
        for stage in stages:
            progress.update(task, description=STAGE_TITLES.get(stage.name, stage.name))
            stage_started = time.perf_counter()
            try:
                ctx = stage(ctx)
            except Exception as exc:  # noqa: BLE001
                progress.stop()
                console.print(Panel(str(exc), title=f"[red]стадия {stage.name} упала",
                                    border_style="red"))
                raise typer.Exit(1) from exc
            timings[stage.name] = time.perf_counter() - stage_started
            progress.advance(task)
    total = time.perf_counter() - started

    table = Table(box=None, pad_edge=False)
    table.add_column("стадия", style="cyan")
    table.add_column("время", justify="right", style="magenta")
    for name, seconds in timings.items():
        table.add_row(STAGE_TITLES.get(name, name), format_duration(seconds))
    table.add_row("[bold]всего", f"[bold]{format_duration(total)}")
    console.print(table)

    doc = ctx.doc
    if doc.duration:
        speed = doc.duration / total if total else 0
        console.print(f"[dim]запись {format_duration(doc.duration)} · "
                      f"обработка {'быстрее' if speed >= 1 else 'медленнее'} реального "
                      f"времени в {max(speed, 1 / speed if speed else 0):.1f} раза[/]")

    console.print(Rule(style="dim"))
    console.print(
        f"реплик [bold]{len(doc.utterances)}[/] · "
        f"состояний доски [bold]{len(doc.boards)}[/] · "
        f"разделов [bold]{len(doc.segments)}[/]")
    cost = run_cost(doc, ctx.workdir)
    if cost.calls:
        console.print(f"[dim]API: {cost.calls} вызовов · {cost.tokens:,} токенов · "
                      f"{format_usd(cost.cost_usd())}[/]".replace(",", " "))
    out = ctx.workdir / "output"
    for name in ("notes.pdf", "notes.tex"):
        if (out / name).exists():
            console.print(f"  [green]✓[/] {out / name}")
    console.print(f"  [green]✓[/] {ctx.workdir / 'lecture.json'}")
    if (doc.meta.get("compose") or {}).get("pdf_error"):
        console.print(f"[yellow]PDF не собран:[/] {doc.meta['compose']['pdf_error']}")


@app.command(help="Задать вопрос по лекции — ответ строится только по её материалу.")
def ask(
    source: Annotated[str, typer.Argument(help="ссылка или путь обработанной лекции")],
    question: Annotated[str, typer.Argument(help="вопрос по материалу лекции")],
    overrides: Annotated[Optional[list[str]], typer.Argument(help="переопределения конфига")] = None,
) -> None:
    cfg = _cfg(overrides)
    doc = _load_doc(cfg, source)
    with console.status("[cyan]ищу в лекции и формулирую ответ…"):
        answer = QaAgent(doc, cfg).ask(question)

    console.print(Panel(answer.text.strip(), title=f"[bold]{question}", border_style="cyan"))
    if answer.sources:
        console.print("[dim]источники в записи:[/] " + ", ".join(tc for tc, _ in answer.sources))
    if answer.unsupported:
        console.print("[yellow]![/] метки вне найденных фрагментов: " + ", ".join(answer.unsupported))


def _print_quiz(items: list[QuizItem], with_answers: bool) -> None:
    letters = "АБВГ"
    for i, item in enumerate(items, 1):
        lines = [f"[bold]{i}. {item.question}[/]", ""]
        for j, option in enumerate(item.options):
            right = with_answers and j == item.answer_index
            lines.append(f"  {'[green]' if right else ''}{letters[j]}) {option}"
                         f"{'[/]' if right else ''}")
        if with_answers and item.explanation:
            lines += ["", f"[dim]{item.explanation}[/]"]
        console.print(Panel("\n".join(lines), border_style="dim",
                            subtitle=f"[dim]{item.difficulty} · {to_timecode(item.ref_ts)}"))


@app.command(help="Составить тест с выбором ответа по материалу лекции.")
def quiz(
    source: Annotated[str, typer.Argument(help="ссылка или путь обработанной лекции")],
    overrides: Annotated[Optional[list[str]], typer.Argument(help="переопределения конфига")] = None,
    questions: Annotated[int, typer.Option("--questions", "-n", min=1, max=50)] = 10,
    answers: Annotated[bool, typer.Option("--answers/--no-answers", "-a",
        help="показать правильные ответы")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="вывести JSON")] = False,
) -> None:
    cfg = _cfg(overrides)
    doc = _load_doc(cfg, source)
    with console.status(f"[cyan]составляю {questions} вопросов…"):
        items = QuizAgent(doc, cfg).generate(questions)

    if as_json:
        console.print_json(json.dumps([i.model_dump() for i in items], ensure_ascii=False))
        return
    if not items:
        console.print("[yellow]Модель не вернула корректных вопросов — попробуй ещё раз.")
        raise typer.Exit(1)
    _print_quiz(items, answers)


@app.command(help="Показать, что получилось из лекции: разделы, доска, время стадий.")
def inspect(
    source: Annotated[str, typer.Argument(help="ссылка или путь обработанной лекции")],
    overrides: Annotated[Optional[list[str]], typer.Argument(help="переопределения конфига")] = None,
    transcript: Annotated[bool, typer.Option("--transcript", "-t",
        help="показать расшифровку речи с таймкодами")] = False,
) -> None:
    cfg = _cfg(overrides)
    doc = _load_doc(cfg, source)

    head = Table.grid(padding=(0, 2))
    head.add_row("длительность", f"[bold]{to_timecode(doc.duration)}")
    head.add_row("реплик речи", f"[bold]{len(doc.utterances)}")
    head.add_row("состояний доски",
                 f"[bold]{len(doc.boards)}[/] (распознано {sum(1 for b in doc.boards if b.latex)})")
    head.add_row("разделов", f"[bold]{len(doc.segments)}")
    console.print(Panel(head, title=f"[bold]{doc.title or 'лекция'}", border_style="cyan"))

    if doc.segments:
        table = Table(title="Разделы", box=None, title_justify="left", pad_edge=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("время", style="magenta")
        table.add_column("заголовок")
        for s in doc.segments:
            table.add_row(str(s.sid + 1), to_timecode(s.span.start), s.title or "[dim]без заголовка")
        console.print(table)

    cost = run_cost(doc, _workdir(cfg, source))
    if cost.stage_seconds:
        console.print("[dim]время стадий:[/] " + "  ".join(
            f"{STAGE_TITLES.get(k, k)} {format_duration(v)}" for k, v in cost.stage_seconds.items()))
        console.print(f"[dim]всего обработки:[/] {format_duration(cost.processing_seconds)}"
                      + (f" · [dim]расшифровка:[/] {format_duration(cost.transcription_seconds)}"
                         if cost.transcription_seconds else ""))

    if transcript:
        console.print(Rule("расшифровка", style="dim"))
        for u in doc.utterances:
            console.print(f"[magenta]{to_timecode(u.span.start)}[/]  {u.text}")


@app.command(help="Показать обработанные лекции и их размер на диске.")
def library(
    overrides: Annotated[Optional[list[str]], typer.Argument(help="переопределения конфига")] = None,
) -> None:
    cfg = _cfg(overrides)
    root = _runs_root(cfg)
    if not root.exists():
        console.print(f"[yellow]Папка с лекциями не создана:[/] {root}")
        raise typer.Exit(1)

    table = Table(box=None, pad_edge=False, title=f"Лекции в {root}", title_justify="left")
    table.add_column("лекция")
    table.add_column("длительность", justify="right", style="magenta")
    table.add_column("разделов", justify="right")
    table.add_column("на диске", justify="right", style="dim")
    table.add_column("папка", style="dim")

    total = 0
    for wd in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0,
                     reverse=True):
        snapshot = wd / "lecture.json"
        if not wd.is_dir() or not snapshot.exists():
            continue
        doc = Lecture.load(snapshot)
        size = sum(f.stat().st_size for f in wd.rglob("*") if f.is_file())
        total += size
        table.add_row(doc.title or Path(doc.source).name, to_timecode(doc.duration),
                      str(len(doc.segments)), _human_size(size), wd.name)
    console.print(table)
    console.print(f"[dim]всего на диске: {_human_size(total)}[/]")


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.0f} {unit}" if unit in ("Б", "КБ") else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ГБ"


@app.command(help="Удалить лекцию со всеми артефактами прогона.")
def remove(
    source: Annotated[str, typer.Argument(help="ссылка или путь удаляемой лекции")],
    overrides: Annotated[Optional[list[str]], typer.Argument(help="переопределения конфига")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="не спрашивать подтверждение")] = False,
) -> None:
    cfg = _cfg(overrides)
    wd = _workdir(cfg, source)
    if not wd.exists():
        console.print(f"[yellow]Такой лекции нет:[/] {wd}")
        raise typer.Exit(1)
    size = sum(f.stat().st_size for f in wd.rglob("*") if f.is_file())
    if not yes and not typer.confirm(f"Удалить {wd} ({_human_size(size)})?"):
        console.print("[dim]отменено")
        raise typer.Exit(1)
    delete_run(wd)
    console.print(f"[green]✓[/] удалено {wd} ({_human_size(size)} освобождено)")


@app.command(help="Во что обходится обработка лекции здесь и у аналогов.")
def cost(
    source: Annotated[Optional[str], typer.Argument(
        help="лекция; без аргумента — таблица тарифов для эталонной длительности")] = None,
    overrides: Annotated[Optional[list[str]], typer.Argument(help="переопределения конфига")] = None,
    minutes: Annotated[float, typer.Option("--minutes", "-m",
        help="длительность, на которую пересчитать тарифы")] = 80.0,
) -> None:
    cfg = _cfg(overrides)
    boards = 0
    if source:
        doc = _load_doc(cfg, source)
        rc = run_cost(doc, _workdir(cfg, source))
        boards = len(doc.boards)
        minutes = doc.duration / 60 or minutes

        own = Table(box=None, pad_edge=False, title="Расход этой лекции", title_justify="left")
        own.add_column("стадия", style="cyan")
        own.add_column("вызовов", justify="right")
        own.add_column("вход", justify="right")
        own.add_column("выход", justify="right")
        own.add_column("стоимость", justify="right", style="magenta")
        for row in rc.rows:
            own.add_row(STAGE_TITLES.get(row.stage, row.stage), str(row.calls),
                        f"{row.input_tokens:,}".replace(",", " "),
                        f"{row.output_tokens:,}".replace(",", " "),
                        format_usd(row.cost_usd()))
        own.add_row("[bold]всего", f"[bold]{rc.calls}",
                    f"[bold]{rc.input_tokens:,}".replace(",", " "),
                    f"[bold]{rc.output_tokens:,}".replace(",", " "),
                    f"[bold]{format_usd(rc.cost_usd())}")
        console.print(own)
        console.print(f"[dim]запись {format_duration(rc.duration)} · "
                      f"обработка {format_duration(rc.processing_seconds)} · "
                      f"расшифровка {format_duration(rc.transcription_seconds)}[/]")
        if rc.has_assumed_models:
            console.print(f"[yellow]![/] в записи прогона нет модели — цена "
                          f"посчитана по [bold]{', '.join(rc.models)}[/]")

    table = Table(box=None, pad_edge=False,
                  title=f"Лекция {minutes:.0f} мин у аналогов", title_justify="left")
    table.add_column("сервис")
    table.add_column("за лекцию", justify="right", style="magenta")
    table.add_column("как посчитано", style="dim")
    for item in analog_prices(minutes, boards=boards):
        table.add_row(item.title, format_usd(item.usd_per_lecture), item.basis)
    console.print(table)

    asr = Table(box=None, pad_edge=False, title="Только распознавание речи", title_justify="left")
    asr.add_column("сервис")
    asr.add_column("за час", justify="right")
    asr.add_column("за лекцию", justify="right", style="magenta")
    for row in asr_prices(minutes):
        asr.add_row(row["title"], format_usd(row["usd_per_hour"]), format_usd(row["usd_per_lecture"]))
    console.print(asr)
    console.print("[dim]Прайсы и ссылки на источники — configs/pricing.yaml;"
                  " полные таблицы — python scripts/cost_report.py[/]")


@app.command(help="Пересобрать конспект в PDF из готовых разделов (без обращения к модели).")
def pdf(
    source: Annotated[str, typer.Argument(help="ссылка или путь обработанной лекции")],
    overrides: Annotated[Optional[list[str]], typer.Argument(help="переопределения конфига")] = None,
    out: Annotated[Optional[Path], typer.Option("--out", "-o",
        help="куда положить готовый файл")] = None,
) -> None:
    from .exporting import available_engine, compile_pdf
    from .stages.compose import assemble_tex

    cfg = _cfg(overrides)
    doc = _load_doc(cfg, source)
    if not any(s.body for s in doc.segments):
        console.print("[yellow]Разделы ещё не написаны — сначала полный прогон.")
        raise typer.Exit(1)
    if available_engine() is None:
        console.print(Panel(
            "Не найден движок LaTeX. Поставь один из них:\n"
            "  [bold]brew install tectonic[/]            (macOS)\n"
            "  [bold]winget install TectonicProject.Tectonic[/]  (Windows)\n"
            "  [bold]apt install texlive-xetex latexmk[/]  (Linux)",
            title="[yellow]нечем собирать PDF", border_style="yellow"))
        raise typer.Exit(1)

    wd = _workdir(cfg, source)
    with console.status("[cyan]собираю PDF…"):
        tex = assemble_tex(doc, wd / "output")
        result = compile_pdf(tex)
    if not result.ok:
        console.print(Panel(result.log_tail or result.error, title="[red]сборка не удалась",
                            border_style="red"))
        raise typer.Exit(1)

    target = result.pdf
    if out is not None:
        target = Path(out).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.pdf.read_bytes())
    console.print(f"[green]✓[/] {target}  [dim]({result.engine})[/]")


@app.command('config', help="Показать или изменить настройки: папки хранения, профиль, модель.")
def show_config(
    overrides: Annotated[Optional[list[str]], typer.Argument(help="переопределения конфига")] = None,
    runs: Annotated[Optional[Path], typer.Option("--runs",
        help="сменить папку, где хранятся лекции")] = None,
    export: Annotated[Optional[Path], typer.Option("--export",
        help="сменить папку по умолчанию для сохранения конспектов")] = None,
    full: Annotated[bool, typer.Option("--full", help="показать конфиг пайплайна целиком")] = False,
) -> None:
    saved = user_settings.load()
    changed = False
    for value, setter, label in ((runs, saved.set_runs_root, "лекции"),
                                 (export, lambda p: setattr(saved, "export_dir", str(p)), "экспорт")):
        if value is None:
            continue
        problem = user_settings.check_writable(value)
        if problem:
            console.print(f"[red]Папка не подходит ({label}):[/] {problem}")
            raise typer.Exit(2)
        setter(Path(value).expanduser().resolve())
        changed = True
    if changed:
        user_settings.save(saved)

    table = Table.grid(padding=(0, 2))
    table.add_row("лекции", f"[bold]{saved.active_runs_root()}")
    table.add_row("экспорт по умолчанию", f"[bold]{saved.export_target()}")
    table.add_row("профиль распознавания", f"[bold]{saved.asr_profile}")
    table.add_row("модель", f"[bold]{saved.model}")
    table.add_row("настройки", f"[dim]{user_settings.SETTINGS_FILE}")
    console.print(Panel(table, title="[bold]bropilot · настройки", border_style="cyan"))
    if changed:
        console.print("[green]✓[/] сохранено")

    if full:
        console.print(Syntax(describe(_cfg(overrides)), "yaml", theme="ansi_dark",
                             background_color="default"))


@app.command(help="Поднять веб-интерфейс на localhost и открыть его в браузере.")
def serve(
    port: Annotated[int, typer.Option("--port", "-p", min=1, max=65535,
        help="порт localhost")] = 8501,
    host: Annotated[str, typer.Option("--host", help="адрес прослушивания")] = "localhost",
    runs: Annotated[Optional[Path], typer.Option("--runs", help="папка с прогонами")] = None,
    browser: Annotated[bool, typer.Option("--browser/--no-browser",
        help="открывать браузер автоматически")] = True,
    background: Annotated[bool, typer.Option("--background", "-b",
        help="запустить фоном и сразу вернуть приглашение терминала")] = False,
    stop: Annotated[bool, typer.Option("--stop", help="остановить фоновый интерфейс")] = False,
) -> None:
    if stop:
        _stop_background()
        return
    page = PROJECT_ROOT / "app.py"
    if not page.exists():
        console.print(f"[red]Не найден веб-интерфейс:[/] {page}")
        raise typer.Exit(1)
    if _port_busy(host, port):
        console.print(Panel(
            f"Порт [bold]{port}[/] уже занят — возможно, интерфейс уже запущен: "
            f"http://{host}:{port}\nИначе возьми другой порт: "
            f"[bold cyan]bropilot serve --port {port + 1}[/]",
            title="[yellow]порт занят", border_style="yellow"))
        raise typer.Exit(1)

    env = os.environ.copy()
    root = Path(runs).expanduser().resolve() if runs else user_settings.load().active_runs_root()
    env["BROPILOT_RUNS"] = str(root)
    env.setdefault("PYTHONUTF8", "1")

    url = f"http://{host}:{port}"
    console.print(Panel(
        f"[bold cyan]{url}[/]\n"
        f"[dim]лекции: {root}\n"
        + ("остановить: bropilot serve --stop" if background else "Ctrl+C — остановить")
        + "[/]",
        title="[bold]bropilot · веб-интерфейс", border_style="cyan"))

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(page),
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    extra: dict = {}
    log_handle = None
    if background:
        log_handle = open(SERVE_LOG, "w", encoding="utf-8")  # noqa: SIM115
        extra = {"stdout": log_handle, "stderr": subprocess.STDOUT, "start_new_session": True}
    try:
        proc = subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT), **extra)
    except FileNotFoundError as exc:
        console.print("[red]Streamlit не установлен:[/] pip install streamlit")
        raise typer.Exit(1) from exc
    finally:
        if log_handle is not None:
            log_handle.close()

    ready = _wait_until_up(proc, host, port)
    if ready and browser:
        typer.launch(url)
        console.print(f"[green]✓[/] открыл {url} в браузере")
    elif not ready:
        console.print(f"[yellow]сервер не ответил за 40 с — смотри {SERVE_LOG}[/]")

    if background:
        SERVE_STATE.write_text(json.dumps({"pid": proc.pid, "url": url, "port": port}),
                               encoding="utf-8")
        console.print(f"[dim]фоновый процесс {proc.pid} · журнал {SERVE_LOG} · "
                      "остановить: [/][bold cyan]bropilot serve --stop[/]")
        return

    code = 0
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        console.print("\n[dim]интерфейс остановлен")
        code = 0
    if code:
        raise typer.Exit(code)


@app.command(help="Список стадий пайплайна.")
def stages() -> None:
    table = Table(box=None, pad_edge=False)
    table.add_column("стадия", style="cyan")
    table.add_column("что делает")
    for name in STAGE_NAMES:
        table.add_row(name, STAGE_TITLES.get(name, ""))
    console.print(table)


def entrypoint() -> None:  # pragma: no cover
    app()


def main(argv: list[str] | None = None) -> int:
    try:
        rv = app(args=list(argv) if argv is not None else None, standalone_mode=False)
        return rv if isinstance(rv, int) else 0
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "exit_code", None)
        if code is None:
            raise
        show = getattr(exc, "show", None)
        if callable(show):
            show()
        return int(code)


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
