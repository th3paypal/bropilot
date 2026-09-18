from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bropilot import exporting, jobs, pricing  # noqa: E402
from bropilot import settings as user_settings  # noqa: E402
from bropilot.config import load_config  # noqa: E402
from bropilot.schema import Lecture  # noqa: E402
from bropilot.utils.common import (  # noqa: E402
    format_duration,
    from_timecode,
    resolve_artifact,
    to_timecode,
)
from bropilot.utils.latex_md import latex_to_markdown, timecodes  # noqa: E402

UPLOADS = ROOT / "uploads"
VIDEO_TYPES = ["mp4", "mkv", "webm", "mov", "avi", "m4v"]
NEW = "__new__"
CITATION = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")

TABS = ["Конспект", "Вопросы", "Тест", "Доска", "Речь", "Стоимость"]

st.set_page_config(page_title="bropilot — конспекты видеолекций",
                   page_icon=":material/menu_book:", layout="wide")

st.markdown(
    """<style>
    .block-container {padding-top: 2.2rem; max-width: 1500px;}
    a.bp-tc {
        font-variant-numeric: tabular-nums; font-size: 0.8em; font-weight: 500;
        padding: 0.05em 0.4em; border-radius: 0.4em; white-space: nowrap;
        text-decoration: none !important;
        color: var(--primary-color, #1F5FA8) !important;
        background: color-mix(in srgb, var(--primary-color, #1F5FA8) 10%, transparent);
    }
    a.bp-tc:hover {background: color-mix(in srgb, var(--primary-color, #1F5FA8) 22%, transparent);}
    .bp-brand {font-size: 1.45rem; font-weight: 650; letter-spacing: -0.02em; margin-bottom: 0;}
    .bp-sub {font-size: 0.82rem; opacity: 0.6; margin-top: -0.2rem;}
    .bp-sec h4 {margin-top: 0; font-size: 1.02rem;}
    .bp-cap {font-size: 0.82rem; opacity: 0.62; margin: 0.3rem 0 0.9rem 0;}
    .st-key-bp-seek {display: none;}
    </style>""",
    unsafe_allow_html=True,
)

with st.container(key="bp-seek"):
    st.iframe(
        """
        <script>
    (function () {
      const doc = window.parent.document;
      if (doc.__bropilotSeek) { return; }
      doc.__bropilotSeek = true;

      function timecode(total) {
        total = Math.max(0, Math.round(total));
        const h = Math.floor(total / 3600);
        const mm = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
        const ss = String(total % 60).padStart(2, "0");
        return h > 0 ? h + ":" + mm + ":" + ss : mm + ":" + ss;
      }

      doc.addEventListener("click", function (event) {
        const target = event.target;
        const link = target && target.closest ? target.closest("a.bp-tc") : null;
        if (!link) { return; }
        const seconds = parseFloat(link.dataset.t);
        const video = doc.querySelector("video");
        if (!video || !isFinite(seconds)) { return; }

        event.preventDefault();
        event.stopPropagation();
        video.currentTime = seconds;
        doc.querySelectorAll(".bp-pos").forEach(function (el) {
          el.textContent = timecode(seconds);
        });
        try {
          window.parent.history.replaceState({}, "", link.getAttribute("href"));
        } catch (err) {}
      }, true);
    })();
        </script>
        """,
        height=1,
    )


def settings() -> user_settings.Settings:
    if "settings" not in st.session_state:
        st.session_state["settings"] = user_settings.load()
    return st.session_state["settings"]


def save_settings() -> None:
    user_settings.save(settings())


def position() -> int:
    raw = st.query_params.get("t")
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return 0


def remember_link_state(**values: str) -> None:
    st.session_state["link_state"] = {**st.session_state.get("link_state", {}), **values}


def _link_query(seconds: str) -> str:
    params = {k: v for k, v in st.session_state.get("link_state", {}).items() if v}
    query = urlencode({**params, "t": seconds}, safe="{}")
    return query.replace("&", "&amp;")


def tc_link(seconds: float, label: str | None = None) -> str:
    seconds = max(0, int(seconds))
    text = label or to_timecode(seconds)
    return (f'<a class="bp-tc" data-t="{seconds}" href="?{_link_query(str(seconds))}" '
            f'target="_self">{text}</a>')


def timecode_link_format() -> str:
    return (f'<a class="bp-tc" data-t="{{sec}}" href="?{_link_query("{sec}")}" '
            f'target="_self">{{tc}}</a>')


def seek_row(marks: list[tuple[str, float]], limit: int = 12) -> None:
    if not marks:
        return
    links = " ".join(tc_link(sec, tc) for tc, sec in marks[:limit])
    st.markdown(links, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> Lecture:
    return Lecture.load(path)


def load_lecture(wd: Path) -> Lecture | None:
    f = wd / "lecture.json"
    if not f.exists():
        return None
    try:
        return _load(str(f), f.stat().st_mtime)
    except Exception:  # noqa: BLE001
        return None


def list_runs() -> list[dict]:
    items: list[dict] = []
    for root in settings().library_roots():
        if not root.exists():
            continue
        for wd in root.iterdir():
            if not wd.is_dir():
                continue
            doc = load_lecture(wd)
            job = wd / jobs.JOB_FILE
            if doc is None and not job.exists():
                continue
            if doc is not None:
                title = doc.title or Path(doc.source).stem
                ready = any(s.body for s in doc.segments)
            else:
                title = json.loads(job.read_text(encoding="utf-8")).get("source", wd.name)
                ready = False
            items.append({
                "wd": wd, "id": wd.name, "root": root, "title": title,
                "ready": ready, "mtime": wd.stat().st_mtime,
                "running": jobs.is_running(wd),
            })
    items.sort(key=lambda it: it["mtime"], reverse=True)
    return items


def find_run(runs: list[dict], run_id: str | None) -> dict | None:
    return next((r for r in runs if r["id"] == run_id), None)


def get_cfg(workdir: Path | None = None):
    saved = settings()
    cfg_name, overrides = user_settings.ASR_PROFILES[saved.asr_profile]
    root = workdir.parent if workdir else saved.active_runs_root()
    return load_config(cfg_name, [*overrides, f"paths.workdir={root.as_posix()}",
                                  f"llm.text_model={saved.model}",
                                  f"llm.vision_model={saved.model}"])


def api_ready() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def show_error(exc: Exception) -> None:
    msg = str(exc)
    if "ANTHROPIC_API_KEY" in msg:
        st.error("Не задан ключ API — введите его слева в боковой панели.")
    elif "401" in msg or "authentication" in msg.lower():
        st.error("Ключ API отклонён (401). Проверьте ключ.")
    else:
        st.error(f"Ошибка: {msg}")


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _side_file(wd: Path, name: str) -> Path:
    return wd / name


def load_side(wd: Path, name: str, default):
    path = _side_file(wd, name)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def save_side(wd: Path, name: str, value) -> None:
    try:
        _side_file(wd, name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def sidebar() -> None:
    saved = settings()
    with st.sidebar:
        st.markdown('<p class="bp-brand">bropilot</p>'
                    '<p class="bp-sub">видеолекция → конспект, вопросы, тест</p>',
                    unsafe_allow_html=True)
        st.divider()

        key = st.text_input("Ключ Anthropic API", value=os.environ.get("ANTHROPIC_API_KEY", ""),
                            type="password",
                            help="Нужен для чтения доски, конспекта, вопросов и теста.")
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key.strip()

        profile = st.selectbox("Распознавание речи", list(user_settings.ASR_PROFILES),
                               index=list(user_settings.ASR_PROFILES).index(saved.asr_profile)
                               if saved.asr_profile in user_settings.ASR_PROFILES else 0)
        model = st.selectbox("Модель Claude", user_settings.MODELS,
                             index=user_settings.MODELS.index(saved.model)
                             if saved.model in user_settings.MODELS else 0)
        if (profile, model) != (saved.asr_profile, saved.model):
            saved.asr_profile, saved.model = profile, model
            save_settings()

        with st.expander("Хранилище", icon=":material/folder:"):
            storage_settings(saved)

        st.caption(f"Лекции: `{saved.active_runs_root()}`")


def storage_settings(saved: user_settings.Settings) -> None:
    pinned = os.environ.get("BROPILOT_RUNS") and not saved.runs_root
    runs = st.text_input("Папка для лекций", value=str(saved.active_runs_root()),
                         disabled=bool(pinned),
                         help="Сюда попадают новые лекции: видео, кадры, конспект. "
                              "Место понадобится — час записи занимает около гигабайта.")
    if pinned:
        st.caption("Задана при запуске: `bropilot serve --runs …`")
    elif runs.strip() and Path(runs).expanduser().resolve() != saved.active_runs_root():
        problem = user_settings.check_writable(runs)
        if problem:
            st.error(f"Не подходит: {problem}")
        elif st.button("Перенести хранилище сюда", type="primary",
                       icon=":material/drive_file_move:", width="stretch"):
            saved.set_runs_root(runs)
            save_settings()
            st.success("Новые лекции пойдут в эту папку. Уже обработанные останутся "
                       "на месте и по-прежнему видны в списке.")
            st.rerun()

    export = st.text_input("Папка для конспектов по умолчанию", value=str(saved.export_target()),
                           help="Подставляется на вкладке «Конспект» в поле сохранения.")
    if export.strip() and Path(export).expanduser() != saved.export_target():
        problem = user_settings.check_writable(export)
        if problem:
            st.error(f"Не подходит: {problem}")
        elif st.button("Запомнить", icon=":material/check:", width="stretch"):
            saved.export_dir = str(Path(export).expanduser())
            save_settings()
            st.rerun()

    others = [r for r in saved.library_roots() if r != saved.active_runs_root()]
    if others:
        st.caption("Лекции также читаются из: " + ", ".join(f"`{p}`" for p in others))


def page_new() -> None:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Новая лекция")
        kind = st.segmented_control(
            "Откуда взять запись",
            ["Загрузить файл", "Ссылка на видео", "Файл на этом компьютере"],
            default="Загрузить файл", key="src_kind", label_visibility="collapsed",
            width="stretch",
        ) or "Загрузить файл"

        source = None
        if kind == "Ссылка на видео":
            raw = st.text_input("Ссылка", placeholder="https://www.youtube.com/watch?v=…",
                                label_visibility="collapsed")
            source = raw.strip() or None
            st.caption("YouTube и другие сайты, которые поддерживает yt-dlp.")
        elif kind == "Файл на этом компьютере":
            raw = st.text_input("Полный путь к видеофайлу", placeholder="/Users/…/lecture04.mp4",
                                label_visibility="collapsed")
            if raw.strip():
                p = Path(raw.strip().strip('"')).expanduser()
                if p.exists():
                    source = str(p.resolve())
                else:
                    st.warning("Файл не найден.")
            st.caption("Файл не копируется — обрабатывается там, где лежит.")
        else:
            up = st.file_uploader("Видео лекции", type=VIDEO_TYPES,
                                  label_visibility="collapsed")
            if up is not None:
                UPLOADS.mkdir(exist_ok=True)
                target = UPLOADS / up.name
                if not target.exists() or target.stat().st_size != up.size:
                    with open(target, "wb") as fh:
                        fh.write(up.getbuffer())
                source = str(target.resolve())
            st.caption(f"Копия сохраняется в `{UPLOADS.name}/`. До 4 ГБ.")

        only_local = st.toggle(
            "Только локальная часть: речь и кадры доски, без обращения к API",
            value=not api_ready(),
            help="Полезно, чтобы заранее проверить распознавание и не тратить API.")

        with st.expander("Параметры обработки", icon=":material/tune:"):
            force = st.checkbox("Пересчитать всё заново, игнорируя кеш")
            conc = st.slider("Параллельных запросов к API", 1, 12, 4)
            tau = st.number_input("Порог стирания доски (vision.tau_erase)", value=0.010,
                                  step=0.002, format="%.3f")

        if not only_local and not api_ready():
            st.info("Для полного конспекта нужен ключ API — введите его слева "
                    "или включите «Только локальная часть».")

        disabled = source is None or (not only_local and not api_ready())
        if st.button("Обработать лекцию", type="primary", icon=":material/play_arrow:",
                     disabled=disabled, width="stretch"):
            start_processing(source, only_local=only_local, force=force, conc=conc, tau=tau)

    with right:
        st.markdown("#### Что будет дальше")
        steps = [
            (":material/download:", "Загрузка", "видео и звук 16 кГц"),
            (":material/graphic_eq:", "Речь", "faster-whisper, локально"),
            (":material/gesture:", "Доска", "моменты, когда запись закончена"),
            (":material/function:", "Формулы", "кадр → LaTeX, с опорой на речь"),
            (":material/segment:", "Разделы", "по смене доски и связности речи"),
            (":material/description:", "Конспект", "разделы, LaTeX, PDF"),
        ]
        for icon, title, note in steps:
            with st.container(border=True):
                st.markdown(f"{icon} **{title}** — {note}")
        st.caption("Каждая стадия кешируется: изменили порог зрения — речь заново "
                   "не распознаётся. Обработку можно закрыть и вернуться позже.")


def start_processing(source: str, *, only_local: bool, force: bool,
                     conc: int, tau: float) -> None:
    saved = settings()
    cfg_name, overrides = user_settings.ASR_PROFILES[saved.asr_profile]
    wd = jobs.start_job(
        saved.active_runs_root(), source, config_name=cfg_name,
        overrides=[*overrides, f"llm.concurrency={conc}", f"vision.tau_erase={tau}",
                   f"llm.text_model={saved.model}", f"llm.vision_model={saved.model}"],
        until="boards" if only_local else None,
        force=force,
    )
    st.session_state["open_lecture"] = wd.name
    st.query_params["t"] = "0"
    st.rerun()


def job_panel(wd: Path) -> None:
    info = jobs.status(wd)
    if not info["exists"]:
        return
    running = info["running"]
    result = info.get("result") or {}
    failed = not running and result and result.get("returncode") not in (0, None)
    if not running and not failed:
        return

    wanted = (jobs.STAGES[: jobs.STAGES.index(info["until"]) + 1]
              if info.get("until") else jobs.STAGES)
    done = sum(1 for s in wanted if info["stages"].get(s) == "done")

    with st.container(border=True):
        head, timer = st.columns([3, 1], vertical_alignment="center")
        if running:
            current = jobs.STAGE_TITLES.get(info.get("current"), "Подготовка")
            head.markdown(f"**Идёт обработка** — {current}")
        else:
            head.markdown("**Обработка завершилась с ошибкой**")
            if result.get("error"):
                st.error(result["error"])
        timer.metric("Идёт уже" if running else "Заняло",
                     format_duration(info.get("elapsed")))

        st.progress(done / len(wanted),
                    text=info.get("progress_line") or f"{done} из {len(wanted)} этапов")

        cols = st.columns(len(wanted))
        for col, stage in zip(cols, wanted):
            state = info["stages"].get(stage)
            seconds = info["stage_seconds"].get(stage)
            if state == "done":
                mark = f":material/check_circle: {format_duration(seconds)}" if seconds \
                    else ":material/check_circle:"
            elif state == "running" and running:
                mark = ":material/hourglass_top:"
            else:
                mark = ":material/radio_button_unchecked:"
            col.caption(f"{mark}  \n{jobs.STAGE_TITLES[stage]}")

        with st.expander("Журнал", expanded=bool(failed), icon=":material/terminal:"):
            st.code(info["log_tail"] or "…", language="log")
        if running and st.button("Остановить", icon=":material/stop_circle:"):
            jobs.stop_job(wd)
            st.rerun()


@st.fragment(run_every=2)
def live_job_panel(wd: Path) -> None:
    job_panel(wd)
    if not jobs.is_running(wd) and st.session_state.get(f"was_running-{wd}"):
        st.session_state[f"was_running-{wd}"] = False
        st.rerun(scope="app")
    if jobs.is_running(wd):
        st.session_state[f"was_running-{wd}"] = True


def video_panel(doc: Lecture, wd: Path) -> None:
    t = position()
    local = resolve_artifact(doc.video_path, wd, "media")
    if local is not None:
        st.video(str(local), start_time=t)
    elif is_url(doc.source):
        st.video(doc.source, start_time=t)
    else:
        st.info("Видеофайл недоступен на этом компьютере.")

    st.markdown(
        f'<div class="bp-cap">Плеер открыт на <span class="bp-pos">{to_timecode(t)}</span> '
        f"из {to_timecode(doc.duration)}. "
        "Метки времени в тексте перематывают запись сюда.</div>",
        unsafe_allow_html=True)

    if doc.segments:
        st.markdown("**Содержание**")
        rows = [f'{tc_link(s.span.start)} {s.title or "Раздел " + str(s.sid + 1)}'
                for s in doc.segments]
        st.markdown("  \n".join(rows), unsafe_allow_html=True)


def tab_notes(doc: Lecture, wd: Path) -> None:
    written = [s for s in doc.segments if s.body]
    if not written:
        st.info("Конспект ещё не сгенерирован — для этого нужен полный прогон с ключом API.")
        return

    export_bar(doc, wd)
    raw = st.toggle("Показывать исходный LaTeX", key="notes_raw")

    link_fmt = timecode_link_format()
    for i, s in enumerate(doc.segments, 1):
        with st.container(border=True):
            st.markdown(
                f'<div class="bp-sec"><h4>{i}. {s.title or "Раздел"}</h4></div>',
                unsafe_allow_html=True)
            st.caption(f"{to_timecode(s.span.start)} – {to_timecode(s.span.end)}")
            if raw:
                st.code(s.body or "", language="latex")
                continue
            body = re.sub(r"^\s*\\section\*?\{[^}]*\}", "", s.body or "", count=1)
            st.markdown(latex_to_markdown(body, heading_level=5, timecode_fmt=link_fmt),
                        unsafe_allow_html=True)
            if not timecodes(s.body or ""):
                seek_row([(to_timecode(s.span.start), s.span.start)])


@st.cache_data(show_spinner=False)
def _artifacts(run_id: str, workdir: str, stamp: float) -> dict[str, bytes | None]:
    wd = Path(workdir)
    doc = Lecture.load(wd / "lecture.json")
    return {fmt: exporting.artifact_bytes(doc, wd, fmt) for fmt in exporting.FORMATS}


def export_bar(doc: Lecture, wd: Path) -> None:
    stamp = (wd / "lecture.json").stat().st_mtime
    pdf = wd / "output" / "notes.pdf"
    ready = _artifacts(wd.name, str(wd), stamp + (pdf.stat().st_mtime if pdf.exists() else 0))

    row = st.container(horizontal=True, gap="small")
    for fmt, payload in ready.items():
        if payload is None:
            continue
        label, filename, mime = exporting.FORMATS[fmt]
        row.download_button(label, payload, filename, mime,
                            icon=":material/download:", key=f"dl-{fmt}-{wd.name}")

    with row.popover("Сохранить в папку", icon=":material/folder:"):
        save_dialog(doc, wd, ready)

    if ready["pdf"] is None:
        pdf_builder(doc, wd, wd / "output")


def save_dialog(doc: Lecture, wd: Path, ready: dict[str, bytes | None]) -> None:
    saved = settings()
    dest = st.text_input("Куда сохранить", value=str(saved.export_target()),
                         key=f"dest-{wd.name}")
    options = [f for f, payload in ready.items() if payload is not None]
    chosen = st.multiselect(
        "Форматы", options, default=[f for f in ("pdf", "tex", "md") if f in options],
        format_func=lambda f: exporting.FORMATS[f][0], key=f"fmt-{wd.name}")
    name = st.text_input("Имя файла", value=exporting.safe_filename(
        doc.title or Path(doc.source).stem), key=f"name-{wd.name}")
    remember = st.checkbox("Запомнить эту папку", value=True, key=f"rem-{wd.name}")

    if st.button("Сохранить", type="primary", icon=":material/save:",
                 disabled=not chosen, key=f"save-{wd.name}"):
        problem = user_settings.check_writable(dest)
        if problem:
            st.error(f"Папка не подходит: {problem}")
            return
        written, problems = exporting.save_to(doc, wd, dest, chosen, stem=name or None)
        if remember:
            saved.export_dir = str(Path(dest).expanduser())
            save_settings()
        for path in written:
            st.success(f"Сохранено: `{path}`")
        for text in problems:
            st.warning(text)


def pdf_builder(doc: Lecture, wd: Path, out: Path) -> None:
    from bropilot.stages.compose import assemble_tex

    engine = exporting.available_engine()
    if engine is None:
        st.caption("PDF не собран: на компьютере нет движка LaTeX. Поставьте tectonic "
                   "(`brew install tectonic`, `winget install TectonicProject.Tectonic`) "
                   "или дистрибутив TeX — LaTeX и Markdown доступны и так.")
        return

    left, right = st.columns([1, 3], vertical_alignment="center")
    right.caption(f"Конспект пока без PDF. Найден движок `{engine}`. "
                  "Сборка идёт локально, к модели не обращается и ничего не стоит.")
    if not left.button("Собрать PDF", icon=":material/picture_as_pdf:", key=f"pdf-{wd.name}"):
        return

    with st.spinner(f"Собираю PDF через {engine}…"):
        result = exporting.compile_pdf(assemble_tex(doc, out))
    exporting.write_manifest(wd, {"engine": result.engine, "ok": result.ok,
                                  "error": result.error, "at": time.time()})
    if result.ok:
        st.rerun()
    st.error(result.error)
    with st.expander("Журнал сборки", icon=":material/terminal:"):
        st.code(result.log_tail or "—", language="log")


def tab_boards(doc: Lecture, wd: Path) -> None:
    if not doc.boards:
        st.info("Кадры доски ещё не извлечены.")
        return
    only_read = st.toggle("Только распознанные", value=any(b.latex for b in doc.boards))
    boards = [b for b in doc.boards if b.latex or not only_read]
    st.caption(f"Показано {len(boards)} из {len(doc.boards)} состояний доски")
    cols = st.columns(2)
    for i, b in enumerate(boards):
        with cols[i % 2].container(border=True):
            img = resolve_artifact(b.frame_path, wd, "frames")
            if img:
                st.image(str(img), width="stretch")
            conf = f" · уверенность {b.confidence:.2f}" if b.confidence is not None else ""
            st.markdown(
                f"{tc_link(b.span.start, to_timecode(b.key_ts))} "
                f'<span style="opacity:.6;font-size:.85rem">'
                f"– {to_timecode(b.span.end)} · {b.kind}{conf}</span>",
                unsafe_allow_html=True)
            if b.latex:
                with st.expander("Что распознано"):
                    st.markdown(latex_to_markdown(b.latex, timecode_fmt=""))
                    st.code(b.latex, language="latex")
            if b.caption:
                st.caption(b.caption)


def _render_answer(text: str) -> None:
    def link(m: re.Match) -> str:
        try:
            return tc_link(from_timecode(m.group(1)), m.group(1))
        except ValueError:
            return m.group(0)

    st.markdown(CITATION.sub(link, text), unsafe_allow_html=True)


def tab_ask(doc: Lecture, wd: Path) -> None:
    if not doc.segments:
        st.info("Сначала нужна полная обработка лекции.")
        return
    from bropilot.agents.qa import QaAgent
    from bropilot.agents.summary import LENGTHS, summarize

    st.markdown("##### Пересказ")
    c1, c2, c3 = st.columns([3, 2, 1], vertical_alignment="bottom")
    focus = c1.text_input("Тема (пусто — вся лекция)", key="sum-focus",
                          placeholder="например: регуляризация")
    length = c2.selectbox("Объём", list(LENGTHS), key="sum-len")
    if c3.button("Пересказать", type="primary", icon=":material/summarize:"):
        with st.spinner("Пишу пересказ…"):
            try:
                text = summarize(doc, get_cfg(wd), focus=focus.strip() or None,
                                 length=length, cache_dir=wd,
                                 on_usage=lambda rep: pricing.record_usage(wd, "summary", rep))
                save_side(wd, "summary.json", {"text": text, "focus": focus, "length": length})
            except Exception as exc:  # noqa: BLE001
                show_error(exc)
    summary = load_side(wd, "summary.json", None)
    if summary and summary.get("text"):
        with st.container(border=True):
            _render_answer(summary["text"])

    st.markdown("##### Вопросы по лекции")
    st.caption("Ответы строятся только по материалу этой лекции, "
               "каждое утверждение помечено временем в записи.")

    history: list[dict] = load_side(wd, "chat.json", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_answer(msg["content"])
                if msg.get("warn"):
                    st.caption(msg["warn"])
            else:
                st.markdown(msg["content"])

    question = st.chat_input("Спросите что-нибудь о лекции", key=f"chat-{wd.name}")
    if question:
        history.append({"role": "user", "content": question})
        try:
            with st.spinner("Ищу в лекции и формулирую ответ…"):
                agent = QaAgent(doc, get_cfg(wd))
                ans = agent.ask(question)
                pricing.record_usage(wd, "qa", agent.client.report())
            warn = ("Модель сослалась на время вне найденных фрагментов: "
                    + ", ".join(ans.unsupported)) if ans.unsupported else ""
            history.append({"role": "assistant", "content": ans.text, "warn": warn})
            save_side(wd, "chat.json", history)
        except Exception as exc:  # noqa: BLE001
            history.pop()
            show_error(exc)
        st.rerun()

    if history and st.button("Очистить диалог", icon=":material/delete_sweep:"):
        save_side(wd, "chat.json", [])
        st.rerun()


def tab_quiz(doc: Lecture, wd: Path) -> None:
    if not doc.segments:
        st.info("Сначала нужна полная обработка лекции.")
        return
    from bropilot.agents.quiz import QuizAgent
    from bropilot.schema import QuizItem

    store = wd / "quiz.json"
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    n = c1.slider("Число вопросов", 3, 15, 8)
    if c2.button("Новый тест", type="primary", icon=":material/quiz:"):
        with st.spinner("Составляю вопросы…"):
            try:
                agent = QuizAgent(doc, get_cfg(wd))
                items = agent.generate(n)
                pricing.record_usage(wd, "quiz", agent.client.report())
                store.write_text(json.dumps([i.model_dump() for i in items],
                                            ensure_ascii=False, indent=2), encoding="utf-8")
                save_side(wd, "quiz_state.json", {})
            except Exception as exc:  # noqa: BLE001
                show_error(exc)
    if not store.exists():
        st.info("Нажмите «Новый тест», чтобы сгенерировать вопросы.")
        return
    items = [QuizItem(**d) for d in json.loads(store.read_text(encoding="utf-8"))]
    if not items:
        st.warning("Модель не вернула корректных вопросов — попробуйте ещё раз.")
        return

    state = load_side(wd, "quiz_state.json", {})
    checked = bool(state.get("checked"))
    answers: dict[str, int] = {str(k): v for k, v in (state.get("answers") or {}).items()}
    letters = "АБВГ"
    score = 0
    changed = False

    for i, item in enumerate(items):
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {item.question}**")
            st.caption(f"{item.difficulty} · {to_timecode(item.ref_ts)}")
            previous = answers.get(str(i))
            choice = st.radio("Ответ", range(len(item.options)),
                              index=previous if previous is not None else None,
                              key=f"quiz-{wd.name}-{i}",
                              format_func=lambda j, it=item: f"{letters[j]}) {it.options[j]}",
                              label_visibility="collapsed", disabled=checked)
            if choice is not None and choice != previous:
                answers[str(i)] = choice
                changed = True
            if checked:
                if previous == item.answer_index:
                    score += 1
                    st.success(f"Верно. {item.explanation}")
                else:
                    st.error(f"Правильный ответ — {letters[item.answer_index]}. "
                             f"{item.explanation}")
                st.markdown("Разбирается в записи с " + tc_link(item.ref_ts),
                            unsafe_allow_html=True)
    if changed:
        save_side(wd, "quiz_state.json", {"answers": answers, "checked": checked})

    if checked:
        st.metric("Результат", f"{score} из {len(items)}")
        if st.button("Пройти заново", icon=":material/restart_alt:"):
            save_side(wd, "quiz_state.json", {})
            for i in range(len(items)):
                st.session_state.pop(f"quiz-{wd.name}-{i}", None)
            st.rerun()
    elif st.button("Проверить ответы", type="primary", icon=":material/check:",
                   disabled=len(answers) < len(items)):
        save_side(wd, "quiz_state.json", {"answers": answers, "checked": True})
        st.rerun()
    elif len(answers) < len(items):
        st.caption(f"Отвечено {len(answers)} из {len(items)}.")


def tab_speech(doc: Lecture, wd: Path) -> None:
    if doc.segments:
        from bropilot.agents.retrieval import Retriever

        query = st.text_input("Поиск по речи и записям на доске",
                              placeholder="например: Adam", icon=":material/search:")
        if query.strip():
            try:
                hits = Retriever(doc, get_cfg(wd).retrieval).search(query, k=8)
            except Exception as exc:  # noqa: BLE001
                hits = []
                show_error(exc)
            for chunk, score in hits:
                with st.container(border=True):
                    st.markdown(f"**{chunk.title}** &nbsp; {tc_link(chunk.start, chunk.tc)} "
                                f'<span style="opacity:.55;font-size:.85rem">'
                                f"релевантность {score:.2f}</span>",
                                unsafe_allow_html=True)
                    st.write(chunk.text[:500] + ("…" if len(chunk.text) > 500 else ""))
    else:
        st.caption("Поиск станет доступен после разбиения на разделы.")

    st.markdown("##### Расшифровка речи")
    if not doc.utterances:
        st.info("Речь ещё не распознана.")
        return
    flt = st.text_input("Фильтр по тексту", key="tr-filter", icon=":material/filter_alt:")
    rows = [u for u in doc.utterances if flt.lower() in u.text.lower()] if flt else doc.utterances
    st.caption(f"{len(rows)} фраз")
    with st.container(height=520):
        st.markdown(
            "  \n".join(f"{tc_link(u.span.start)} {u.text}" for u in rows[:1500]),
            unsafe_allow_html=True)


def tab_cost(doc: Lecture, wd: Path) -> None:
    cost = pricing.run_cost(doc, wd, default_model=settings().model)
    info = jobs.status(wd)

    st.markdown("##### Сколько заняла обработка")
    c = st.columns(4)
    c[0].metric("Длительность записи", to_timecode(doc.duration))
    c[1].metric("Расшифровка", format_duration(cost.transcription_seconds),
                help="Скачивание записи и распознавание речи — стадии, которые "
                     "занимают львиную долю времени и выполняются локально.")
    c[2].metric("Вся обработка", format_duration(cost.processing_seconds),
                help="Сумма времени всех стадий. Кешированные стадии в неё не входят.")
    factor = cost.realtime_factor
    c[3].metric("Скорость", f"×{factor:.1f}" if factor else "—",
                help="Во сколько раз обработка быстрее реального времени. "
                     "×1 — минута обработки на минуту записи.")
    if info.get("elapsed") and not info["running"]:
        st.caption(f"Последний запуск занял {format_duration(info['elapsed'])} "
                   "по часам, включая ожидание и кешированные стадии.")

    if cost.stage_seconds:
        st.bar_chart({jobs.STAGE_TITLES.get(k, k): v for k, v in cost.stage_seconds.items()},
                     horizontal=True, height=220)

    st.markdown("##### Сколько стоила обработка")
    if not cost.calls:
        st.info("Обращений к API в этом прогоне не было — значит, и денег он не стоил.")
    else:
        m = st.columns(4)
        m[0].metric("Вызовов к API", cost.calls)
        m[1].metric("Токенов на входе", f"{cost.input_tokens:,}".replace(",", " "))
        m[2].metric("Токенов на выходе", f"{cost.output_tokens:,}".replace(",", " "))
        m[3].metric("Стоимость", pricing.format_usd(cost.cost_usd()))

        rows = [{
            "Стадия": jobs.STAGE_TITLES.get(r.stage, r.stage),
            "Модель": r.model,
            "Вызовов": r.calls,
            "Вход": r.input_tokens,
            "Выход": r.output_tokens,
            "Стоимость": pricing.format_usd(r.cost_usd()),
        } for r in cost.rows]
        st.dataframe(rows, hide_index=True, width="stretch")

        per_hour = cost.per_hour(cost.cost_usd())
        if per_hour:
            st.caption(f"Это {pricing.format_usd(per_hour)} за час записи. "
                       "Локальные стадии — распознавание речи и отбор кадров — "
                       "денег не стоят вовсе.")
        if cost.has_assumed_models:
            st.caption("Прогон сделан до того, как модель стала записываться в "
                       f"метаданные — цена посчитана по `{settings().model}`.")

    minutes = (doc.duration or 0) / 60 or 80
    st.markdown("##### Столько же у аналогов")
    st.caption(f"Пересчёт публичных тарифов на лекцию {minutes:.0f} мин. "
               "Источники и даты проверки — `configs/pricing.yaml`.")
    own = cost.cost_usd() if cost.calls else None
    st.dataframe([{
        "Сервис": a.title,
        "За лекцию": pricing.format_usd(own if a.model == "pay-per-use"
                                        else a.usd_per_lecture),
        "Как посчитано": ("фактический расход этой лекции" if a.model == "pay-per-use"
                          else a.basis),
        "Проверено": a.checked or "—",
    } for a in pricing.analog_prices(minutes, boards=len(doc.boards))],
        hide_index=True, width="stretch")

    with st.expander("Только распознавание речи", icon=":material/graphic_eq:"):
        st.dataframe([{
            "Сервис": r["title"],
            "За час": pricing.format_usd(r["usd_per_hour"]),
            "За эту лекцию": pricing.format_usd(r["usd_per_lecture"]),
        } for r in pricing.asr_prices(minutes)], hide_index=True, width="stretch")

    with st.expander("Технические метаданные прогона", icon=":material/data_object:"):
        st.json(doc.meta)


def lecture_header(doc: Lecture, wd: Path, cost: pricing.RunCost) -> None:
    title, actions = st.columns([5, 2], vertical_alignment="center")
    title.subheader(doc.title or Path(doc.source).name, anchor=False)

    chips = [
        (":material/schedule:", to_timecode(doc.duration)),
        (":material/segment:", f"разделов {len(doc.segments)}"),
        (":material/gesture:", f"доска {len(doc.boards)}"),
        (":material/graphic_eq:", f"реплик {len(doc.utterances)}"),
    ]
    if cost.processing_seconds:
        chips.append((":material/hourglass_bottom:",
                      f"обработка {format_duration(cost.processing_seconds)}"))
    if cost.calls:
        chips.append((":material/payments:", pricing.format_usd(cost.cost_usd())))
    row = title.container(horizontal=True, gap="small")
    for icon, text in chips:
        row.badge(text, icon=icon, color="gray")

    title.caption(f"[{doc.source}]({doc.source})" if is_url(doc.source) else f"`{doc.source}`")

    with actions.popover("Управление", icon=":material/more_horiz:", width="stretch"):
        lecture_actions(doc, wd)


def lecture_actions(doc: Lecture, wd: Path) -> None:
    size = sum(f.stat().st_size for f in wd.rglob("*") if f.is_file())
    st.caption(f"Папка `{wd}`  \nЗанимает {size / 2**20:.0f} МБ")

    missing = [jobs.STAGE_TITLES[s] for s, ok in [
        ("asr", doc.utterances), ("boards", doc.boards),
        ("compose", any(s.body for s in doc.segments)),
    ] if not ok]
    if missing and not jobs.is_running(wd):
        st.caption("Не выполнено: " + ", ".join(missing))
        if st.button("Продолжить обработку", icon=":material/resume:",
                     disabled=not api_ready(), width="stretch"):
            saved = settings()
            cfg_name, overrides = user_settings.ASR_PROFILES[saved.asr_profile]
            jobs.start_job(wd.parent, doc.source, config_name=cfg_name,
                           overrides=[*overrides, f"llm.text_model={saved.model}",
                                      f"llm.vision_model={saved.model}"])
            st.rerun()

    st.divider()
    confirm = st.checkbox("Я понимаю, что удаление необратимо", key=f"del-ok-{wd.name}")
    if st.button("Удалить лекцию", icon=":material/delete:", type="primary",
                 disabled=not confirm, width="stretch", key=f"del-{wd.name}"):
        try:
            jobs.delete_run(wd)
        except OSError as exc:
            st.error(str(exc))
            return
        st.session_state.pop("lecture", None)
        st.query_params.pop("t", None)
        st.cache_data.clear()
        st.rerun()


def page_lecture(wd: Path) -> None:
    if jobs.is_running(wd):
        live_job_panel(wd)
    else:
        job_panel(wd)

    doc = load_lecture(wd)
    if doc is None:
        if not jobs.is_running(wd):
            st.warning("Для этой лекции ещё нет результатов.")
            with st.popover("Управление", icon=":material/more_horiz:"):
                st.caption(f"Папка `{wd}`")
                if st.button("Удалить папку прогона", icon=":material/delete:", type="primary"):
                    jobs.delete_run(wd)
                    st.session_state.pop("lecture", None)
                    st.rerun()
        return

    cost = pricing.run_cost(doc, wd, default_model=settings().model)
    lecture_header(doc, wd, cost)

    left, right = st.columns([2, 3], gap="large")
    with right:
        tab = st.segmented_control("Разделы", TABS, default=TABS[0], key="tab",
                                   bind="query-params", required=True,
                                   label_visibility="collapsed",
                                   width="stretch") or TABS[0]
    remember_link_state(tab="" if tab == TABS[0] else tab)

    with left:
        video_panel(doc, wd)
    with right:
        {
            "Конспект": tab_notes,
            "Вопросы": tab_ask,
            "Тест": tab_quiz,
            "Доска": tab_boards,
            "Речь": tab_speech,
            "Стоимость": tab_cost,
        }[tab](doc, wd)


def short(title: str, limit: int = 26) -> str:
    title = " ".join(title.split())
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"


def tab_labels(runs: list[dict]) -> dict[str, str]:
    labels = {NEW: "＋"}
    used: dict[str, int] = {}
    for r in runs:
        base = short(r["title"])
        used[base] = used.get(base, 0) + 1
        labels[r["id"]] = base if used[base] == 1 else f"{base} · {r['id'][:4]}"
    return labels


def navigation(runs: list[dict]) -> str:
    options = [NEW] + [r["id"] for r in runs]
    labels = tab_labels(runs)

    pending = st.session_state.pop("open_lecture", None)
    if pending in options:
        st.session_state["lecture"] = pending

    stored = st.session_state.get("lecture")
    if stored is not None and stored not in options:
        del st.session_state["lecture"]
        stored = None

    extra: dict = {}
    if stored is None:
        extra["default"] = options[1] if len(options) > 1 else NEW

    selected = st.segmented_control(
        "Лекции", options, key="lecture", bind="query-params", required=True,
        format_func=lambda o: labels.get(o, o), label_visibility="collapsed", **extra,
    ) or NEW

    remember_link_state(lecture=labels.get(selected, ""))

    busy = [r["title"] for r in runs if r["running"]]
    if busy:
        st.caption(":material/hourglass_top: идёт обработка: " + ", ".join(busy))
    return selected


sidebar()
_runs = list_runs()
_selected = navigation(_runs)
st.divider()

if _selected == NEW:
    page_new()
else:
    _run = find_run(_runs, _selected)
    if _run is None:
        st.warning("Эта лекция больше не найдена — возможно, её удалили.")
    else:
        page_lecture(_run["wd"])
