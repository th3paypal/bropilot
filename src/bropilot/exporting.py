from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .schema import Lecture
from .utils.common import get_logger
from .utils.latex_md import lecture_to_markdown

log = get_logger(__name__)

ENGINES = ("tectonic", "latexmk", "xelatex", "lualatex")

FORMATS = {
    "pdf": ("PDF", "notes.pdf", "application/pdf"),
    "tex": ("LaTeX", "notes.tex", "text/x-tex"),
    "md": ("Markdown", "notes.md", "text/markdown"),
    "json": ("Данные лекции", "lecture.json", "application/json"),
}


def available_engine() -> str | None:
    for name in ENGINES:
        if shutil.which(name):
            return name
    return None


@dataclass
class BuildResult:
    pdf: Path | None
    engine: str | None
    log_tail: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.pdf is not None


def compile_pdf(tex_path: Path | str, out_dir: Path | str | None = None,
                *, timeout: float = 600.0) -> BuildResult:
    tex_path = Path(tex_path)
    if not tex_path.exists():
        return BuildResult(None, None, error=f"нет файла {tex_path.name}")

    tex_path = tex_path.resolve()
    out_dir = Path(out_dir).resolve() if out_dir else tex_path.parent
    build = out_dir / "build"
    build.mkdir(parents=True, exist_ok=True)

    engine = available_engine()
    if engine is None:
        return BuildResult(None, None, error=(
            "на этом компьютере нет ни одного движка LaTeX. Установите tectonic "
            "(`brew install tectonic`, `winget install TectonicProject.Tectonic`, "
            "`cargo install tectonic`) или дистрибутив TeX (MiKTeX, MacTeX, texlive-xetex)."
        ))

    commands = _commands(engine, tex_path, build)
    failure = ""
    for cmd in commands:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", check=False, timeout=timeout,
                              cwd=str(tex_path.parent))
        if proc.returncode != 0 and not failure:
            failure = (proc.stderr or proc.stdout or "").strip()

    produced = build / f"{tex_path.stem}.pdf"
    log_file = build / f"{tex_path.stem}.log"
    tail = ""
    if log_file.exists():
        tail = "\n".join(log_file.read_text(encoding="utf-8", errors="replace")
                         .splitlines()[-60:])
    elif failure:
        tail = "\n".join(failure.splitlines()[-60:])

    if not produced.exists():
        return BuildResult(None, engine, log_tail=tail,
                           error=f"{engine} не собрал PDF — см. журнал сборки")

    final = out_dir / "notes.pdf"
    shutil.copy2(produced, final)
    if failure:
        log.warning("%s reported errors, PDF kept anyway (%s)", engine, log_file)
    log.info("compiled %s with %s", final, engine)
    return BuildResult(final, engine, log_tail=tail)


def _commands(engine: str, tex_path: Path, build: Path) -> list[list[str]]:
    tex, out = str(tex_path), str(build)
    if engine == "tectonic":
        return [["tectonic", "-Z", "continue-on-errors", "--keep-logs", "--outdir", out, tex]]
    if engine == "latexmk":
        return [["latexmk", "-xelatex", "-interaction=nonstopmode", "-f",
                 f"-outdir={out}", tex]]
    return [[engine, "-interaction=nonstopmode", f"-output-directory={out}", tex]] * 2


def artifact_bytes(doc: Lecture, workdir: Path, fmt: str) -> bytes | None:
    out = Path(workdir) / "output"
    if fmt == "pdf":
        path = out / "notes.pdf"
        return path.read_bytes() if path.exists() else None
    if fmt == "tex":
        path = out / "notes.tex"
        return path.read_bytes() if path.exists() else None
    if fmt == "md":
        from .utils.common import video_link_prefix

        url = doc.source if video_link_prefix(doc.source) else None
        return lecture_to_markdown(doc, video_url=url).encode("utf-8")
    if fmt == "json":
        return doc.model_dump_json(indent=2).encode("utf-8")
    raise ValueError(f"unknown export format: {fmt!r}")


def safe_filename(title: str, fallback: str = "конспект") -> str:
    cleaned = "".join(" " if ch in '<>:"/\\|?*' else ch for ch in (title or ""))
    cleaned = " ".join(cleaned.split()).strip(" .")
    return (cleaned or fallback)[:80]


def save_to(doc: Lecture, workdir: Path, dest: Path | str, formats: list[str],
            *, stem: str | None = None) -> tuple[list[Path], list[str]]:
    dest = Path(dest).expanduser()
    written: list[Path] = []
    problems: list[str] = []
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return [], [f"не удаётся создать папку {dest}: {exc.strerror or exc}"]

    base = stem or safe_filename(doc.title or Path(doc.source).stem)
    for fmt in formats:
        label, default_name, _ = FORMATS[fmt]
        suffix = Path(default_name).suffix
        try:
            payload = artifact_bytes(doc, workdir, fmt)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{label}: {exc}")
            continue
        if payload is None:
            problems.append(f"{label}: ещё не собран")
            continue
        target = dest / f"{base}{suffix}"
        try:
            target.write_bytes(payload)
        except OSError as exc:
            problems.append(f"{label}: {exc.strerror or exc}")
            continue
        written.append(target)
    return written, problems


def write_manifest(workdir: Path, payload: dict) -> None:
    out = Path(workdir) / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "build.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
