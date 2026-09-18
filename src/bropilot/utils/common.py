from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "faster_whisper", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def to_timecode(seconds: float, with_hours: bool | None = None) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    show_hours = h > 0 if with_hours is None else with_hours
    return f"{h}:{m:02d}:{s:02d}" if show_hours else f"{m:02d}:{s:02d}"


def from_timecode(tc: str) -> float:
    parts = [float(p) for p in tc.strip().split(":")]
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"unparseable timecode: {tc!r}")
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def youtube_url_at(source: str, seconds: float) -> str | None:
    prefix = video_link_prefix(source)
    return f"{prefix}{int(seconds)}s" if prefix else None


_YOUTUBE_ID = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/|live/))([A-Za-z0-9_-]{11})"
)


def video_link_prefix(source: str) -> str | None:
    if not source:
        return None
    match = _YOUTUBE_ID.search(source)
    if match:
        return f"https://youtu.be/{match.group(1)}?t="
    if not source.startswith(("http://", "https://")):
        return None
    if "&" in source or "?" in source:
        return None
    return f"{source}?t="


def stable_hash(*parts: Any, length: int = 12) -> str:
    h = hashlib.blake2b(digest_size=32)
    for part in parts:
        if isinstance(part, (str, int, float, bool)) or part is None:
            payload = json.dumps(part, sort_keys=True, ensure_ascii=False)
        elif isinstance(part, Path):
            payload = part.name
        else:
            payload = json.dumps(part, sort_keys=True, ensure_ascii=False, default=str)
        h.update(payload.encode("utf-8"))
    return h.hexdigest()[:length]


def resolve_artifact(path: str | Path | None, workdir: str | Path, subdir: str) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p
    alt = Path(workdir) / subdir / p.name
    return alt if alt.exists() else None


def file_digest(path: str | Path, chunk: int = 1 << 20, length: int = 12) -> str:
    h = hashlib.blake2b(digest_size=32)
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()[:length]


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"`{name}` was not found in PATH. Install it and retry "
            f"(ffmpeg/ffprobe: `apt install ffmpeg`; yt-dlp: `pip install yt-dlp`)."
        )
    return path


def run(cmd: Sequence[str], *, quiet: bool = True) -> str:
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"command failed ({' '.join(cmd[:3])}...):\n{tail}")
    if not quiet and proc.stdout:
        get_logger("shell").debug(proc.stdout.strip())
    return proc.stdout


def ytdlp_command() -> list[str]:
    import importlib.util
    import sys

    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    return [require_binary("yt-dlp")]


def ffmpeg_exe() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ffmpeg not found: install it (apt/brew/winget) or `pip install imageio-ffmpeg`"
        ) from exc


def probe_duration(path: str | Path) -> float:
    if shutil.which("ffprobe"):
        out = run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
        return float(out.strip())
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not match:
        raise RuntimeError(f"could not read the duration of {path}")
    h, m, sec = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(sec)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total = int(round(max(0.0, float(seconds))))
    if total < 60:
        return f"{total} с"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} ч {m:02d} мин"
    return f"{m} мин {s:02d} с"


def chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    buf: list[Any] = []
    for it in items:
        buf.append(it)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + " …"
