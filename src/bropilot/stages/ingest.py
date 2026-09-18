from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

from ..schema import Lecture
from ..utils.common import (
    ffmpeg_exe,
    file_digest,
    get_logger,
    probe_duration,
    run,
    stable_hash,
    ytdlp_command,
)
from .base import Context, Stage

log = get_logger(__name__)


def _is_url(source: str) -> bool:
    return urlparse(source).scheme in {"http", "https"}


class IngestStage(Stage):
    name = "ingest"
    config_keys = ("ingest",)

    def apply(self, ctx: Context) -> Lecture:
        cfg = self.cfg.ingest
        media = ctx.subdir("media")
        source = ctx.doc.source

        if _is_url(source):
            video_path, title = self._download(source, media, cfg)
        else:
            video_path = Path(source).resolve()
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            title = video_path.stem

        audio_path = self._extract_audio(video_path, media, cfg)

        doc = ctx.doc
        doc.video_path = str(video_path)
        doc.audio_path = str(audio_path)
        doc.duration = probe_duration(video_path)
        doc.title = doc.title or title
        doc.meta["video_digest"] = file_digest(video_path)
        log.info("media ready: %.1f min of video", doc.duration / 60)
        return doc

    def _download(self, url: str, media: Path, cfg) -> tuple[Path, str]:
        ytdlp = ytdlp_command()
        slug = stable_hash(url)
        target = media / f"{slug}.mp4"
        if target.exists():
            log.info("video already downloaded: %s", target.name)
        else:
            log.info("downloading %s", url)
            run(
                [
                    *ytdlp,
                    "-f", str(cfg.ytdlp_format),
                    "--merge-output-format", "mp4",
                    "--no-playlist",
                    "--ffmpeg-location", ffmpeg_exe(),
                    "-o", str(target),
                    url,
                ]
            )
        try:
            title = run([*ytdlp, "--no-playlist", "--get-title", url]).strip()
        except RuntimeError:
            title = slug
        return target, title

    def _extract_audio(self, video: Path, media: Path, cfg) -> Path:
        ffmpeg = ffmpeg_exe()
        out = media / f"{video.stem}.{cfg.sample_rate}.wav"
        if out.exists():
            return out
        if shutil.which("ffprobe"):
            streams = run([
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", str(video),
            ]).strip()
            if not streams:
                raise RuntimeError(
                    f"{video.name} has no audio stream — the ASR stage needs one. "
                    "If the recording is video-only, supply the audio separately."
                )
        log.info("extracting audio -> %s", out.name)
        run(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-i", str(video),
                "-vn",
                "-ac", "1",
                "-ar", str(cfg.sample_rate),
                "-acodec", "pcm_s16le",
                str(out),
            ]
        )
        return out
