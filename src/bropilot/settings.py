from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOME = Path(os.environ.get("BROPILOT_HOME") or (Path.home() / ".bropilot"))
SETTINGS_FILE = HOME / "settings.json"

DEFAULT_RUNS = PROJECT_ROOT / "runs"
DEFAULT_EXPORT = Path.home() / "Downloads"

ASR_PROFILES: dict[str, tuple[str, list[str]]] = {
    "Быстро — Whisper small (CPU)": ("cpu", ["asr.model=small"]),
    "Точнее — Whisper medium (CPU)": ("cpu", ["asr.model=medium"]),
    "Максимум — Whisper large-v3 (GPU)": ("default", []),
}
MODELS = ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"]


@dataclass
class Settings:
    runs_root: str = ""
    export_dir: str = ""
    known_roots: list[str] = field(default_factory=list)
    asr_profile: str = next(iter(ASR_PROFILES))
    model: str = MODELS[0]

    def active_runs_root(self) -> Path:
        env = os.environ.get("BROPILOT_RUNS")
        if env:
            return Path(env).expanduser().resolve()
        if self.runs_root:
            return Path(self.runs_root).expanduser().resolve()
        return DEFAULT_RUNS

    def library_roots(self) -> list[Path]:
        active = self.active_runs_root()
        seen = {active}
        roots = [active]
        for raw in [*self.known_roots, str(DEFAULT_RUNS)]:
            p = Path(raw).expanduser()
            if p.exists() and p.resolve() not in seen:
                seen.add(p.resolve())
                roots.append(p.resolve())
        return roots

    def export_target(self) -> Path:
        return Path(self.export_dir).expanduser() if self.export_dir else DEFAULT_EXPORT

    def set_runs_root(self, path: str | Path) -> None:
        new = Path(path).expanduser().resolve()
        previous = self.active_runs_root()
        if previous != new and str(previous) not in self.known_roots:
            self.known_roots.append(str(previous))
        self.runs_root = str(new)


def load() -> Settings:
    if not SETTINGS_FILE.exists():
        return Settings()
    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return Settings()
    known = {f for f in Settings.__dataclass_fields__}
    return Settings(**{k: v for k, v in raw.items() if k in known})


def save(settings: Settings) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def check_writable(path: str | Path) -> str | None:
    if not str(path).strip():
        return "путь пустой"
    p = Path(path).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"не удаётся создать папку: {exc.strerror or exc}"
    if not os.access(p, os.W_OK):
        return "нет прав на запись в эту папку"
    return None
