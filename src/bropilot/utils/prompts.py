from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .common import stable_hash

PROMPT_ROOT = Path(__file__).resolve().parents[3] / "templates" / "prompts"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPT_ROOT)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render(name: str, **kwargs) -> str:
    return _env().get_template(f"{name}.j2").render(**kwargs)


@lru_cache(maxsize=64)
def prompt_digest(name: str) -> str:
    return stable_hash((PROMPT_ROOT / f"{name}.j2").read_text(encoding="utf-8"))
