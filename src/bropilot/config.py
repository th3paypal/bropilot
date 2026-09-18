from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"

GROUPS = ("asr", "vision", "llm", "compose")


def _parse_override(item: str) -> tuple[str, object]:
    if "=" not in item:
        raise ValueError(f"override must look like key=value, got {item!r}")
    key, _, raw = item.partition("=")
    return key.strip(), OmegaConf.create(f"v: {raw}").v


def load_config(
    config_name: str = "default",
    overrides: Sequence[str] | None = None,
    extra_paths: Iterable[Path] = (),
) -> DictConfig:
    base_path = CONFIG_ROOT / f"{config_name}.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"no config `{config_name}` in {CONFIG_ROOT}")
    cfg = OmegaConf.load(base_path)

    defaults = cfg.pop("defaults", {}) or {}
    composed = OmegaConf.create({})
    for group in GROUPS:
        choice = defaults.get(group)
        if not choice:
            continue
        path = CONFIG_ROOT / group / f"{choice}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"config group file not found: {path}")
        composed[group] = OmegaConf.load(path)

    cfg = OmegaConf.merge(composed, cfg)

    for path in extra_paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))

    for item in overrides or []:
        key, value = _parse_override(item)
        OmegaConf.update(cfg, key, value, merge=True)

    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def describe(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)
