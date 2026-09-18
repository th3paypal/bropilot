from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..schema import Lecture
from ..utils.common import get_logger, stable_hash

log = get_logger(__name__)


@dataclass
class Context:
    cfg: DictConfig
    workdir: Path
    doc: Lecture
    upstream_key: str = "root"
    stats: dict[str, Any] = field(default_factory=dict)

    def subdir(self, name: str) -> Path:
        p = self.workdir / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_dir(self) -> Path:
        return self.subdir("cache")


class Stage(ABC):
    name: str = "stage"
    config_keys: tuple[str, ...] = ()

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def apply(self, ctx: Context) -> Lecture:  # pragma: no cover
        ...

    def cache_salt(self) -> dict[str, Any]:
        return {}

    def signature(self, ctx: Context) -> str:
        relevant = {
            k: OmegaConf.to_container(self.cfg[k], resolve=True)
            for k in self.config_keys
            if k in self.cfg
        }
        return stable_hash(self.name, relevant, self.cache_salt(), ctx.upstream_key)

    def __call__(self, ctx: Context) -> Context:
        sig = self.signature(ctx)
        snapshot = ctx.cache_dir / f"{self.name}.{sig}.json"

        if snapshot.exists() and not bool(self.cfg.get("force", False)):
            log.info("[%s] cache hit (%s) — skipping", self.name, sig)
            ctx.doc = Lecture.load(snapshot)
            ctx.upstream_key = sig
            return ctx

        log.info("[%s] running (%s)", self.name, sig)
        started = time.perf_counter()
        ctx.doc = self.apply(ctx)
        elapsed = time.perf_counter() - started

        ctx.doc.meta.setdefault("timings", {})[self.name] = round(elapsed, 2)
        ctx.doc.save(snapshot)
        ctx.doc.save(ctx.workdir / "lecture.json")
        ctx.upstream_key = sig
        ctx.stats[self.name] = {"seconds": round(elapsed, 2), "signature": sig}
        log.info("[%s] done in %.1fs", self.name, elapsed)
        return ctx


class Pipeline:
    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages

    def run(self, ctx: Context, until: str | None = None) -> Context:
        for stage in self.stages:
            ctx = stage(ctx)
            if until and stage.name == until:
                log.info("stopping after `%s` as requested", until)
                break
        return ctx

    def __repr__(self) -> str:  # pragma: no cover
        return " -> ".join(s.name for s in self.stages)
