from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from .schema import Lecture
from .stages.asr import AsrStage
from .stages.base import Context, Pipeline, Stage
from .stages.board_reading import BoardReadingStage
from .stages.compose import ComposeStage
from .stages.frames import BoardTrackingStage
from .stages.ingest import IngestStage
from .stages.segmentation import SegmentationStage
from .utils.common import get_logger, stable_hash

log = get_logger(__name__)

STAGE_ORDER: list[type[Stage]] = [
    IngestStage,
    AsrStage,
    BoardTrackingStage,
    BoardReadingStage,
    SegmentationStage,
    ComposeStage,
]

STAGE_NAMES = [cls.name for cls in STAGE_ORDER]


def build_pipeline(cfg: DictConfig, skip: set[str] | None = None) -> Pipeline:
    skip = skip or set()
    unknown = skip - set(STAGE_NAMES)
    if unknown:
        raise ValueError(f"unknown stage(s) to skip: {sorted(unknown)}; known: {STAGE_NAMES}")
    return Pipeline([cls(cfg) for cls in STAGE_ORDER if cls.name not in skip])


def make_context(cfg: DictConfig, source: str) -> Context:
    root = Path(cfg.paths.workdir).expanduser().resolve() / stable_hash(source)
    root.mkdir(parents=True, exist_ok=True)

    snapshot = root / "lecture.json"
    if snapshot.exists():
        doc = Lecture.load(snapshot)
        if doc.source != source:
            doc = Lecture(source=source)
    else:
        doc = Lecture(source=source)

    log.info("workdir: %s", root)
    return Context(cfg=cfg, workdir=root, doc=doc)


def run_pipeline(
    cfg: DictConfig,
    source: str,
    *,
    until: str | None = None,
    skip: set[str] | None = None,
) -> Context:
    ctx = make_context(cfg, source)
    pipeline = build_pipeline(cfg, skip)
    log.info("pipeline: %s", pipeline)
    return pipeline.run(ctx, until=until)
