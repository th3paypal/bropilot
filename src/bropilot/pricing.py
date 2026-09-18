from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .schema import Lecture
from .utils.common import get_logger

log = get_logger(__name__)

PRICING_FILE = Path(__file__).resolve().parents[2] / "configs" / "pricing.yaml"
USAGE_LEDGER = "usage.jsonl"

BILLED_STAGES = ("board_reading", "compose")


@lru_cache(maxsize=1)
def load_pricing(path: str | None = None) -> DictConfig:
    cfg = OmegaConf.load(Path(path) if path else PRICING_FILE)
    return cfg  # type: ignore[return-value]


def model_rate(model: str, pricing: DictConfig | None = None) -> tuple[float, float]:
    pricing = pricing or load_pricing()
    entry = pricing.llm.models.get(model)
    if entry is None:
        log.warning("no price for model %r in %s", model, PRICING_FILE.name)
        return 0.0, 0.0
    return float(entry.input_per_mtok), float(entry.output_per_mtok)


@dataclass
class UsageRow:
    stage: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    model_assumed: bool = False

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, pricing: DictConfig | None = None) -> float:
        rate_in, rate_out = model_rate(self.model, pricing)
        return (self.input_tokens * rate_in + self.output_tokens * rate_out) / 1e6


@dataclass
class RunCost:
    rows: list[UsageRow] = field(default_factory=list)
    duration: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def calls(self) -> int:
        return sum(r.calls for r in self.rows)

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.rows)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.rows)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def processing_seconds(self) -> float:
        return sum(self.stage_seconds.values())

    @property
    def transcription_seconds(self) -> float:
        return sum(self.stage_seconds.get(k, 0.0) for k in ("ingest", "asr"))

    def cost_usd(self, pricing: DictConfig | None = None) -> float:
        pricing = pricing or load_pricing()
        return sum(r.cost_usd(pricing) for r in self.rows)

    @property
    def hours(self) -> float:
        return self.duration / 3600 if self.duration else 0.0

    def per_hour(self, value: float) -> float | None:
        return value / self.hours if self.hours else None

    @property
    def has_assumed_models(self) -> bool:
        return any(r.model_assumed for r in self.rows)

    @property
    def models(self) -> list[str]:
        return sorted({r.model for r in self.rows if r.model})

    @property
    def realtime_factor(self) -> float | None:
        secs = self.processing_seconds
        return self.duration / secs if secs and self.duration else None

    def scaled_to(self, minutes: float, pricing: DictConfig | None = None) -> dict[str, float]:
        if not self.duration:
            return {}
        k = (minutes * 60) / self.duration
        return {
            "calls": self.calls * k,
            "input_tokens": self.input_tokens * k,
            "output_tokens": self.output_tokens * k,
            "cost_usd": self.cost_usd(pricing) * k,
            "processing_seconds": self.processing_seconds * k,
            "transcription_seconds": self.transcription_seconds * k,
        }


DEFAULT_MODEL = "claude-sonnet-5"


def read_ledger(workdir: Path | str) -> list[UsageRow]:
    path = Path(workdir) / USAGE_LEDGER
    if not path.exists():
        return []
    rows: list[UsageRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(UsageRow(
            stage=rec.get("kind", "agent"),
            model=rec.get("model", ""),
            calls=int(rec.get("calls", 0)),
            input_tokens=int(rec.get("input_tokens", 0)),
            output_tokens=int(rec.get("output_tokens", 0)),
        ))
    return rows


def record_usage(workdir: Path | str, kind: str, report: dict[str, Any]) -> None:
    path = Path(workdir) / USAGE_LEDGER
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": kind, **report}, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("could not append to %s: %s", path, exc)


def run_cost(doc: Lecture, workdir: Path | str | None = None, *,
             include_agents: bool = True, default_model: str = DEFAULT_MODEL) -> RunCost:
    rows: list[UsageRow] = []
    for stage in BILLED_STAGES:
        meta = doc.meta.get(stage) or {}
        if not meta.get("calls"):
            continue
        model = meta.get("model") or ""
        rows.append(UsageRow(
            stage=stage,
            model=model or default_model,
            model_assumed=not model,
            calls=int(meta.get("calls", 0)),
            input_tokens=int(meta.get("input_tokens", 0)),
            output_tokens=int(meta.get("output_tokens", 0)),
            seconds=float((doc.meta.get("timings") or {}).get(stage, 0.0)),
        ))
    if include_agents and workdir is not None:
        for row in read_ledger(workdir):
            if not row.model:
                row.model, row.model_assumed = default_model, True
            rows.append(row)
    return RunCost(
        rows=rows,
        duration=float(doc.duration or 0.0),
        stage_seconds={k: float(v) for k, v in (doc.meta.get("timings") or {}).items()},
    )


@dataclass
class AnalogPrice:
    key: str
    title: str
    model: str
    usd_per_lecture: float | None
    basis: str
    source: str = ""
    checked: str = ""


def analog_prices(minutes: float, *, boards: int = 0,
                  pricing: DictConfig | None = None) -> list[AnalogPrice]:
    pricing = pricing or load_pricing()
    workload = float(pricing.workload.hours_per_month)
    hours = minutes / 60
    out: list[AnalogPrice] = []

    for entry in pricing.analogs:
        kind = entry.get("model")
        price: float | None = None
        basis = entry.get("quota_note", "") or ""

        if kind == "subscription":
            monthly = float(entry.get("price_per_month") or 0.0)
            included = entry.get("included_hours_per_month")
            covered = float(included) if included is not None else workload
            price = monthly / covered * hours if covered else None
            basis = (f"${monthly:.2f}/мес ÷ {covered:g} ч"
                     + ("" if included is not None else " учебной нагрузки"))
        elif kind == "per-page":
            per_page = float(entry.get("price_per_page") or 0.0)
            price = per_page * boards
            basis = f"${per_page} × {boards} кадров доски"
        elif kind == "self-hosted":
            price = 0.0
            basis = "локальный запуск, платы нет"
        elif kind == "pay-per-use":
            price = None
            basis = "по фактическому расходу токенов"

        out.append(AnalogPrice(
            key=str(entry.key), title=str(entry.title), model=str(kind),
            usd_per_lecture=price, basis=basis,
            source=str(entry.get("source") or ""), checked=str(entry.get("checked") or ""),
        ))
    return out


def asr_prices(minutes: float, pricing: DictConfig | None = None) -> list[dict[str, Any]]:
    pricing = pricing or load_pricing()
    hours = minutes / 60
    rows = []
    for key, entry in pricing.asr.items():
        rows.append({
            "key": key,
            "title": str(entry.title),
            "usd_per_hour": float(entry.price_per_hour),
            "usd_per_lecture": float(entry.price_per_hour) * hours,
            "source": str(entry.get("source") or ""),
            "checked": str(entry.get("checked") or ""),
        })
    return rows


def format_usd(value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"
