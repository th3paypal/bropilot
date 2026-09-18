#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bropilot import pricing  # noqa: E402
from bropilot import settings as user_settings  # noqa: E402
from bropilot.jobs import STAGE_TITLES  # noqa: E402
from bropilot.schema import Lecture  # noqa: E402
from bropilot.utils.common import format_duration, to_timecode  # noqa: E402

OUT = ROOT / "bench_results"

DESCRIPTION = """Таблицы времени и стоимости обработки для отчёта.

    python scripts/cost_report.py                       # по всем прогонам
    python scripts/cost_report.py --runs ~/lectures     # другое хранилище
    python scripts/cost_report.py --minutes 80          # пересчёт на лекцию
    python scripts/cost_report.py --latex               # ещё и .tex-таблицы

Результаты печатаются таблицами и сохраняются в bench_results/cost_*.csv.
Прайсы берутся из configs/pricing.yaml."""


def show(name: str, rows: list[dict], *, save: bool = True) -> None:
    if not rows:
        print(f"\n[{name}] нет данных")
        return
    keys = list(rows[0])
    widths = [max(len(str(k)), *(len(str(r.get(k, ""))) for r in rows)) for k in keys]
    print(f"\n[{name}]")
    print("  ".join(str(k).ljust(w) for k, w in zip(keys, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(r.get(k, "")).ljust(w) for k, w in zip(keys, widths)))
    if save:
        OUT.mkdir(exist_ok=True)
        path = OUT / f"cost_{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def latex_table(name: str, rows: list[dict], caption: str, label: str) -> None:
    if not rows:
        return
    keys = list(rows[0])
    spec = "".join("r" if _numeric_column(rows, k) else "l" for k in keys)
    wide = len(keys) > 6
    lines = [
        r"\begin{table}[H]", r"\centering", f"\\caption{{{caption}}}\\label{{{label}}}",
        r"\small",
    ]
    if wide:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule",
              " & ".join(_tex(str(k)) for k in keys) + r" \\", r"\midrule"]
    lines += [" & ".join(_tex(str(r.get(k, ""))) for k in keys) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if wide:
        lines.append(r"}")
    lines += [r"\end{table}", ""]
    OUT.mkdir(exist_ok=True)
    (OUT / f"cost_{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def _numeric_column(rows: list[dict], key: str) -> bool:
    values = [str(r.get(key, "")).replace("×", "").replace(",", ".").strip()
              for r in rows]
    return all(_is_number(v) for v in values if v and v != "—")


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _tex(text: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                 ("×", r"$\times$"), ("÷", r"$\div$")):
        text = text.replace(a, b)
    return text


def _num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") or "0"


def is_synthetic(doc: Lecture) -> bool:
    return (doc.meta.get("asr") or {}).get("model") == "fake"


def collect(roots: list[Path], default_model: str,
            include_synthetic: bool = False) -> list[tuple[Path, Lecture, pricing.RunCost]]:
    found, skipped = [], 0
    for root in roots:
        if not root.exists():
            continue
        for wd in sorted(root.iterdir()):
            snapshot = wd / "lecture.json"
            if not wd.is_dir() or not snapshot.exists():
                continue
            doc = Lecture.load(snapshot)
            if is_synthetic(doc) and not include_synthetic:
                skipped += 1
                continue
            found.append((wd, doc, pricing.run_cost(doc, wd, default_model=default_model)))
    if skipped:
        print(f"Пропущено синтетических прогонов (smoke-тест): {skipped}. "
              "Включить: --include-synthetic.")
    return found


def table_runs(data) -> list[dict]:
    rows = []
    for wd, doc, cost in data:
        rows.append({
            "прогон": wd.name,
            "лекция": (doc.title or Path(doc.source).stem)[:34],
            "длительность": to_timecode(doc.duration),
            "разделов": len(doc.segments),
            "кадров доски": len(doc.boards),
            "вызовов": cost.calls,
            "токенов вход": cost.input_tokens,
            "токенов выход": cost.output_tokens,
            "стоимость, $": _num(cost.cost_usd(), 4),
        })
    return rows


def table_perf(data) -> list[dict]:
    rows = []
    for wd, doc, cost in data:
        hours = cost.hours
        rows.append({
            "прогон": wd.name,
            "длительность": to_timecode(doc.duration),
            **{STAGE_TITLES.get(k, k)[:18]: _num(cost.stage_seconds.get(k, 0.0), 1)
               for k in ("ingest", "asr", "boards", "board_reading", "segmentation", "compose")},
            "всего, с": _num(cost.processing_seconds, 1),
            "мин обработки на час записи": _num(
                (cost.processing_seconds / 60 / hours) if hours else 0.0, 1),
            "быстрее реального времени": (f"×{cost.realtime_factor:.1f}"
                                          if cost.realtime_factor else "—"),
        })
    return rows


def table_per_hour(data, minutes: float) -> list[dict]:
    rows = []
    for wd, doc, cost in data:
        if not cost.calls or not doc.duration:
            continue
        scaled = cost.scaled_to(minutes)
        rows.append({
            "основа расчёта": (doc.title or Path(doc.source).stem)[:30],
            "исходная длительность": to_timecode(doc.duration),
            f"вызовов на {minutes:.0f} мин": _num(scaled["calls"], 1),
            "токенов вход": f"{scaled['input_tokens']:.0f}",
            "токенов выход": f"{scaled['output_tokens']:.0f}",
            "стоимость, $": _num(scaled["cost_usd"], 3),
            "обработка": format_duration(scaled["processing_seconds"]),
            "в т.ч. расшифровка": format_duration(scaled["transcription_seconds"]),
        })
    return rows


def table_analogs(minutes: float, boards: int, own: float | None) -> list[dict]:
    rows = []
    for item in pricing.analog_prices(minutes, boards=boards):
        price = own if item.model == "pay-per-use" else item.usd_per_lecture
        basis = ("фактический расход токенов" if item.model == "pay-per-use" else item.basis)
        rows.append({
            "сервис": item.title,
            f"за лекцию {minutes:.0f} мин, $": _num(price, 4) if price is not None else "—",
            "как посчитано": basis,
            "проверено": item.checked or "—",
        })
    return rows


def table_asr(minutes: float) -> list[dict]:
    return [{
        "сервис": r["title"],
        "за час аудио, $": _num(r["usd_per_hour"], 3),
        f"за лекцию {minutes:.0f} мин, $": _num(r["usd_per_lecture"], 3),
        "проверено": r["checked"] or "—",
    } for r in pricing.asr_prices(minutes)]


def main() -> int:
    ap = argparse.ArgumentParser(description=DESCRIPTION,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=None,
                    help="папка с прогонами (по умолчанию — настроенное хранилище)")
    ap.add_argument("--minutes", type=float, default=None,
                    help="длительность лекции для пересчёта тарифов")
    ap.add_argument("--model", default=pricing.DEFAULT_MODEL,
                    help="модель для прогонов, где она не записана в метаданных")
    ap.add_argument("--latex", action="store_true", help="сохранить ещё и .tex-таблицы")
    ap.add_argument("--include-synthetic", action="store_true",
                    help="учитывать прогоны smoke-теста (расход у них выдуманный)")
    args = ap.parse_args()

    saved = user_settings.load()
    roots = [args.runs.expanduser()] if args.runs else saved.library_roots()
    prices = pricing.load_pricing()
    minutes = args.minutes or float(prices.workload.reference_lecture_minutes)

    print(f"Python {platform.python_version()} · {platform.platform()}")
    print(f"Прогоны: {', '.join(str(r) for r in roots)}")
    print(f"Прайсы: {pricing.PRICING_FILE.name}, проверены {prices.meta.checked}")

    data = collect(roots, args.model, args.include_synthetic)
    if not data:
        print("\nНет обработанных лекций. Сначала обработайте хотя бы одну: "
              "`bropilot run <ссылка>`.")
        return 1

    billed = [d for d in data if d[2].calls]
    if not billed:
        print("\nВНИМАНИЕ: ни в одном прогоне не было обращений к API, поэтому "
              "стоимость обработки посчитать не на чем — сравнение ниже опирается "
              "только на тарифы аналогов.")

    show("runs", table_runs(data))
    show("perf", table_perf(data))

    scaled_rows = table_per_hour(data, minutes)
    show("scaled", scaled_rows)

    own_price = None
    boards = 0
    if billed:
        wd, doc, cost = max(billed, key=lambda d: d[1].duration)
        own_price = cost.scaled_to(minutes)["cost_usd"]
        boards = round(len(doc.boards) * (minutes * 60 / doc.duration)) if doc.duration else 0
        print(f"\nОснова для строки bropilot: прогон {wd.name} "
              f"({to_timecode(doc.duration)}), пересчитан на {minutes:.0f} мин.")

    show("analogs", table_analogs(minutes, boards, own_price))
    show("asr", table_asr(minutes))

    if args.latex:
        latex_table("runs", table_runs(data),
                    "Замеры обработки по прогонам", "tab:cost-runs")
        latex_table("perf", table_perf(data),
                    "Время стадий обработки", "tab:cost-perf")
        latex_table("analogs", table_analogs(minutes, boards, own_price),
                    f"Стоимость обработки лекции {minutes:.0f} мин", "tab:cost-analogs")
        latex_table("asr", table_asr(minutes),
                    "Стоимость распознавания речи", "tab:cost-asr")
        print(f"\n.tex-таблицы сохранены в {OUT}")

    print(f"\nCSV сохранены в {OUT}")
    print("Цифры аналогов устаревают — перед сдачей сверьте configs/pricing.yaml "
          "с сайтами сервисов и обновите поле `checked`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
