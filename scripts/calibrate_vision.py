#!/usr/bin/env python

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bropilot.config import load_config  # noqa: E402
from bropilot.pipeline import make_context  # noqa: E402
from bropilot.stages.frames import BoardTrackingStage  # noqa: E402
from bropilot.stages.ingest import IngestStage  # noqa: E402
from bropilot.utils.common import from_timecode, setup_logging, to_timecode  # noqa: E402

DESCRIPTION = """Подбор порогов tau_erase и tau_jump по размеченным вручную моментам.

    python scripts/calibrate_vision.py lecture.mp4 --marks marks.txt

В файле разметки — по одному MM:SS на строку, по одному на каждую смену доски."""


def match(predicted: list[float], truth: list[float], tolerance: float) -> tuple[float, float]:
    used: set[int] = set()
    hits = 0
    for t in truth:
        for i, p in enumerate(predicted):
            if i not in used and abs(p - t) <= tolerance:
                used.add(i)
                hits += 1
                break
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(truth) if truth else 0.0
    return precision, recall


def main() -> int:
    ap = argparse.ArgumentParser(description=DESCRIPTION,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--marks", required=True, help="file with one MM:SS per board change")
    ap.add_argument("--tolerance", type=float, default=6.0)
    ap.add_argument("--erase", nargs="*", type=float,
                    default=[0.004, 0.007, 0.010, 0.015, 0.022])
    ap.add_argument("--jump", nargs="*", type=float, default=[0.04, 0.06, 0.09])
    args = ap.parse_args()

    setup_logging("WARNING")
    truth = [
        from_timecode(line.strip())
        for line in Path(args.marks).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"{len(truth)} размеченных смен доски, допуск ±{args.tolerance:.0f} с\n")

    header = f"{'tau_erase':>10} {'tau_jump':>9} {'states':>7} {'P':>6} {'R':>6} {'F1':>6}"
    print(header)
    print("-" * len(header))

    best = None
    for tau_erase, tau_jump in itertools.product(args.erase, args.jump):
        cfg = load_config(
            "default",
            [f"vision.tau_erase={tau_erase}", f"vision.tau_jump={tau_jump}", "force=true"],
        )
        ctx = make_context(cfg, args.source)
        ctx = IngestStage(cfg)(ctx)
        ctx = BoardTrackingStage(cfg)(ctx)

        predicted = [b.span.start for b in ctx.doc.boards]
        precision, recall = match(predicted, truth, args.tolerance)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{tau_erase:>10.3f} {tau_jump:>9.3f} {len(predicted):>7} "
              f"{precision:>6.2f} {recall:>6.2f} {f1:>6.2f}")
        if best is None or f1 > best[0]:
            best = (f1, tau_erase, tau_jump, predicted)

    if best:
        f1, tau_erase, tau_jump, predicted = best
        print(f"\nлучшее: vision.tau_erase={tau_erase} vision.tau_jump={tau_jump} (F1 {f1:.2f})")
        missed = [t for t in truth if not any(abs(p - t) <= args.tolerance for p in predicted)]
        if missed:
            print("пропущенные смены:", ", ".join(to_timecode(t) for t in missed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
