#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bropilot.evaluation import score_notes  # noqa: E402
from bropilot.schema import Lecture  # noqa: E402

DESCRIPTION = """Сравнение сгенерированного конспекта с эталонным .tex.

    python scripts/evaluate.py runs/<hash>/output/notes.tex \
        --reference notes/lecture04.tex --ablation runs/<hash>/lecture.json

С --ablation тот же эталон дополнительно сравнивается с базовой линией
«только расшифровка речи»."""


def main() -> int:
    ap = argparse.ArgumentParser(description=DESCRIPTION,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("generated", type=Path, help="generated notes.tex")
    ap.add_argument("--reference", type=Path, required=True, help="human-written .tex")
    ap.add_argument("--ablation", type=Path, default=None,
                    help="lecture.json to build a transcript-only baseline from")
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    generated = args.generated.read_text(encoding="utf-8")
    reference = args.reference.read_text(encoding="utf-8")
    score = score_notes(generated, reference)

    payload = {"pipeline": score.__dict__ | {"formula_f1": score.formula_f1}}

    if args.ablation:
        doc = Lecture.load(args.ablation)
        baseline = score_notes(doc.full_text(), reference)
        payload["transcript_only"] = baseline.__dict__ | {"formula_f1": baseline.formula_f1}

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=list))
        return 0

    print(f"=== {args.generated} против {args.reference.name} ===")
    print(score.summary())
    if score.missing_terms:
        print("не найденные термины эталона:", ", ".join(score.missing_terms[:15]))
    if args.ablation:
        baseline = payload["transcript_only"]
        print("\n--- базовая линия «только расшифровка речи» ---")
        print(f"формулы: recall {baseline['formula_recall']:.2f}, "
              f"термины: покрытие {baseline['term_coverage']:.2f}")
        delta = score.formula_recall - baseline["formula_recall"]
        print(f"вклад визуальной ветки в recall формул: {delta:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
