from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_MATH_PATTERNS = [
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\\begin\{(?:equation\*?|align\*?|gather\*?)\}(.+?)\\end\{[^}]+\}", re.DOTALL),
    re.compile(r"(?<!\$)\$([^$]+?)\$(?!\$)"),
]

_ALIASES = {
    r"\left": "", r"\right": "", r"\limits": "", r"\!": "", r"\,": " ",
    r"\;": " ", r"\:": " ", r"\quad": " ", r"\qquad": " ", r"\displaystyle": "",
    r"\mathbb": r"\mathbf", r"\hat": r"\widehat", r"\tilde": r"\widetilde",
}

_TERM = re.compile(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-]{4,}")

_STOPWORDS = {
    "который", "которая", "которые", "поэтому", "например", "значит", "будем",
    "можно", "нужно", "также", "здесь", "тогда", "таким", "образом", "просто",
    "этого", "этому", "потому", "давайте", "получаем", "получается",
}


def extract_math(tex: str) -> list[str]:
    found: list[str] = []
    remaining = tex
    for pattern in _MATH_PATTERNS:
        for match in pattern.finditer(remaining):
            found.append(match.group(1))
    return [n for n in (normalise_math(f) for f in found) if len(n) >= 3]


def normalise_math(expr: str) -> str:
    text = expr
    for src, dst in _ALIASES.items():
        text = text.replace(src, dst)
    text = re.sub(r"\\(?:label|tag|bropilotunsure|text|mathrm)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"[{}\s]+", "", text)
    return text.strip()


def extract_terms(text: str, min_count: int = 2) -> set[str]:
    words = [w.lower() for w in _TERM.findall(text)]
    counts: dict[str, int] = {}
    for w in words:
        if w in _STOPWORDS:
            continue
        counts[w] = counts.get(w, 0) + 1
    return {w for w, c in counts.items() if c >= min_count}


@dataclass
class NotesScore:
    formula_recall: float = 0.0
    formula_precision: float = 0.0
    term_coverage: float = 0.0
    reference_formulas: int = 0
    generated_formulas: int = 0
    missing_terms: list[str] = field(default_factory=list)

    @property
    def formula_f1(self) -> float:
        p, r = self.formula_precision, self.formula_recall
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    def summary(self) -> str:
        return (
            f"формулы: recall {self.formula_recall:.2f}, precision "
            f"{self.formula_precision:.2f}, F1 {self.formula_f1:.2f} "
            f"({self.generated_formulas} против {self.reference_formulas} эталонных)\n"
            f"термины: покрытие {self.term_coverage:.2f}"
        )


def score_notes(generated: str, reference: str, *, top_missing: int = 25) -> NotesScore:
    gen_math = set(extract_math(generated))
    ref_math = set(extract_math(reference))
    matched = gen_math & ref_math

    gen_terms = extract_terms(generated, min_count=1)
    ref_terms = extract_terms(reference, min_count=2)
    covered = ref_terms & gen_terms

    return NotesScore(
        formula_recall=len(matched) / len(ref_math) if ref_math else 0.0,
        formula_precision=len(matched) / len(gen_math) if gen_math else 0.0,
        term_coverage=len(covered) / len(ref_terms) if ref_terms else 0.0,
        reference_formulas=len(ref_math),
        generated_formulas=len(gen_math),
        missing_terms=sorted(ref_terms - gen_terms)[:top_missing],
    )


def _to_boundary_vector(boundaries: list[int], length: int) -> np.ndarray:
    vec = np.zeros(length, dtype=np.int8)
    for b in boundaries:
        if 0 <= b < length:
            vec[b] = 1
    return vec


def pk(reference: list[int], hypothesis: list[int], length: int, k: int | None = None) -> float:
    ref = _to_boundary_vector(reference, length)
    hyp = _to_boundary_vector(hypothesis, length)
    if k is None:
        segments = max(1, int(ref.sum()) + 1)
        k = max(2, int(round(length / (2 * segments))))
    if length <= k:
        return 0.0
    errors = 0
    for i in range(length - k):
        same_ref = ref[i : i + k].sum() == 0
        same_hyp = hyp[i : i + k].sum() == 0
        errors += same_ref != same_hyp
    return errors / (length - k)


def window_diff(reference: list[int], hypothesis: list[int], length: int,
                k: int | None = None) -> float:
    ref = _to_boundary_vector(reference, length)
    hyp = _to_boundary_vector(hypothesis, length)
    if k is None:
        segments = max(1, int(ref.sum()) + 1)
        k = max(2, int(round(length / (2 * segments))))
    if length <= k:
        return 0.0
    errors = sum(
        int(ref[i : i + k].sum() != hyp[i : i + k].sum()) for i in range(length - k)
    )
    return errors / (length - k)
