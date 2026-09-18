from __future__ import annotations

import pytest

from bropilot.evaluation import (
    extract_math,
    normalise_math,
    pk,
    score_notes,
    window_diff,
)


class TestMathExtraction:
    def test_finds_inline_and_display_math(self):
        tex = r"Пусть $x > 0$. Тогда \[ y = x^2 \] и $$ z = y + 1 $$"
        assert len(extract_math(tex)) == 3

    def test_finds_align_environments(self):
        tex = r"\begin{align} a &= b + c \\ d &= e \end{align}"
        assert extract_math(tex)

    def test_ignores_prose(self):
        assert extract_math("никакой математики здесь нет, только текст") == []


class TestMathNormalisation:
    @pytest.mark.parametrize(
        "a,b",
        [
            (r"\frac{1}{n}\sum_{i=1}^{n} x_i", r"\frac 1n \sum\limits_{i=1}^n x_i"),
            (r"\left( a + b \right)", r"(a+b)"),
            (r"\mathbb{E}[X]", r"\mathbf{E}[X]"),
        ],
    )
    def test_equivalent_spellings_collapse(self, a, b):
        assert normalise_math(a) == normalise_math(b)

    def test_different_formulas_stay_different(self):
        assert normalise_math(r"x^2 + y^2") != normalise_math(r"x^2 - y^2")


class TestScoreNotes:
    def test_identical_documents_score_one(self):
        tex = r"\section{A} Пусть $x^2 + y^2 = z^2$ — регуляризация и переобучение. " \
              r"Регуляризация важна, переобучение опасно."
        score = score_notes(tex, tex)
        assert score.formula_recall == 1.0
        assert score.formula_precision == 1.0
        assert score.term_coverage == 1.0

    def test_transcript_only_output_recovers_no_formulas(self):
        reference = r"\[ Q(w) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, a(x_i)) \]"
        speech_only = "лектор выписал функционал среднего значения потерь по выборке"
        assert score_notes(speech_only, reference).formula_recall == 0.0

    def test_partial_overlap_is_between_zero_and_one(self):
        reference = r"$a^2$ и $b^2$ и $c^2$"
        generated = r"$a^2$ и $b^2$"
        score = score_notes(generated, reference)
        assert 0.0 < score.formula_recall < 1.0
        assert score.formula_precision == 1.0

    def test_missing_terms_are_reported(self):
        reference = "бустинг бустинг и кросс-валидация кросс-валидация"
        score = score_notes("бустинг применяется часто", reference)
        assert any("валидация" in t for t in score.missing_terms)


class TestSegmentationMetrics:
    def test_perfect_segmentation_scores_zero(self):
        ref = [10, 20, 30]
        assert pk(ref, ref, 40) == 0.0
        assert window_diff(ref, ref, 40) == 0.0

    def test_near_miss_is_penalised_less_than_a_missed_boundary(self):
        ref = [20]
        near = pk(ref, [21], 40)
        missed = pk(ref, [], 40)
        assert 0 < near < missed

    def test_window_diff_notices_spurious_boundaries(self):
        ref = [20]
        assert window_diff(ref, [20, 21, 22], 40) > 0.0

    def test_metrics_stay_in_range(self):
        for hyp in ([], [5], [5, 15, 25, 35]):
            assert 0.0 <= pk([20], hyp, 40) <= 1.0
            assert 0.0 <= window_diff([20], hyp, 40) <= 1.0
