from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from bropilot.config import load_config  # noqa: E402
from bropilot.pipeline import run_pipeline  # noqa: E402
from bropilot.schema import Span, Utterance  # noqa: E402
from bropilot.stages import asr as asr_mod  # noqa: E402
from bropilot.utils import llm as llm_mod  # noqa: E402
from bropilot.utils.common import ffmpeg_exe, setup_logging  # noqa: E402

BOARDS = [
    ["Gradient descent", "w_{k+1} = w_k - eta * grad Q(w_k)", "Q(w) = 1/l * sum_i q_i(w)"],
    ["Stochastic GD", "w_{k+1} = w_k - eta * grad q_i(w_k)", "i ~ U{1..l}"],
    ["Regularization", "Q(w) + lambda * ||w||^2", "L2 -> ridge,  L1 -> lasso"],
]
SPEECH = [
    "Сегодня мы поговорим о градиентном спуске и о том, как минимизировать функционал ошибки.",
    "Градиентный спуск делает шаг против направления градиента с длиной шага эта.",
    "Функционал ошибки это среднее значение функции потерь по всем объектам выборки.",
    "Вычислять полный градиент на большой выборке дорого, поэтому используют стохастический вариант.",
    "В стохастическом градиентном спуске на каждом шаге берётся один случайный объект.",
    "Оценка градиента получается несмещённой, но у неё большая дисперсия.",
    "Теперь перейдём к регуляризации, которая борется с переобучением модели.",
    "Добавим к функционалу квадрат нормы весов с коэффициентом лямбда.",
    "Такая регуляризация называется L2, а модель называют гребневой регрессией.",
    "L1 регуляризация приводит к отбору признаков, это лассо.",
]


def make_video(path: Path, seconds_per_board: int = 30) -> None:
    w, h, fps = 960, 540, 5
    tmp = path.with_suffix(".silent.avi")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for lines in BOARDS:
        for f in range(seconds_per_board * fps):
            t = f / fps
            frame = np.full((h, w, 3), 255, np.uint8)
            cv2.rectangle(frame, (w - 170, h - 120), (w - 10, h - 10), (80, 80, 80), -1)
            cv2.circle(frame, (w - 90 + int(20 * np.sin(f)), h - 65), 25, (200, 180, 160), -1)
            visible = min(len(lines), int(t // (seconds_per_board / (len(lines) + 1))) + 1)
            for k in range(visible):
                cv2.putText(frame, lines[k], (40, 80 + 70 * k), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1 if k else 1.5, (20, 20, 20), 2, cv2.LINE_AA)
            vw.write(frame)
    vw.release()
    total = seconds_per_board * len(BOARDS)
    subprocess.run(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp),
         "-f", "lavfi", "-i", f"sine=frequency=220:duration={total}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )
    tmp.unlink()


def fake_asr(self, ctx):
    doc = ctx.doc
    step = doc.duration / len(SPEECH)
    doc.utterances = [
        Utterance(uid=i, span=Span(start=i * step, end=(i + 1) * step - 0.5), text=t,
                  avg_logprob=-0.2, no_speech_prob=0.1)
        for i, t in enumerate(SPEECH)
    ]
    doc.meta["asr"] = {"model": "fake", "raw_segments": len(SPEECH), "utterances": len(SPEECH),
                       "dropped": 0, "coverage": 0.95}
    return doc


class FakeMessages:
    def create(self, **params):
        content = params["messages"][-1]["content"]
        text = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        has_image = any(b.get("type") == "image" for b in content)
        assert "temperature" not in params, "temperature must not be sent to claude-sonnet-5"
        assert params["messages"][-1]["role"] == "user", "prefill must not be sent"
        if has_image:
            reply = {"latex": r"\[ w_{k+1} = w_k - \eta \nabla Q(w_k) \]", "caption": "формула шага",
                     "kind": "handwriting", "confidence": 0.9}
        elif "Напиши ОДИН раздел" in text:
            tc = text.split("интервал лекции ")[1].split("–")[0]
            reply = {"title": "Градиентный спуск", "latex": (
                "\\section{Градиентный спуск}\n\\ts{" + tc + "}\n"
                "\\begin{definition}\\emph{Градиентный спуск} — итерационный метод:\n"
                "\\[ w_{k+1} = w_k - \\eta \\nabla Q(w_k). \\]\\end{definition}\n"
                "\\begin{itemize}\\item шаг $\\eta > 0$;\\item $Q(w) = \\frac{1}{\\ell}\\sum_i q_i(w)$.\\end{itemize}\n"
                "\\begin{note}Полный градиент дорог — используют SGD.\\end{note}")}
        elif "вопросов с выбором" in text:
            reply = [{"question": "Что делает шаг градиентного спуска?",
                      "options": ["Идёт против градиента", "Идёт по градиенту",
                                  "Обнуляет веса", "Меняет выборку"],
                      "answer_index": 0, "explanation": "Антиградиент — направление убывания.",
                      "ref_ts": 10, "difficulty": "recall"}]
        elif "перескажи" in text or "объясни тему" in text:
            reply = "- Градиентный спуск минимизирует функционал [00:00]\n- L2-регуляризация [01:00]"
        else:
            reply = "Шаг делается против градиента: $w_{k+1} = w_k - \\eta \\nabla Q$ [00:10]."
        body = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=body)],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=300),
                               stop_reason="end_turn")


def install_fakes(real_asr: bool) -> None:
    if not real_asr:
        asr_mod.AsrStage.apply = fake_asr
    original_init = llm_mod.LlmClient.__init__

    def init(self, *a, **kw):
        kw["api_key"] = "fake"
        original_init(self, *a, **kw)
        self._client = SimpleNamespace(messages=FakeMessages())

    llm_mod.LlmClient.__init__ = init


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-asr", action="store_true", help="use Whisper tiny on CPU")
    ap.add_argument("--keep", action="store_true", help="keep the result in runs/ for the UI")
    args = ap.parse_args()
    setup_logging("INFO")

    from bropilot.exporting import available_engine

    checks = {"ffmpeg": ffmpeg_exe(), "движок LaTeX": available_engine()}
    for name, found in checks.items():
        print(f"  {name:16s} {'OK  ' + str(found) if found else 'НЕ НАЙДЕН (PDF не соберётся)'}")

    install_fakes(args.real_asr)
    media = ROOT / "uploads"
    media.mkdir(exist_ok=True)
    video = media / "smoke_test_lecture.mp4"
    if not video.exists():
        print("  генерирую синтетическое видео…")
        make_video(video)

    runs = ROOT / "runs" if args.keep else Path(tempfile.mkdtemp())
    overrides = [f"paths.workdir={runs.as_posix()}", "force=true",
                 "vision.min_state_seconds=4", "segmentation.min_seconds=20",
                 "segmentation.max_seconds=60", "segmentation.cohesion_window=2"]
    if args.real_asr:
        overrides += ["asr.model=tiny", "asr.vad_filter=false"]
    cfg = load_config("cpu", overrides)
    ctx = run_pipeline(cfg, str(video.resolve()))
    doc = ctx.doc
    doc.title = "Синтетическая лекция (smoke test)"
    doc.save(ctx.workdir / "lecture.json")

    from bropilot.agents.qa import QaAgent
    from bropilot.agents.quiz import QuizAgent

    answer = QaAgent(doc, cfg).ask("Что такое градиентный спуск?")
    quiz = QuizAgent(doc, cfg).generate(3)

    out = ctx.workdir / "output"
    results = {
        "реплик речи": len(doc.utterances),
        "состояний доски (ожидается 3)": len(doc.boards),
        "разделов": len(doc.segments),
        "разделов с текстом": sum(1 for s in doc.segments if s.body and "не удалось" not in s.body),
        "notes.tex": (out / "notes.tex").exists(),
        "notes.pdf": (out / "notes.pdf").exists(),
        "ответ агента": bool(answer.text),
        "вопросов теста": len(quiz),
    }
    print("\nРезультат:")
    for k, v in results.items():
        print(f"  {k:32s} {v}")
    ok = (len(doc.boards) >= 2 and results["разделов с текстом"] == len(doc.segments) > 0
          and results["notes.tex"] and answer.text and quiz)
    if checks["движок LaTeX"]:
        ok = ok and results["notes.pdf"]
    print("\nВСЁ РАБОТАЕТ" if ok else "\nЕСТЬ ПРОБЛЕМЫ — см. журнал выше")
    if args.keep:
        print(f"Результат лежит в {ctx.workdir} и виден в веб-интерфейсе.")
    else:
        shutil.rmtree(runs, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
