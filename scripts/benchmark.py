from __future__ import annotations

import argparse
import csv
import gc
import platform
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

OUT = ROOT / "bench_results"


def save(name: str, rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[{name}]")
    keys = list(rows[0])
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))


def measure(fn):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / 2**20


def _old_volatility(masks):
    stack = np.stack(masks).astype(np.int8)
    return np.abs(np.diff(stack, axis=0)).mean(axis=0).astype(np.float32)


def _fake_masks(n: int, h: int, w: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    page = np.zeros((h, w), bool)
    for i in range(n):
        if i % 120 == 0:
            page[:] = False
        y, x = rng.integers(0, h - 4), rng.integers(0, w - 40)
        page[y : y + 3, x : x + 40] = True
        yield page.copy()


def bench_memory(sizes_old, sizes_new, h=540, w=960):
    from bropilot.utils.imaging import PackedMasks, volatility_map

    rows = []
    for n in sorted(set(sizes_old) | set(sizes_new)):
        row = {"frames": n, "minutes_at_1fps": round(n / 60, 1)}
        if n in sizes_old:
            def old():
                masks = list(_fake_masks(n, h, w))
                return _old_volatility(masks)
            _, t, mem = measure(old)
            row.update(old_seconds=round(t, 2), old_peak_mb=round(mem, 1))
        else:
            row.update(old_seconds="", old_peak_mb="")

        if n in sizes_new:
            def new():
                masks = PackedMasks(_fake_masks(n, h, w))
                return volatility_map(masks, smooth=1)
            _, t, mem = measure(new)
            row.update(new_seconds=round(t, 2), new_peak_mb=round(mem, 1))
        else:
            row.update(new_seconds="", new_peak_mb="")
        rows.append(row)
    save("memory", rows)


def bench_boards(minutes_list):
    from omegaconf import OmegaConf

    from bropilot.config import load_config
    from bropilot.schema import Lecture
    from bropilot.stages.base import Context
    from bropilot.stages.frames import BoardTrackingStage
    from bropilot.utils.common import probe_duration
    from smoke_test import BOARDS, make_video

    import smoke_test

    rows = []
    tmp = Path(tempfile.mkdtemp())
    cfg = load_config("cpu", [f"paths.workdir={tmp.as_posix()}"])
    for minutes in minutes_list:
        video = tmp / f"synthetic_{minutes}min.mp4"
        pages = max(1, int(minutes * 2))
        smoke_test.BOARDS = (BOARDS * pages)[:pages]
        make_video(video, seconds_per_board=30)
        doc = Lecture(source=str(video), video_path=str(video), duration=probe_duration(video))
        ctx = Context(cfg=cfg, workdir=tmp / f"w{minutes}", doc=doc)
        ctx.workdir.mkdir(exist_ok=True)
        stage = BoardTrackingStage(cfg)
        result, t, mem = measure(lambda: stage.apply(ctx))
        rows.append({
            "minutes": minutes,
            "sampled_frames": result.meta["vision"]["sampled_frames"],
            "states_found": len(result.boards),
            "states_expected": pages,
            "seconds": round(t, 2),
            "ms_per_frame": round(1000 * t / result.meta["vision"]["sampled_frames"], 2),
            "peak_mb": round(mem, 1),
        })
        video.unlink()
    smoke_test.BOARDS = BOARDS
    save("boards", rows)
    _ = OmegaConf


_VOCAB = ("градиент функционал выборка модель регуляризация норма весов признак "
          "объект ошибка дисперсия смещение шаг спуск стохастический метод "
          "оптимизация сходимость матрица вектор производная потери класс "
          "линейный дерево ансамбль бустинг кластер метрика точность полнота").split()


def _fake_utterances(n: int, seed: int = 0):
    from bropilot.schema import Span, Utterance

    rng = np.random.default_rng(seed)
    topics = [rng.choice(_VOCAB, size=8, replace=False) for _ in range(max(1, n // 60))]
    out, t = [], 0.0
    for i in range(n):
        words = rng.choice(topics[min(i // 60, len(topics) - 1)], size=12)
        dur = float(rng.uniform(3, 7))
        out.append(Utterance(uid=i, span=Span(start=t, end=t + dur), text=" ".join(words)))
        t += dur + 0.3
    return out


def bench_segmentation(sizes):
    from bropilot.config import load_config
    from bropilot.schema import Lecture
    from bropilot.stages.base import Context
    from bropilot.stages.segmentation import SegmentationStage

    cfg = load_config("cpu")
    rows = []
    warm = _fake_utterances(50)
    SegmentationStage(cfg).apply(Context(cfg=cfg, workdir=Path(tempfile.mkdtemp()),
                                         doc=Lecture(source="w", duration=warm[-1].span.end, utterances=warm)))
    for n in sizes:
        utts = _fake_utterances(n)
        doc = Lecture(source="bench", duration=utts[-1].span.end, utterances=utts)
        ctx = Context(cfg=cfg, workdir=Path(tempfile.mkdtemp()), doc=doc)
        result, t, mem = measure(lambda: SegmentationStage(cfg).apply(ctx))
        rows.append({"utterances": n, "minutes": round(utts[-1].span.end / 60, 1),
                     "segments": len(result.segments), "seconds": round(t, 3),
                     "peak_mb": round(mem, 1)})
    save("segmentation", rows)


def bench_retrieval(sizes, queries=50):
    from bropilot.agents.retrieval import Retriever
    from bropilot.config import load_config
    from bropilot.schema import Lecture, Segment, Span

    cfg = load_config("cpu")
    rows = []
    for n_chunks in sizes:
        n_utts = n_chunks * 20
        utts = _fake_utterances(n_utts, seed=1)
        seg_len = 60
        segments = [
            Segment(sid=k, span=Span(start=utts[i].span.start, end=utts[min(i + seg_len, n_utts) - 1].span.end),
                    utterance_ids=list(range(i, min(i + seg_len, n_utts))), title=f"Раздел {k}")
            for k, i in enumerate(range(0, n_utts, seg_len))
        ]
        doc = Lecture(source="bench", duration=utts[-1].span.end, utterances=utts, segments=segments)
        retriever, t_build, mem = measure(lambda: Retriever(doc, cfg.retrieval))
        rng = np.random.default_rng(2)
        qs = [" ".join(rng.choice(_VOCAB, size=3)) for _ in range(queries)]
        t0 = time.perf_counter()
        for q in qs:
            retriever.search(q)
        t_query = (time.perf_counter() - t0) / queries
        rows.append({"chunks": len(retriever.chunks), "build_seconds": round(t_build, 3),
                     "query_ms": round(1000 * t_query, 2), "peak_mb": round(mem, 1)})
    save("retrieval", rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["memory", "boards", "segment", "retrieval"])
    args = ap.parse_args()

    print(f"Python {platform.python_version()} · {platform.platform()} · {platform.processor() or '?'}")
    q = args.quick
    todo = [args.only] if args.only else ["memory", "boards", "segment", "retrieval"]
    if "memory" in todo:
        bench_memory(sizes_old=[75, 150, 300] if q else [150, 300, 600],
                     sizes_new=[75, 150, 300, 600] if q else [150, 300, 600, 1200, 2400, 4800])
    if "boards" in todo:
        bench_boards([1, 2] if q else [2, 4, 8, 16])
    if "segment" in todo:
        bench_segmentation([100, 200, 400] if q else [100, 250, 500, 1000, 2000])
    if "retrieval" in todo:
        bench_retrieval([20, 50] if q else [25, 50, 100, 200, 400])
    print(f"\nCSV сохранены в {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
