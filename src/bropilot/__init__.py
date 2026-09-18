from .config import load_config
from .pipeline import build_pipeline, run_pipeline
from .schema import BoardState, Lecture, QuizItem, Segment, Span, Utterance

__version__ = "0.1.0"
__all__ = [
    "load_config",
    "build_pipeline",
    "run_pipeline",
    "Lecture",
    "Segment",
    "BoardState",
    "Utterance",
    "QuizItem",
    "Span",
]
