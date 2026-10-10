import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import model_spec


@dataclass
class Request:
    source: str
    model: str = "whisper-small"
    language: str | None = "zh"
    device: str = "cpu"
    compute_type: str | None = None
    word_timestamps: bool = True
    context: str = ""
    beam_size: int = 5
    cpu_threads: int = 0
    model_path: str | None = None
    revision: str | None = None
    aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    aligner_revision: str | None = None
    offline: bool = False

    @classmethod
    def parse(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("Request must be a JSON object")
        try:
            request = cls(**payload)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(request.source, str) or not Path(request.source).is_file():
            raise ValueError("source must be an existing local media file")
        request.source = str(Path(request.source).resolve())
        request.model = model_spec(request.model).id
        if request.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        for field in ("word_timestamps", "offline"):
            if type(getattr(request, field)) is not bool:
                raise ValueError(f"{field} must be boolean")
        for field, minimum in (("beam_size", 1), ("cpu_threads", 0)):
            value = getattr(request, field)
            if type(value) is not int or not minimum <= value <= 256:
                raise ValueError(f"{field} must be an integer between {minimum} and 256")
        for field in ("context", "aligner"):
            if not isinstance(getattr(request, field), str):
                raise ValueError(f"{field} must be a string")
        for field in ("language", "compute_type", "model_path", "revision", "aligner_revision"):
            value = getattr(request, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field} must be a nonempty string or null")
        if request.model == "r2t2" and request.compute_type not in {None, "float32", "float16", "bfloat16"}:
            raise ValueError("R2T2 supports float32/float16/bfloat16; Whisper INT8 does not apply")
        if (
            request.model == "r2t2"
            and request.device == "cpu"
            and request.compute_type in {"float16", "bfloat16"}
        ):
            raise ValueError("R2T2 CPU adapter currently supports float32 only")
        return request

    def to_dict(self):
        return asdict(self)


def validate_spans(rows, duration):
    previous = 0.0
    for row in rows:
        start, end = row["start"], row["end"]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, end)):
            raise ValueError("Non-finite timestamp from backend")
        if not 0 <= start <= end <= duration + 0.5 or start + 0.05 < previous:
            raise ValueError("Invalid or non-monotonic timestamp from backend")
        previous = start
        if not isinstance(row.get("text"), str):
            raise ValueError("Backend span text must be a string")
