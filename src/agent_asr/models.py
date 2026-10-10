import importlib.util
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    id: str
    backend: str
    repository: str
    timestamp_source: str
    install_extra: str


MODELS = {
    s.id: s
    for s in (
        ModelSpec("whisper-small", "faster-whisper", "Systran/faster-whisper-small", "whisper", "whisper"),
        ModelSpec(
            "whisper-large-v3", "faster-whisper", "Systran/faster-whisper-large-v3", "whisper", "whisper"
        ),
        ModelSpec("r2t2", "qwen-transformers", "netease-youdao/Confucius4-R2T2", "forced-alignment", "r2t2"),
    )
}
ALIASES = {"small": "whisper-small", "large-v3": "whisper-large-v3", "R2T2": "r2t2"}


def model_spec(name):
    name = ALIASES.get(name, name)
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODELS)}")
    return MODELS[name]


def data_dir():
    base = os.environ.get("AGENT_ASR_HOME")
    if base:
        return Path(base).expanduser().resolve()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "agent-asr"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "agent-asr"


def model_list():
    result = []
    for spec in MODELS.values():
        module = "faster_whisper" if spec.backend == "faster-whisper" else "qwen_asr"
        result.append(
            {
                **asdict(spec),
                "dependency_installed": importlib.util.find_spec(module) is not None,
                "worker_python": os.environ.get(
                    "AGENT_ASR_R2T2_PYTHON" if spec.id == "r2t2" else "AGENT_ASR_WHISPER_PYTHON"
                ),
                "note": "dependency_installed describes this Python environment, not model weights",
            }
        )
    return result
