import hashlib
import importlib.metadata
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from . import API_VERSION, __version__
from .models import data_dir, model_spec
from .schema import Request, validate_spans


def snapshot(repository, revision=None, offline=False):
    if Path(repository).is_dir():
        return str(Path(repository).resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repository, revision=revision, local_files_only=offline, cache_dir=str(data_dir() / "models")
    )


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_audio(source, output):
    try:
        completed = subprocess.run(
            [
                os.environ.get("AGENT_ASR_FFMPEG", "ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                source,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=7200,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg not found; install it or set AGENT_ASR_FFMPEG") from exc
    if completed.returncode:
        raise RuntimeError(f"Audio decoding failed: {completed.stderr[-2000:]}")
    with wave.open(str(output), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def whisper(request, audio, path):
    from faster_whisper import WhisperModel

    compute = request.compute_type or ("int8" if request.device == "cpu" else "float16")
    model = WhisperModel(
        path,
        device=request.device,
        compute_type=compute,
        cpu_threads=request.cpu_threads or max(1, (os.cpu_count() or 4) - 2),
    )
    decoded, info = model.transcribe(
        str(audio),
        language=request.language,
        word_timestamps=request.word_timestamps,
        beam_size=request.beam_size,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=request.context or None,
    )
    segments, words, original_text = [], [], []
    for segment in decoded:
        original_text.append(segment.text)
        segments.append({"start": segment.start, "end": segment.end, "text": segment.text.strip()})
        words.extend(
            {"start": w.start, "end": w.end, "text": w.word.strip()}
            for w in (segment.words or [])
            if w.word.strip()
        )
    return {
        "text": "".join(original_text).strip(),
        "language": info.language,
        "segments": segments,
        "words": words,
        "compute_type": compute,
        "timestamp_source": "whisper" if request.word_timestamps else "segment-only",
    }


def r2t2(request, audio, path):
    # R2T2 inherits Qwen3ASRModel.transcribe unchanged. Offline inference uses the
    # official Qwen wrapper with R2T2 weights; no streaming/vLLM dependencies.
    import torch
    from qwen_asr import Qwen3ASRModel

    compute = request.compute_type or ("float32" if request.device == "cpu" else "float16")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[compute]
    torch.set_num_threads(request.cpu_threads or max(1, (os.cpu_count() or 4) - 2))
    aligner = (
        snapshot(request.aligner, request.aligner_revision, request.offline)
        if request.word_timestamps
        else None
    )
    model = Qwen3ASRModel.from_pretrained(
        path,
        dtype=dtype,
        device_map=request.device,
        attn_implementation="eager",
        max_inference_batch_size=1,
        max_new_tokens=4096,
        forced_aligner=aligner,
        forced_aligner_kwargs={"dtype": dtype, "device_map": request.device, "attn_implementation": "eager"},
    )
    language = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean"}.get(
        request.language, request.language
    )
    result = model.transcribe(
        audio=str(audio),
        context=request.context,
        language=language,
        return_time_stamps=request.word_timestamps,
    )[0]
    words = []
    if result.time_stamps is not None:
        words = [
            {"start": float(item.start_time), "end": float(item.end_time), "text": item.text}
            for item in result.time_stamps.items
        ]
    # One aligned utterance, NOT an invented sentence split. Alignment tokens
    # can omit punctuation; full recognized text is preserved separately.
    segments = [{"start": words[0]["start"], "end": words[-1]["end"], "text": result.text}] if words else []
    return {
        "text": result.text,
        "language": result.language,
        "segments": segments,
        "words": words,
        "compute_type": compute,
        "timestamp_source": "forced-alignment" if words else "none",
        "aligner": aligner,
    }


def transcribe(payload):
    request = Request.parse(payload) if isinstance(payload, dict) else Request.parse(payload.to_dict())
    spec = model_spec(request.model)
    started = time.perf_counter()
    path = snapshot(request.model_path or spec.repository, request.revision, request.offline)
    load_source = "local-directory" if request.model_path else spec.repository
    with tempfile.TemporaryDirectory(prefix="agent-asr-") as folder:
        audio = Path(folder) / "audio.wav"
        duration = prepare_audio(request.source, audio)
        inference_started = time.perf_counter()
        result = (
            whisper(request, audio, path) if spec.backend == "faster-whisper" else r2t2(request, audio, path)
        )
        inference_seconds = time.perf_counter() - inference_started
    validate_spans(result["segments"], duration)
    validate_spans(result["words"], duration)
    if request.word_timestamps and result["text"].strip() and not result["words"]:
        raise RuntimeError(
            "Nonempty transcript has no requested word timestamps; refusing fabricated boundaries"
        )
    versions = {}
    for name in ("faster-whisper", "ctranslate2", "qwen-asr", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "api_version": API_VERSION,
        "component_version": __version__,
        "source": request.source,
        "source_sha256": fingerprint(request.source),
        "model": spec.id,
        "backend": spec.backend,
        "model_source": load_source,
        "resolved_model_path": path,
        "device": request.device,
        "duration": duration,
        "request": request.to_dict(),
        "dependencies": versions,
        "metrics": {
            "total_seconds": time.perf_counter() - started,
            "inference_seconds": inference_seconds,
            "rtf": inference_seconds / duration if duration else None,
        },
        **result,
    }
