import json
from types import SimpleNamespace

import pytest

from agent_asr.metrics import character_errors, normalize
from agent_asr.runner import compare, validate_comparison
from agent_asr.schema import Request, validate_spans


def test_cer_counts_insertions_deletions_substitutions():
    assert character_errors("甲乙丙丁", "甲错丙新增")["edit_distance"] == 3
    assert character_errors("女装，150斤！", "女装150斤")["cer"] == 0
    assert character_errors("甲", "甲乙丙")["cer"] == 2
    with pytest.raises(ValueError):
        character_errors("，", "字")


def test_number_policy_is_explicit():
    assert normalize("150斤 一百五十斤", True) == "斤斤"
    assert character_errors("150斤", "一百五十斤")["cer"] > 0
    assert character_errors("150斤", "一百五十斤", True)["cer"] == 0


def test_validation_rejects_missing_sources_unknown_fields_and_fake_spans(tmp_path):
    source = tmp_path / "input.wav"
    source.write_bytes(b"placeholder")
    assert Request.parse({"source": str(source), "model": "large-v3"}).model == "whisper-large-v3"
    for extras in (
        {"typo": 1},
        {"word_timestamps": "true"},
        {"beam_size": True},
        {"device": "metal"},
        {"model": "r2t2", "compute_type": "int8"},
    ):
        with pytest.raises(ValueError):
            Request.parse({"source": str(source), **extras})
    with pytest.raises(ValueError):
        Request.parse({"source": str(tmp_path / "missing")})
    for spans in (
        [{"start": float("nan"), "end": 1, "text": "字"}],
        [{"start": 2, "end": 1, "text": "字"}],
        [{"start": 1, "end": 2, "text": "甲"}, {"start": 0, "end": 1, "text": "乙"}],
    ):
        with pytest.raises(ValueError):
            validate_spans(spans, 3)


def test_comparison_no_reference_has_no_accuracy_score_and_retains_failures(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.touch()

    def infer(request, _timeout):
        if request["model"] == "r2t2":
            raise RuntimeError("missing backend")
        return {"text": "女装" if request["model"] == "whisper-small" else "女装外套"}

    monkeypatch.setattr("agent_asr.runner.isolated_transcribe", infer)
    report = compare({"request": {"source": str(source)}})
    assert report["has_reference"] is False
    assert report["runs"][0]["score"] is None
    assert report["runs"][2]["status"] == "failed"
    assert report["diffs"][0]["differences"]
    report = compare({"request": {"source": str(source)}, "models": ["small"], "reference_text": "女装"})
    assert report["runs"][0]["score"]["cer"] == 0
    with pytest.raises(ValueError):
        validate_comparison({"request": {"source": str(source)}, "models": ["small", "whisper-small"]})
    with pytest.raises(ValueError):
        validate_comparison({"request": {"source": str(source)}, "reference_text": "!!!"})
    with pytest.raises(ValueError):
        validate_comparison({"request": {"source": str(source), "model_path": "shared-model"}})
    requests, *_ = validate_comparison(
        {"request": {"source": str(source)}, "model_overrides": {"r2t2": {"compute_type": "float32"}}}
    )
    assert requests[-1]["compute_type"] == "float32"


def test_whisper_adapter_passes_timestamps_and_cpu_config(tmp_path, monkeypatch):
    import sys
    from agent_asr.core import whisper

    calls = {}

    class Model:
        def __init__(self, path, **kwargs):
            calls.update(path=path, **kwargs)

        def transcribe(self, audio, **kwargs):
            calls.update(kwargs)
            return iter(
                [
                    SimpleNamespace(
                        start=0.2,
                        end=1,
                        text=" 女装。",
                        words=[SimpleNamespace(start=0.2, end=1, word=" 女装。")],
                    )
                ]
            ), SimpleNamespace(language="zh")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    source = tmp_path / "a.wav"
    source.touch()
    result = whisper(Request.parse({"source": str(source)}), source, "model-dir")
    assert calls["compute_type"] == "int8"
    assert calls["word_timestamps"] is True
    assert result["words"][0]["start"] == 0.2


def test_r2t2_adapter_uses_r2t2_weights_and_real_alignment(tmp_path, monkeypatch):
    import sys
    from agent_asr.core import r2t2

    calls = {}

    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.update(path=path, **kwargs)
            return cls()

        def transcribe(self, **kwargs):
            calls.update(kwargs)
            return [
                SimpleNamespace(
                    text="女装。",
                    language="Chinese",
                    time_stamps=SimpleNamespace(
                        items=[
                            SimpleNamespace(start_time=0.1, end_time=0.4, text="女"),
                            SimpleNamespace(start_time=0.4, end_time=0.8, text="装"),
                        ]
                    ),
                )
            ]

    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ASRModel=Model))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float32="f32", float16="f16", bfloat16="bf16", set_num_threads=lambda _n: None),
    )
    monkeypatch.setattr("agent_asr.core.snapshot", lambda *_args: "aligner-dir")
    source = tmp_path / "a.wav"
    source.touch()
    result = r2t2(Request.parse({"source": str(source), "model": "r2t2"}), source, "r2t2-weights")
    assert calls["path"] == "r2t2-weights"
    assert calls["language"] == "Chinese"
    assert calls["return_time_stamps"] is True
    assert calls["forced_aligner"] == "aligner-dir"
    assert result["words"][-1]["end"] == 0.8
    assert result["segments"][0]["text"] == "女装。"


def test_cli_failure_is_structured(tmp_path):
    from agent_asr.cli import main

    request = tmp_path / "request.json"
    output = tmp_path / "result.json"
    request.write_text(json.dumps({"source": "does-not-exist.wav"}), encoding="utf-8")
    assert main(["run", "--request", str(request), "--output", str(output)]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["error_type"] == "ValueError"
