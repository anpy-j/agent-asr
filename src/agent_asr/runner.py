import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

from . import API_VERSION, __version__
from .metrics import character_errors, text_diff
from .models import model_spec
from .schema import Request


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def kill_tree(process):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        psutil.wait_procs([parent, *children], timeout=5)
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=10)


def isolated_transcribe(payload, timeout=7200):
    request = Request.parse(payload)
    key = "AGENT_ASR_R2T2_PYTHON" if request.model == "r2t2" else "AGENT_ASR_WHISPER_PYTHON"
    executable = os.environ.get(key, sys.executable)
    with tempfile.TemporaryDirectory(prefix="agent-asr-worker-") as folder:
        folder = Path(folder)
        write_json(folder / "request.json", request.to_dict())
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    executable,
                    "-m",
                    "agent_asr",
                    "worker",
                    "--request",
                    str(folder / "request.json"),
                    "--output",
                    str(folder / "result.json"),
                ],
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            peak_rss = 0
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if time.monotonic() - started > timeout:
                        raise TimeoutError("Transcription timed out")
                    try:
                        root_process = psutil.Process(process.pid)
                        tree = [root_process, *root_process.children(recursive=True)]
                        rss = 0
                        for member in tree:
                            try:
                                rss += member.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                        peak_rss = max(peak_rss, rss)
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(0.1)
            except BaseException:
                kill_tree(process)
                raise
        if not (folder / "result.json").exists():
            detail = (folder / "worker.log").read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"Worker exited {process.returncode} without a result: {detail}")
        envelope = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if process.returncode or envelope.get("error"):
            raise RuntimeError(envelope.get("error", "Worker failed"))
        if envelope.get("api_version") != API_VERSION:
            raise RuntimeError("Incompatible worker API version")
        envelope["metrics"]["peak_worker_tree_rss_bytes"] = peak_rss
        return envelope


def validate_comparison(payload):
    if not isinstance(payload, dict):
        raise ValueError("Comparison request must be an object")
    allowed = {"request", "models", "model_overrides", "reference_text", "ignore_digits", "timeout"}
    if set(payload) - allowed:
        raise ValueError(f"Unknown comparison fields: {sorted(set(payload) - allowed)}")
    base = dict(payload.get("request", {}))
    names = payload.get("models", ["whisper-small", "whisper-large-v3", "r2t2"])
    if not isinstance(names, list) or not names or len(names) > 16:
        raise ValueError("models must be a nonempty list of at most 16 model IDs")
    names = [model_spec(name).id for name in names]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate comparison models")
    if len(names) > 1 and base.get("model_path"):
        raise ValueError(
            "Use model_overrides for per-model paths; one shared model_path mislabels comparisons"
        )
    overrides = payload.get("model_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - set(names):
        raise ValueError("model_overrides keys must be canonical IDs in models")
    for override in overrides.values():
        if not isinstance(override, dict) or set(override) & {"source", "model", "language", "context"}:
            raise ValueError("model_overrides cannot change source/model/language/context")
    reference = payload.get("reference_text")
    if reference is not None and (not isinstance(reference, str) or not reference.strip()):
        raise ValueError("reference_text must be a nonempty manually corrected transcript")
    ignore_digits = payload.get("ignore_digits", False)
    if type(ignore_digits) is not bool:
        raise ValueError("ignore_digits must be boolean")
    timeout = payload.get("timeout", 7200)
    if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 86400:
        raise ValueError("timeout must be between 1 and 86400 seconds")
    # Validate all requests before loading any expensive model.
    requests = [Request.parse({**base, **overrides.get(name, {}), "model": name}).to_dict() for name in names]
    if reference is not None:
        character_errors(reference, "", ignore_digits)
    return requests, reference, ignore_digits, timeout


def compare(payload):
    requests, reference, ignore_digits, timeout = validate_comparison(payload)
    runs = []
    for request in requests:
        try:
            result = isolated_transcribe(request, timeout)
            score = (
                character_errors(reference, result["text"], ignore_digits) if reference is not None else None
            )
            runs.append({"model": request["model"], "status": "succeeded", "result": result, "score": score})
        except Exception as exc:
            runs.append({"model": request["model"], "status": "failed", "error": str(exc)})
    successful = [run for run in runs if run["status"] == "succeeded"]
    diffs = []
    for i, left in enumerate(successful):
        for right in successful[i + 1 :]:
            diffs.append(
                {
                    "left_model": left["model"],
                    "right_model": right["model"],
                    "differences": text_diff(left["result"]["text"], right["result"]["text"]),
                }
            )
    return {
        "api_version": API_VERSION,
        "component_version": __version__,
        "runs": runs,
        "diffs": diffs,
        "has_reference": reference is not None,
        "note": "Sequential isolated runs. No reference means differences only, not an accuracy ranking. "
        "Total time includes downloads and cold loading; sampled process-tree RSS excludes GPU memory.",
    }
