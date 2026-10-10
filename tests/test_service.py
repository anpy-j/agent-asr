import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agent_asr.client import Client
from agent_asr.jobs import Jobs


def wait_until(predicate, timeout=15):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for worker")


@pytest.fixture
def fake_environment(tmp_path, monkeypatch):
    # Real subprocess/FFmpeg/service, fake neural model: no model downloads.
    import wave

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000)
    model = tmp_path / "model"
    model.mkdir()
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "faster_whisper.py").write_text(
        """
import os, time
from types import SimpleNamespace as S
class WhisperModel:
    def __init__(self, *a, **k): pass
    def transcribe(self, *a, **k):
        if os.environ.get("TEST_ASR_STARTED"):
            from pathlib import Path
            Path(os.environ["TEST_ASR_STARTED"]).touch()
        time.sleep(float(os.environ.get("TEST_ASR_DELAY", "0")))
        return iter([S(start=.1,end=.9,text="女装。",words=[S(start=.1,end=.9,word="女装。")])]), S(language="zh")
""",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(fake), str(root)]))
    return {"source": str(source), "model_path": str(model), "offline": True}


def test_queue_runs_worker_and_rejects_second_owner(tmp_path, fake_environment):
    jobs = Jobs(tmp_path / "queue")
    try:
        with pytest.raises(RuntimeError, match="owns this data"):
            Jobs(tmp_path / "queue")
        job = jobs.submit(fake_environment)
        final = wait_until(
            lambda: j if (j := jobs.get(job["id"]))["status"] not in {"queued", "running"} else None
        )
        assert final["status"] == "succeeded", final
        result = jobs.result(job["id"])
        assert result["text"] == "女装。"
        assert result["words"][0]["start"] == 0.1
        assert result["metrics"]["peak_worker_tree_rss_bytes"] > 0
    finally:
        jobs.close()


def test_cancel_running_and_queued_jobs(tmp_path, fake_environment, monkeypatch):
    monkeypatch.setenv("TEST_ASR_DELAY", "30")
    marker = tmp_path / "started"
    monkeypatch.setenv("TEST_ASR_STARTED", str(marker))
    jobs = Jobs(tmp_path / "queue")
    try:
        first = jobs.submit(fake_environment)
        second = jobs.submit(fake_environment)
        wait_until(marker.exists)
        assert jobs.cancel(second["id"])["status"] == "cancelled"
        assert jobs.cancel(first["id"])["status"] == "cancelled"
        wait_until(lambda: jobs.process is None)
        assert jobs.get(first["id"])["status"] == "cancelled"
    finally:
        jobs.close()


def test_http_auth_submission_and_results(tmp_path, fake_environment):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token_file = tmp_path / "service.token"
    token = "test-only-token-01234567890123456789"
    token_file.write_text(token, encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_asr",
            "serve",
            "--port",
            str(port),
            "--data-dir",
            str(tmp_path / "http"),
            "--token-file",
            str(token_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"

    def health():
        try:
            with urllib.request.urlopen(url + "/health", timeout=1) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError):
            return None

    try:
        assert wait_until(health)["api_version"] == 1
        with pytest.raises(RuntimeError, match="401"):
            Client("wrong", url).models()
        client = Client(token, url)
        assert len(client.models()) == 3
        job = client.transcribe(fake_environment)
        final = wait_until(
            lambda: j if (j := client.job(job["id"]))["status"] not in {"queued", "running"} else None
        )
        assert final["status"] == "succeeded", final
        assert client.result(job["id"])["text"] == "女装。"
        with pytest.raises(RuntimeError, match="400"):
            client.transcribe({"source": "missing.wav"})
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_failed_comparison_keeps_partial_report(tmp_path, fake_environment):
    jobs = Jobs(tmp_path / "queue")
    try:
        job = jobs.submit(
            {
                "request": {"source": fake_environment["source"], "offline": True},
                "models": ["whisper-small", "r2t2"],
                "model_overrides": {
                    "whisper-small": {"model_path": fake_environment["model_path"]},
                    "r2t2": {"model_path": str(tmp_path / "missing-model")},
                },
            },
            "compare",
        )
        final = wait_until(
            lambda: j if (j := jobs.get(job["id"]))["status"] not in {"queued", "running"} else None
        )
        assert final["status"] == "failed"
        report = jobs.result(job["id"])
        assert report["runs"][0]["status"] == "succeeded"
        assert report["runs"][1]["status"] == "failed"
    finally:
        jobs.close()


def test_restart_marks_interrupted_running_job_failed(tmp_path):
    import sqlite3

    root = tmp_path / "recovery"
    root.mkdir()
    with sqlite3.connect(root / "jobs.sqlite3") as con:
        con.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        con.execute(
            "INSERT INTO jobs VALUES (?,?)",
            ("interrupted", json.dumps({"id": "interrupted", "status": "running", "kind": "transcribe"})),
        )
    jobs = Jobs(root)
    try:
        assert jobs.get("interrupted")["status"] == "failed"
    finally:
        jobs.close()


def test_mcp_stdio_gateway_lists_tools_and_calls_authenticated_service(tmp_path):
    pytest.importorskip("mcp")
    import asyncio

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = "mcp-test-only-token-01234567890123456789"
    environment = {**os.environ, "AGENT_ASR_TOKEN": token, "AGENT_ASR_URL": f"http://127.0.0.1:{port}"}
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_asr",
            "serve",
            "--port",
            str(port),
            "--data-dir",
            str(tmp_path / "mcp-service"),
        ],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    client = Client(token, environment["AGENT_ASR_URL"])

    def ready():
        try:
            return client.models()
        except OSError:
            return None

    async def check():
        parameters = StdioServerParameters(
            command=sys.executable, args=["-m", "agent_asr", "mcp"], env=environment
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "list_asr_models",
                    "create_transcription",
                    "compare_transcriptions",
                    "get_transcription",
                    "cancel_transcription",
                }
                response = await session.call_tool("list_asr_models", {})
                assert response.isError is False
                assert "whisper-large-v3" in str(response.content)

    try:
        wait_until(ready)
        asyncio.run(asyncio.wait_for(check(), timeout=20))
    finally:
        server.terminate()
        server.wait(timeout=10)
