"""Optional stdio MCP gateway to the authenticated local runtime."""

import os
from pathlib import Path

from .client import Client
from .models import data_dir


def main():
    from mcp.server.fastmcp import FastMCP

    token = os.environ.get("AGENT_ASR_TOKEN")
    if not token:
        token = (
            Path(os.environ.get("AGENT_ASR_TOKEN_FILE", str(data_dir() / "service.token")))
            .read_text(encoding="utf-8")
            .strip()
        )
    client = Client(token, os.environ.get("AGENT_ASR_URL", "http://127.0.0.1:8766"))
    mcp = FastMCP("agent-asr")

    @mcp.tool()
    def list_asr_models() -> list:
        """List supported models and dependency availability in the runtime environment."""
        return client.models()

    @mcp.tool()
    def create_transcription(request: dict) -> dict:
        """Queue local media transcription; returns a job ID immediately."""
        return client.transcribe(request)

    @mcp.tool()
    def compare_transcriptions(request: dict) -> dict:
        """Queue a sequential model comparison; CER requires reference_text."""
        return client.compare(request)

    @mcp.tool()
    def get_transcription(job_id: str, include_result: bool = False) -> dict:
        """Read job status; optionally fetch successful transcript or comparison result."""
        job = client.job(job_id)
        if include_result and (
            job["status"] == "succeeded" or (job["status"] == "failed" and job["kind"] == "compare")
        ):
            job["result"] = client.result(job_id)
        return job

    @mcp.tool()
    def cancel_transcription(job_id: str) -> dict:
        """Cancel a queued/running job and terminate its worker process tree."""
        return client.cancel(job_id)

    mcp.run(transport="stdio")
