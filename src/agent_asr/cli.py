import argparse
import json
import sys
from pathlib import Path

from . import API_VERSION, __version__
from .models import model_list, model_spec
from .runner import compare, isolated_transcribe, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agent-asr", description="Independent local ASR runtime")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List backends without loading model weights")
    for name in ("run", "worker", "compare"):
        sub = commands.add_parser(name)
        sub.add_argument("--request", required=True, help="UTF-8 JSON request file")
        sub.add_argument("--output", required=True, help="JSON result file")
    download = commands.add_parser("download", help="Pre-download model weights")
    download.add_argument("model", choices=list(row["id"] for row in model_list()))
    download.add_argument("--revision")
    download.add_argument("--aligner", action="store_true")
    service = commands.add_parser("serve", help="Start local asynchronous HTTP runtime")
    service.add_argument("--port", type=int, default=8766)
    service.add_argument("--data-dir")
    service.add_argument("--token-file")
    commands.add_parser("mcp", help="Run optional stdio MCP gateway; requires running HTTP service")
    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            print(json.dumps(model_list(), ensure_ascii=False, indent=2))
        elif args.command == "serve":
            from .server import serve

            serve(args.port, args.token_file, args.data_dir)
        elif args.command == "mcp":
            from .mcp_server import main as mcp_main

            mcp_main()
        elif args.command == "download":
            from .core import snapshot

            path = snapshot(model_spec(args.model).repository, args.revision)
            result = {"model": args.model, "path": path}
            if args.aligner:
                result["aligner_path"] = snapshot("Qwen/Qwen3-ForcedAligner-0.6B")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            payload = json.loads(Path(args.request).read_text(encoding="utf-8-sig"))
            if args.command == "worker":
                from .core import transcribe

                result = transcribe(payload)
            elif args.command == "run":
                result = isolated_transcribe(payload)
            else:
                result = compare(payload)
            write_json(args.output, result)
            if args.command == "compare" and any(row["status"] != "succeeded" for row in result["runs"]):
                return 2  # Partial report remains available, never masquerades as a successful benchmark.
        return 0
    except Exception as exc:
        error = {
            "api_version": API_VERSION,
            "component_version": __version__,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        if hasattr(args, "output"):
            write_json(args.output, error)
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 1
