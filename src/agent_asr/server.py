import hmac
import json
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import API_VERSION, __version__
from .jobs import Jobs
from .models import data_dir, model_list


def serve(port=8766, token_file=None, root=None):
    root = Path(root or data_dir())
    root.mkdir(parents=True, exist_ok=True)
    token_path = Path(token_file or root / "service.token")
    token = os.environ.get("AGENT_ASR_TOKEN")
    if not token:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        if not token_path.exists():
            descriptor = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(secrets.token_urlsafe(32))
        token = token_path.read_text(encoding="utf-8").strip()
    if len(token) < 24:
        raise ValueError("Service token must contain at least 24 characters")
    jobs = Jobs(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Do not log Authorization headers or local source paths.

        def send_json(self, code, payload):
            content = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def authorized(self):
            if hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                return True
            self.send_json(401, {"error": "Authentication required"})
            return False

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                self.send_json(200, {"api_version": API_VERSION, "component_version": __version__})
                return
            if not self.authorized():
                return
            try:
                if path == "/v1/models":
                    result = model_list()
                elif path.startswith("/v1/jobs/"):
                    parts = path.strip("/").split("/")
                    if len(parts) == 3:
                        result = jobs.get(parts[2])
                    elif len(parts) == 4 and parts[3] == "result":
                        result = jobs.result(parts[2])
                    else:
                        raise KeyError("Endpoint not found")
                else:
                    raise KeyError("Endpoint not found")
                self.send_json(200, result)
            except KeyError as exc:
                self.send_json(404, {"error": str(exc)})
            except ValueError as exc:
                self.send_json(409, {"error": str(exc)})

        def do_POST(self):
            if not self.authorized():
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2 * 1024 * 1024:
                    raise ValueError("JSON request body must be between 1 byte and 2 MiB")
                payload = json.loads(self.rfile.read(size))
                path = urlparse(self.path).path
                if path in {"/v1/transcriptions", "/v1/comparisons"}:
                    result = jobs.submit(payload, "compare" if path.endswith("comparisons") else "transcribe")
                    self.send_json(202, result)
                elif path.startswith("/v1/jobs/") and path.endswith("/cancel"):
                    parts = path.strip("/").split("/")
                    if len(parts) != 4:
                        raise KeyError("Endpoint not found")
                    self.send_json(200, jobs.cancel(parts[2]))
                else:
                    raise KeyError("Endpoint not found")
            except KeyError as exc:
                self.send_json(404, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)})

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.daemon_threads = True
        print(f"agent-asr {__version__} listening on http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if "server" in locals():
            server.server_close()
        jobs.close()
