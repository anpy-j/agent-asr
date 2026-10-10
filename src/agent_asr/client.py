import json
import urllib.error
import urllib.request
from urllib.parse import quote, urlparse


class Client:
    def __init__(self, token, url="http://127.0.0.1:8766"):
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Client supports local loopback HTTP only")
        self.url = url.rstrip("/")
        self.token = token

    def request(self, path, body=None):
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"ASR service HTTP {exc.code}: {exc.read().decode('utf-8')}") from exc

    def models(self):
        return self.request("/v1/models")

    def transcribe(self, payload):
        return self.request("/v1/transcriptions", payload)

    def compare(self, payload):
        return self.request("/v1/comparisons", payload)

    def job(self, job_id):
        return self.request(f"/v1/jobs/{quote(job_id, safe='')}")

    def result(self, job_id):
        return self.request(f"/v1/jobs/{quote(job_id, safe='')}/result")

    def cancel(self, job_id):
        return self.request(f"/v1/jobs/{quote(job_id, safe='')}/cancel", {})
