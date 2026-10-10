import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .models import data_dir
from .runner import kill_tree, write_json
from .schema import Request


class Jobs:
    """Durable serial queue; inference lives in cancellable child processes."""

    def __init__(self, root=None):
        self.root = Path(root or data_dir())
        self.root.mkdir(parents=True, exist_ok=True)
        self.guard = (self.root / "queue.lock").open("a+b")
        self.guard.seek(0)
        self.guard.write(b"0")
        self.guard.flush()
        self.guard.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.guard.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.guard.close()
            raise RuntimeError(
                "Another ASR service owns this data directory; reuse it or choose a different root"
            ) from exc
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.process = None
        with self.connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
            for row in con.execute("SELECT data FROM jobs").fetchall():
                job = json.loads(row[0])
                if job["status"] == "running":
                    job.update(status="failed", error="Service stopped during inference; submit a new job")
                    self.save(job, con)
        self.thread = threading.Thread(target=self.run, daemon=True, name="agent-asr-queue")
        self.thread.start()
        self.wake.set()

    def connect(self):
        return sqlite3.connect(self.root / "jobs.sqlite3", timeout=30)

    def save(self, job, con=None):
        if con is None:
            with self.connect() as con:
                self.save(job, con)
        else:
            con.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)", (job["id"], json.dumps(job)))

    def get(self, job_id):
        with self.connect() as con:
            row = con.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError("Job not found")
        return json.loads(row[0])

    def submit(self, payload, kind="transcribe"):
        if kind == "transcribe":
            payload = Request.parse(payload).to_dict()
        elif kind != "compare":
            raise ValueError("Unknown job kind")
        else:
            from .runner import validate_comparison

            validate_comparison(payload)
        job_id = uuid.uuid4().hex
        folder = self.root / "jobs" / job_id
        folder.mkdir(parents=True)
        write_json(folder / "request.json", payload)
        job = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": time.time(),
            "result_path": str(folder / "result.json"),
            "log_path": str(folder / "worker.log"),
        }
        with self.lock:
            self.save(job)
            self.wake.set()
        return job

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job["status"] in {"queued", "running"}:
                job.update(status="cancelled", finished_at=time.time())
                self.save(job)
                if self.process and self.process[0] == job_id:
                    kill_tree(self.process[1])
            return job

    def result(self, job_id):
        job = self.get(job_id)
        path = Path(job["result_path"])
        if job["status"] in {"succeeded", "failed"} and path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] == "succeeded" or (job["kind"] == "compare" and "runs" in result):
                return result
        raise ValueError(f"Job status is {job['status']}; no result report available")

    def run(self):
        while not self.stop.is_set():
            self.wake.wait(0.5)
            self.wake.clear()
            with self.lock:
                if self.stop.is_set():
                    break
                with self.connect() as con:
                    jobs = [json.loads(row[0]) for row in con.execute("SELECT data FROM jobs ORDER BY rowid")]
                job = next((row for row in jobs if row["status"] == "queued"), None)
                if not job:
                    continue
                job.update(status="running", started_at=time.time())
                self.save(job)
            folder = Path(job["result_path"]).parent
            command = "run" if job["kind"] == "transcribe" else "compare"
            try:
                with (folder / "worker.log").open("w", encoding="utf-8") as log:
                    with self.lock:
                        if self.get(job["id"])["status"] == "cancelled" or self.stop.is_set():
                            continue
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "agent_asr",
                                command,
                                "--request",
                                str(folder / "request.json"),
                                "--output",
                                job["result_path"],
                            ],
                            stdout=log,
                            stderr=log,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                        )
                        self.process = (job["id"], process)
                    code = process.wait()
                with self.lock:
                    job = self.get(job["id"])
                    if job["status"] == "running":
                        job.update(status="succeeded" if code == 0 else "failed", finished_at=time.time())
                        if code:
                            result_file = Path(job["result_path"])
                            job["error"] = (
                                (
                                    json.loads(result_file.read_text(encoding="utf-8")).get("error")
                                    or "Some comparison models failed; see result report"
                                )
                                if result_file.exists()
                                else f"Worker exited {code}; see log"
                            )
                        self.save(job)
            except Exception as exc:
                with self.lock:
                    job = self.get(job["id"])
                    if job["status"] == "running":
                        job.update(status="failed", error=str(exc), finished_at=time.time())
                        self.save(job)
            finally:
                with self.lock:
                    self.process = None
                    self.wake.set()

    def close(self):
        self.stop.set()
        self.wake.set()
        with self.lock:
            if self.process:
                job = self.get(self.process[0])
                if job["status"] == "running":
                    job.update(status="failed", error="Service shutting down", finished_at=time.time())
                    self.save(job)
                kill_tree(self.process[1])
        self.thread.join(timeout=15)
        self.guard.close()
