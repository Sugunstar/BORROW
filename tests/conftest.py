import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import pytest

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import borrow


class MockHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logging to keep pytest output clean

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            req_json = json.loads(body)
        except Exception:
            req_json = body
        self.server.requests.append({"path": self.path, "headers": dict(self.headers), "body": req_json})

        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        # Check configured responses
        if self.server.response_queue:
            resp_cfg = self.server.response_queue.pop(0)
        else:
            resp_cfg = {"text": "default mock reply", "finish_reason": "stop", "status": 200}

        status = resp_cfg.get("status", 200)
        if status != 200:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_body = resp_cfg.get("error_body", json.dumps({"error": f"HTTP {status}"}))
            self.wfile.write(err_body.encode("utf-8"))
            return

        delay = resp_cfg.get("delay", 0)
        if delay > 0:
            time.sleep(min(delay, 0.5))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        text = resp_cfg.get("text", "")
        finish_reason = resp_cfg.get("finish_reason", "stop")
        disconnect = resp_cfg.get("disconnect", False)

        # Stream chunk
        chunk = {
            "choices": [{
                "delta": {"content": text},
                "finish_reason": finish_reason
            }]
        }
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
        self.wfile.flush()

        if disconnect:
            return

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class MockServer(http.server.HTTPServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests = []
        self.response_queue = []


@pytest.fixture
def mock_server():
    server = MockServer(("127.0.0.1", 0), MockHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    port = server.server_port
    server.base_url = f"http://127.0.0.1:{port}"
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def cli(tmp_path, monkeypatch, mock_server):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("ASK_URL", mock_server.base_url)
    monkeypatch.setenv("ASK_KEY", "test-mock-key")
    monkeypatch.setenv("BORROW_TEST", "1")
    monkeypatch.chdir(work_dir)

    borrow.STATE["url"] = mock_server.base_url
    borrow.STATE["key"] = "test-mock-key"
    borrow.STATE["history"].clear()
    borrow.STATE["piped"] = ""

    class Runner:
        def __init__(self):
            self.home = home_dir
            self.work = work_dir
            self.server = mock_server

        def handle(self, cmd_line):
            return borrow.handle(cmd_line)

        def run_one_shot(self, args_list, input_data=None):
            env = os.environ.copy()
            env["HOME"] = str(home_dir)
            env["ASK_URL"] = mock_server.base_url
            env["ASK_KEY"] = "test-mock-key"
            env["PYTHONPATH"] = str(REPO_ROOT)
            return subprocess.run(
                [sys.executable, str(REPO_ROOT / "borrow.py")] + args_list,
                input=input_data,
                text=True,
                capture_output=True,
                cwd=str(work_dir),
                env=env
            )

    return Runner()
