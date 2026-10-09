from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PATH = re.compile(r"^/v2/models/(?P<model>[^/]+)(?:/versions/(?P<version>[^/]+))?/infer$")


class MockTriton:
    def __init__(self):
        self.requests: list[dict] = []
        self.handler = lambda model, request: (404, {"error": "no handler"}, 0.0)
        self.http_handler = lambda path, body, headers: (404, {"error": "no http handler"}, 0.0)
        self.lock = threading.Lock()
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_PUT(self):
                self.do_POST()

            def do_POST(self):
                match = PATH.match(self.path)
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if not match:
                    with mock.lock:
                        mock.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                    status, payload, delay = mock.http_handler(self.path, body, dict(self.headers))
                else:
                    with mock.lock:
                        mock.requests.append({"model": match["model"], "version": match["version"], "body": body})
                    status, payload, delay = mock.handler(match["model"], body)
                if delay:
                    time.sleep(delay)
                data = json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def text_output(model: str, name: str, values: list[str], version: str = "1") -> dict:
    return {
        "model_name": model,
        "model_version": version,
        "outputs": [{"name": name, "datatype": "BYTES", "shape": [len(values)], "data": values}],
    }
