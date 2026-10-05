"""Explicit bounded Lance importer regression against an isolated REST mock."""

import argparse
import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    stored = {}
    batches = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond({"status": "ok", "result": {"status": "green"}})

        def do_PUT(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            batches.append(len(body["points"]))
            for point in body["points"]:
                row = np.asarray(point["vector"], dtype=np.float32)
                row /= np.linalg.norm(row)
                stored[point["id"]] = row.tolist()
            self.respond({"status": "ok", "result": {"status": "completed"}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.respond({"status": "ok", "result": [
                {"id": row_id, "vector": stored[row_id]} for row_id in body["ids"]
            ]})

        def respond(self, value):
            encoded = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            common = [sys.executable, str(Path(__file__).with_name("lmi_laion10m_lance_stream.py")),
                      "--source", str(args.source), "--url", f"http://127.0.0.1:{server.server_port}",
                      "--collection", "mock", "--scan-batch", "80", "--upsert-batch", "32",
                      "--checkpoint", str(checkpoint), "--min-disk-free-gib", "0",
                      "--verify-samples", "8"]
            first = subprocess.run(common + ["--start", "5", "--end", "85"],
                                   capture_output=True, text=True, check=True)
            assert json.loads(first.stdout.splitlines()[-1])["verification"]["checked"] >= 2
            second = subprocess.run(common + ["--start", "5", "--end", "130"],
                                    capture_output=True, text=True, check=True)
            assert json.loads(second.stdout.splitlines()[0])["start"] == 85
            assert sorted(stored) == list(range(5, 130))
            assert batches == [32, 32, 16, 32, 13], batches
            assert json.loads(checkpoint.read_text())["next_source_row"] == 130
            print(json.dumps({"status": "pass", "source_rows": len(stored),
                              "first_id": min(stored), "last_id": max(stored),
                              "api_batches": batches, "resume_row": 85}))
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
