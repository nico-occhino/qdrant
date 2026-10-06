"""Exercise SISAP10M state transitions against an isolated in-memory REST mock."""

import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main():
    state = {"points_count": 0, "segments_count": 8, "indexed_vectors_count": 0,
             "status": "green", "optimizer_status": "ok", "config": {}}
    patches = []

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/collections":
                self.send_json({"status": "ok", "result": {"collections": [
                    {"name": "sisap2023_10m_f16_lmi"}] if state["config"] else []}})
            else:
                self.send_json({"status": "ok", "result": state})

        def do_PUT(self):
            state["config"] = {"params": json.loads(self.rfile.read(
                int(self.headers["Content-Length"]))) }
            body = state["config"]["params"]
            state["config"] = {"params": {"vectors": body["vectors"]},
                               "optimizer_config": body["optimizers_config"]}
            self.send_json({"status": "ok"})

        def do_PATCH(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["config"]["optimizer_config"].update(body["optimizers_config"])
            if "max_segment_size" in body["optimizers_config"]:
                state["segments_count"] = 1
            patches.append(body)
            self.send_json({"status": "ok"})

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_file, evidence = root / "state.json", root / "admission.json"
            script = Path(__file__).with_name("lmi_sisap10m_guard.py")
            common = [sys.executable, str(script), "--url",
                      f"http://127.0.0.1:{server.server_port}", "--collection",
                      "sisap2023_10m_f16_lmi", "--state", str(state_file),
                      "--evidence", str(evidence)]
            subprocess.run(common + ["create"], capture_output=True, text=True, check=True)
            early = subprocess.run(common + ["admit-build", "--segments-dir", str(root)],
                                    capture_output=True, text=True)
            assert early.returncode != 0 and not patches
            state["points_count"] = 10_120_191
            checkpoint = root / "checkpoint.json"
            checkpoint.write_text(json.dumps({"source": "/mnt/c/datasets/sisap2023/"
                "laion2B-en-clip768v2-n=10M.h5", "start": 0, "end": 10_120_191,
                "next_source_row": 10_120_191,
                "source_verification": {"checked": 3, "tolerance": 0.002}}))
            subprocess.run(common + ["verify-ingest", "--checkpoint", str(checkpoint)],
                           capture_output=True, text=True, check=True)
            subprocess.run(common + ["consolidate", "--wait-seconds", "1"],
                           capture_output=True, text=True, check=True)
            subprocess.run(common + ["admit-build", "--segments-dir", str(root)],
                           capture_output=True, text=True, check=True)
            assert len(patches) == 2, patches
            assert patches[0]["optimizers_config"]["max_segment_size"] == 32_000_000
            assert patches[1]["optimizers_config"]["indexing_threshold"] == 1
            assert json.loads(state_file.read_text())["phase"] == "build_enabled_once"
            print(json.dumps({"status": "pass", "blocked_early_build": True,
                              "ordered_transitions": ["create", "verify-ingest",
                                  "consolidate", "admit-build"],
                              "merge_ceiling_patch": patches[0], "single_build_patch": patches[1]}))
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
