"""Mock-REST regression for SISAP row-to-ID mapping and resumable import."""

import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import h5py
import numpy as np


def main():
    points = {}
    batches = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = {"status": "ok", "result": {
                "points_count": len(points), "config": {"params": {"vectors": {
                    "size": 768, "datatype": "float16", "distance": "Cosine"}}}}}
            self.send_json(body)

        def do_PUT(self):
            body = self.read_json()
            batches.append(len(body["points"]))
            for point in body["points"]:
                vector = np.asarray(point["vector"], dtype=np.float32)
                vector /= np.linalg.norm(vector)
                points[int(point["id"])] = vector.astype(np.float16).astype(np.float32).tolist()
            self.send_json({"status": "ok", "result": {"status": "completed"}})

        def do_POST(self):
            body = self.read_json()
            self.send_json({"status": "ok", "result": [
                {"id": point_id, "vector": points[point_id]}
                for point_id in body["ids"] if point_id in points]})

        def read_json(self):
            return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

        def send_json(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tiny.h5"
            values = np.arange(12 * 768, dtype=np.float32).reshape(12, 768) % 17
            values = (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(np.float16)
            with h5py.File(source, "w") as handle:
                handle.create_dataset("emb", data=values)
            checkpoint = root / "checkpoint.json"
            command = [sys.executable, str(Path(__file__).with_name("lmi_sisap_hdf5_stream.py")),
                       "--source", str(source), "--url", f"http://127.0.0.1:{server.server_port}",
                       "--collection", "mock", "--checkpoint", str(checkpoint),
                       "--start", "3", "--end", "10", "--scan-batch", "3",
                       "--upsert-batch", "3", "--min-disk-free-gib", "0", "--verify-samples", "5"]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            summary = json.loads(result.stdout.splitlines()[-1])
            assert sorted(points) == list(range(4, 11)), sorted(points)
            assert batches == [3, 3, 1], batches
            assert summary["source_id_first"] == 4 and summary["source_id_last"] == 10
            assert summary["source_verification"]["checked"] == 5
            assert json.loads(checkpoint.read_text())["next_source_row"] == 10
            print(json.dumps({"status": "pass", "rows": len(points),
                              "official_ids": [min(points), max(points)],
                              "upsert_batch_sizes": batches,
                              "checkpoint_row": 10,
                              "verification_max_error": summary["source_verification"]["max_abs_error"]}))
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
