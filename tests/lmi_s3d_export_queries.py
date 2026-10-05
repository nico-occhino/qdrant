"""Export only the 4,992 Lance queries and exact top-10 tie sets for native evaluation."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import lance
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite preserved query input")
    dataset = lance.dataset(str(args.queries))
    if dataset.count_rows() != 4992:
        raise ValueError("unexpected Lance query count")
    count = 0
    boundary_ties = 0
    max_eligible = 0
    with args.output.open("xb") as output:
        output.write(b"LMIQ10M1")
        output.write(struct.pack("<II", 4992, 768))
        for batch in dataset.to_batches(columns=["query_id", "vector", "neighbors", "distances"],
                                        batch_size=32, scan_in_order=True):
            vectors = batch.column(1).values.to_numpy(zero_copy_only=False).reshape(batch.num_rows, 768)
            query_ids = batch.column(0).to_pylist()
            neighbors = batch.column(2).to_pylist()
            distances = batch.column(3).to_pylist()
            for query_id, vector, ids, scores in zip(query_ids, vectors, neighbors, distances):
                if len(ids) != 4096 or len(scores) != 4096 or not np.isfinite(vector).all():
                    raise ValueError(f"invalid query row {count}")
                threshold = scores[9]
                eligible = [int(point_id) for point_id, distance in zip(ids, scores)
                            if distance <= threshold]
                if scores[-1] == threshold:
                    raise ValueError(f"incomplete boundary tie at query {count}")
                if len(eligible) > 10:
                    boundary_ties += 1
                max_eligible = max(max_eligible, len(eligible))
                output.write(struct.pack("<Q", int(query_id)))
                output.write(np.asarray(vector, dtype="<f4").tobytes())
                output.write(struct.pack("<10Q", *map(int, ids[:10])))
                output.write(struct.pack("<I", len(eligible)))
                output.write(struct.pack(f"<{len(eligible)}Q", *eligible))
                count += 1
    if count != 4992 or boundary_ties != 104:
        raise ValueError(f"unexpected query/tie counts: {count}/{boundary_ties}")
    digest = hashlib.sha256()
    with args.output.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    metadata = {"source": str(args.queries), "queries": count, "dimension": 768,
                "boundary_tied_queries": boundary_ties, "max_tie_eligible": max_eligible,
                "bytes": args.output.stat().st_size, "sha256": digest.hexdigest(),
                "format": "magic LMIQ10M1; u32 count,d; per query u64 query_id, d f32 vector, 10 u64 conventional ids, u32 tie count, tie-count u64 ids"}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
