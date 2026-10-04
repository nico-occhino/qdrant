# LAION10M remote-storage gate (not executed)

The [official SISAP 2024 dataset page](https://sisap-challenges.github.io/2024/datasets/) lists the 768-dimensional float16 10M file as approximately 15 GB with MD5 `c05e4b1d2b2a0c7663ac9767753e25e1`. It also lists 10k public queries (about 30 MB; MD5 `f8f3e61bd22d7d64234a0f587ead9fcf`) and the matching 10M gold standard (about 77 MB; MD5 `342794391dafed7bd90dabb740fc15ba`). Gold-standard HDF5 uses `knns` and `dists`, with 1-based IDs and 1,000 neighbors per query. The challenge task uses dot-product similarity; cosine normalization in Qdrant is close but should be assessed against the official raw-dot ground truth, not silently treated as identical.

No authorized remote scratch destination has been established. The links for query and gold-standard data returned gateway errors during this preparation; verify live access before launching a transfer. Nothing was downloaded to the laptop. Use the following only after choosing an explicit, authorized remote scratch directory with comfortably more than 60 GB free, preferably over 100 GB once Qdrant f32 vectors, WAL, optimizer scratch, and index files are included.

```bash
# Run on the approved remote/HPC host, never in this repository.
export LMI_REMOTE_SCRATCH=/absolute/authorized/remote/scratch
test -d "$LMI_REMOTE_SCRATCH"
df -h "$LMI_REMOTE_SCRATCH"
mkdir -p "$LMI_REMOTE_SCRATCH/datasets/laion10m"
cd "$LMI_REMOTE_SCRATCH/datasets/laion10m"

curl -fL -C - -o laion2B-en-clip768v2-n=10M.h5 \
  'https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=10M.h5'
printf '%s  %s\n' c05e4b1d2b2a0c7663ac9767753e25e1 laion2B-en-clip768v2-n=10M.h5 | md5sum -c -

curl -fL -C - -o public-queries-2024-laion2B-en-clip768v2-n=10k.h5 \
  'https://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5'
printf '%s  %s\n' f8f3e61bd22d7d64234a0f587ead9fcf public-queries-2024-laion2B-en-clip768v2-n=10k.h5 | md5sum -c -

curl -fL -C - -o gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5 \
  'https://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'
printf '%s  %s\n' 342794391dafed7bd90dabb740fc15ba gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5 | md5sum -c -
```

Do not ingest any file with a failed checksum. Inspect the remote HDF5 shape and dataset keys first; the previously used 100k file has `/emb`, shape `(100000, 768)`, dtype float16, but the 10M file has not been opened here. The intended importer reads a bounded HDF5 chunk (for example 512 rows), converts only that chunk to f32, sends smaller Qdrant batches (for example 64 points), and releases it before the next read. Assign IDs 1 through N so they align with the official 1-based gold standard. Record both client and server RSS, source bytes/sec, upsert vectors/sec, segment count, and optimizer activity. A bounded importer is available at `tests/lmi_laion10m_stream.py`. It was tested against the existing 100k HDF5 source in dry-run mode and against an isolated mock REST endpoint (125 points, IDs 6 through 130, five API batches). It has not been tested against a live 10M Qdrant collection. Once a collection with an approved LMI configuration is created and a small live subset passes, the remote launch command is:

```bash
python tests/lmi_laion10m_stream.py \
  --source "$LMI_REMOTE_SCRATCH/datasets/laion10m/laion2B-en-clip768v2-n=10M.h5" \
  --url http://127.0.0.1:6333 --collection laion10m_lmi \
  --hdf5-chunk 512 --api-batch 64 --start 0 --end 10000000 \
  | tee "$LMI_REMOTE_SCRATCH/laion10m-ingestion.jsonl"
```

First use `--end 128` and an isolated test collection. On interruption, resume at the last printed `next_zero_based_row`; upserts are idempotent by 1-based point ID. Do not activate the optimizer for the full build until the approved LMI sample size and the existing component-memory admission policy are reconciled. The exact collection-create/PATCH commands therefore remain a reviewed gate, not a ready-to-run claim.

Only after S.3C correctness, sample-scale feasibility, remote capacity, checksums, and streaming ingestion are verified should a first 10M build be attempted. Record sampling, clustering, Torch training/export, each corpus routing/posting pass, persistence, reopen, query breakdown, bucket distribution, and Torch/native tie fallback. Keep `B≈3,162` a stated initial hypothesis rather than a guarantee of feasible runtime or good retrieval quality. The present builder has a conservative component-memory admission ceiling, so a large sample cannot simply be configured without a separately reviewed budget change.
