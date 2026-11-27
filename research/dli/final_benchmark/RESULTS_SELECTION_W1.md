# Selection pilot, stage 1 (W1), result

Status: **COMPLETE.** The choices are recorded in amendment A3 (`PREREGISTRATION.md` §13). Plan seed 20261202,
validation panel `val` (64 queries), and G1 gold. No test panel was read.

## Jobs
| Job | Arm | Outcome |
|---|---|---|
| 26011817 | plan W1 | complete |
| 26011819 | DLI-best | **failed**, run phase: EBADF writing `batches.jsonl` after batch 16 (A1.1); no data used |
| 26027409 | DLI-best (rerun) | complete, all gates pass, 46 min |
| 26021472 | E1 | complete, all gates pass |
| 26021473–75 | S-LMI with 250, 1,000 and 3,162 buckets | complete, all gates pass |
| 26021477–79 | IVF with 250, 1,000 and 3,162 buckets | complete, all gates pass |
| 26021480–81 | HNSW with m = 16 and 32 | complete, all gates pass (HNSW `upper_bound`, see A2) |
| 26027410 | S-LMI with 10,000 buckets | gate `monotone_work` failed in the job; passes under A3.3 (one widening restart: batch 0, nprobe 1→2); 10.2 h |
| 26027411 | IVF with 10,000 buckets | as above (two restarts: batches 0 and 1, nprobe 1→2); 6.6 h |

## Choice (mean validation W@0.90 over batches 1–20; p50 latency at the first grid point ≥ 0.90)
| Arm | Candidate | Mean W | Mean p50 (µs) |
|---|---|---|---|
| S-LMI | 250 / 1,000 / 3,162 / **10,000** | 15,357 / 9,051 / 6,333 / **3,307** | 5,373 / 3,046 / 2,072 / **1,275** |
| IVF | 250 / 1,000 / 3,162 / **10,000** | 14,501 / 6,634 / 3,857 / **1,686** | 5,758 / 3,127 / 1,807 / **1,424** |
| HNSW | **16** / 32 | **4,698** / 5,714 (upper bound) | **521** / 784 |

## Notes
- **Widening restart (A3.2).** Of 34,944 (n, 2n) query pairs in the 10,000-bucket runs, 486 fall in work. All 486
  fall by 1–29 vectors, at nprobe 1, 2 and 4. W@0.90 lies between nprobe 8 and 128.
- **Edge of the grid.** Both static arms choose 10,000, the largest candidate. Per A1.2 the grid is not extended.
- **Integrity.**
  - The two jobs that failed a gate never wrote `SHA256SUMS`. Their local copies were compared file by file with
    storage (87 files each, identical).
  - Archives on C: are `Research-Results/2026-11-26-final-benchmark-stage1` (884 files) and
    `…-a1` (283 files), each with a verified `STORAGE_SHA256.txt`.
- **Analysis.** `analyze_final.py --manifest` in pilot mode, at this revision. The derived JSON is `derived/selection_W1.json`.
