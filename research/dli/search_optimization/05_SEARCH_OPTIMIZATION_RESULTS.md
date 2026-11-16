# Search optimization checkpoint

No full ingestion, new model training or size-tiered benchmark was launched. Original geometric and size-tiered artifacts remain the controls. The storage fix affects future generations, not retained vector files.

Verified improvements so far: deterministic destination-placement tests, more precise native phase accounting, offline geometric routing reconstruction and an explicit reserved-level experimental policy. At roughly29K selected offsets, reserved/density policies provide .643/.633 top10 bucket coverage; baseline provides .567 at32K. This is an offline inclusion advantage, not a completed native recall/latency win.

The geometric baseline touches the6M Cold member on every measured query. Its model-scope10 ceiling is.600 and bucket coverage.407, close to measured final recall.402. Query latency cannot be explained by neural inference alone from these old broad counters. Detailed scorer/materialization results and warm repeats are needed; storage causation remains incompletely isolated.

The optional per-segment top-k materialization reduces objects and ID lookups without reducing selected-vector scoring work. It remains default-off. No optimized before/after speedup is claimed from debug fixtures or offline reconstruction.

See HYPOTHESES.md for supported/rejected ideas. Continue geometric DLI as an experimental branch with storage and visibility protections intact; do not rerun10M ingestion until native frozen-topology measurements validate an execution/routing improvement. Temperature fitting, larger samples, teacher diagnostics, matched-recall optimized curves, I/O/page-fault controls and statistically replicated conclusions remain unverified.
