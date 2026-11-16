# Search optimization checkpoint

No full ingestion, new model training or size-tiered benchmark was launched. Original geometric and size-tiered artifacts remain the controls. The storage fix affects future generations, not retained vector files.

Verified improvements so far: deterministic destination-placement tests, more precise native phase accounting, offline geometric routing reconstruction and an explicit reserved-level experimental policy. At roughly29K selected offsets, reserved/density policies provide .643/.633 top10 bucket coverage; baseline provides .567 at32K. This is an offline inclusion advantage, not a completed native recall/latency win.

The geometric baseline touches the6M Cold member on every measured query. Its model-scope10 ceiling is.600 and bucket coverage.407, close to measured final recall.402. Query latency cannot be explained by neural inference alone from these old broad counters. Detailed scorer/materialization results and warm repeats are needed; storage causation remains incompletely isolated.

The optional per-segment top-k materialization reduces objects and ID lookups without reducing selected-vector scoring work. It remains default-off. No optimized before/after speedup is claimed from debug fixtures or offline reconstruction.

See HYPOTHESES.md for supported/rejected ideas. Continue geometric DLI as an experimental branch with storage and visibility protections intact; do not rerun10M ingestion until native frozen-topology measurements validate an execution/routing improvement. Temperature fitting, larger samples, teacher diagnostics, matched-recall optimized curves, I/O/page-fault controls and statistically replicated conclusions remain unverified.

Bounded native execution now PASS:12 learned searches,0 fallback, identical58,610 scored vectors per3-query round, unchanged Recall10/30. Subsequent full→top-k query total1.2603→.4049s in debug; scorer.3155→.2930s, materialization.1518→.00114s, resolver.6524→.00214s. This is measured mechanism evidence, not optimized campaign performance. The first full pass spends23.1649s in scoring versus.3155s on the repeated full pass: cold/order sensitivity dwarfs model inference in that control. See04 for limitations.

Native reserved/top-k bounded run PASS1/1 (161.71s including reopening): official query rows0–11, two identical rounds,24 learned/0 fallback,362,047 successful vector evaluations per round (30,171/query), Recall10=.633333/Recall30=.572222 in both. First/repeated p50=7331.69/213.38ms in debug. This validates execution of the reserved policy on the retained four-model topology; it is not the100-query offline estimate or release-mode latency. Raw continuation26/frozen-reserved-debug/{rounds.json}, frozen-reserved-debug.log, reserved-native-deltas.json.

CPU/fault control now supplies direct outlier evidence: on query rows20–22, initial native access18.924s wall/2.44s CPU and75,702 major faults, repeated full access.306s wall/.30s CPU andzero major faults. All runs score61,565 vectors, regardless of whether90 or61,565 offsets are materialized. This strongly distinguishes physical-access behavior from multi-router inference. It does not establish release-mode steady-state latency or a universal Cold/Cached ratio. Optional detailed traces and raw deltas are described in04.
