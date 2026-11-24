---
name: ann-benchmark
description: Design or audit an ANN comparison, matched operating point, or benchmark claim.
---

Find the existing report, manifest and raw rows first. Record source/build identity, dataset and query hashes, embedding and normalization, metric, precision, mutation order, snapshot and tie policy. Validate exact top-k against an independent oracle on stored coordinates and quantify any precision ceiling against source gold.
Keep training, tuning and held-out queries independent. Select settings on validation, lock them, then evaluate once on test. Report recall at fixed work and work/latency at the target recall. Count actual visited postings, distinct candidates, scored vectors and exact fallbacks wherever available. Mark missing counters unknown.
Prefer observed configurations. For descriptive interpolation give the two bracketing measured points and axes, compute the linear fraction (target minus lower recall divided by the recall interval), and label the estimate. Reject flat or unbracketed intervals; handle nonmonotone points explicitly. Never represent interpolation as a deployable setting.
Pair identical queries/snapshots. Compute paired effects and resample pairs for uncertainty; replicate index builds, seeds or runs for build-level variability. Preserve cold/warm regimes, concurrency, hardware, run order, thread count and latency quantiles. Separate index bytes, resident process memory and scheduler accounting.
Keep profiled runs separate from timed runs. Distinguish in-process segment search, server RPC and SQL executor timing; verify the actual search route or SQL plan. Publish immutable manifest, raw rows, checksums, analysis command and exclusions. Report failed gates, sensitivity and negative results.
