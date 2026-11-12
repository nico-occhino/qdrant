# Temporal DLI development journal

## 2026-11-12 — Isolation and initial status

Research objective: evaluate temporal hierarchies over Qdrant-owned segments against an encapsulated hierarchy, preserving visibility, durability and model/posting generation consistency. Temporal DLI is distinct from continual classifier adaptation (CLI). Cross-segment placement is the preferred direction subject to feasibility evidence; no production implementation is yet authorized by the gates.

Baseline: clean `port/qdrant-v1.20` at `b5c195bd5248589f3c01b940fb8365b14256f838`; new branch `research/dli-cross-segment-20261112` in `/home/nicoo/worktrees/qdrant-dli-cross-segment`. Profiling commits are present in the baseline history. Historical artifacts and source worktrees are preserved. Experiment directory: `/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture`.

Initial attempts to inspect the workspace failed before execution because the sandbox could not enforce the read-only `.aws` symlink. On retry the sandbox allowed commands; no root cause for the environment change has been established.

G0: PASS for local clean starting provenance and isolation; remote freshness and historical validation are not independently re-executed. G1–G11: NOT RUN. Next: inspect local papers and native Qdrant lifecycle before any implementation.

## 2026-11-12 — Initial reference and source inspection

Local PDFs matching the temporal DLI, Rust DLI and CLI topics are present in `Learning stuff/Learned`; their contents have not yet been read. `pdfinfo` and common Python PDF libraries are unavailable in the default environment. This is a tooling limitation, not evidence that the papers are unavailable.

The [Plhák reference repository](https://github.com/simonplhak/DynamicLearnedIndex) reports master revision `4e756476ae9352fcef7896cf5ef551282751c32b` and LGPL-3.0 metadata. Its README identifies `dynamic_learned_index` as the library, a buffer, arity, level model configuration, Bentley–Saxe compaction and several rebuild strategies; its source-level implementation and license terms still need inspection. No code was copied.

Initial Qdrant findings are in [the preliminary audit](QDRANT_ARCHITECTURE_AUDIT.md): search fans out over holder segments, while `swap_new` assigns a fresh segment ID. This exposes a concrete registry coordination question, not proof of an insurmountable incompatibility. G1 and G2 remain NOT RUN as complete gates; G3–G10 remain NOT RUN.

Mirror synchronization attempted with `cp` to the requested Windows path; it failed with `Read-only file system`. The mount is readable but outside writable roots. No unrelated file was overwritten. The journal here remains authoritative; retry synchronization when the mount is writable.

Next: inspect the actual paper contents and reference source; trace optimizer replacement and update contracts, then build the two bounded proofs. The architecture is not selected as feasible by the evidence so far. No experiments or tests have run.

## 2026-11-12 — Resumed research and documentation checkpoint

`findmnt -T` reports `/mnt/c` as a 9p mount with `ro` in its mount options inside the command environment. The failed mirror write is therefore consistent with a read-only mount as presented to this sandbox; this does not establish how the host mounts it outside the sandbox. No mount or security settings were changed. The existing journal and preliminary audit are being checkpointed before implementation; both explicitly mark incomplete work.
