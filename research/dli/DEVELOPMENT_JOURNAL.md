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

## 2026-11-12 — Literature and optimizer investigation

Extracted all pages of the three local PDFs using a project-specific pure Python reader. See [literature notes](LITERATURE_AND_REFERENCE.md) for source hashes, page references, scope and an observed potential reference indexing discrepancy. The initial package install via `uv pip` failed because DNS/network access was unavailable; GitHub connector retrieval of pure-Python source provided a workaround. No diagrams have been visually inspected, so the literature gate remains incomplete. Reference source was read selectively; it declares MIT in its crate manifest while its root LICENSE contains LGPL-3.0, requiring clarification before direct reuse.

[Audit](QDRANT_ARCHITECTURE_AUDIT.md), [comparison](ARCHITECTURE_COMPARISON.md) and [ADR](ADR_001_DLI_HIERARCHY_PLACEMENT.md) record source-derived findings. A registry update detached from optimizer publication is unsafe; whether an integrated registry can be made crash-consistent is unresolved. Search needs a generation-bound routing/selection/scoring interface for a true global bucket budget. One local shard is the initial scope.

A clean-target native test build was started offline. The first 35-second bounded compile timed out (exit 124), without running tests. A longer bounded continuation uses the same isolated target. A small holder replacement control was added in test code; it does not establish concurrent update or full DLI correctness. Results to follow.

A mock-only deterministic global budget control passed (1 test): a budget of five selects whole buckets containing seven points, and selects a different set from fixed per-level probing. See [test matrix](TEST_MATRIX.md) for command, hashes and limitations. It does not establish native integration or correctness.

The isolated offline test build finished in 1m42s. The new native Qdrant holder control passed (1/1): replacing a segment changed its holder ID, while a previously cloned segment handle remained readable. This uses equivalent native fixtures, not an LMI level or a concurrently running query. Three existing `finish_optimization` controls also passed, including propagation and a post-swap failure control. They exercise native optimizer internals, not a DLI manifest, arbitrary crash boundaries, or concurrent cross-level search. Exact commands, binary hash and logs are in [test matrix](TEST_MATRIX.md).

Current gates: G0 PASS; G1 and G2 CONDITIONAL PASS only for the scoped textual/source investigation (diagrams and end-to-end paths remain unread); G3 CONDITIONAL PASS as a preliminary written comparison, with B feasibility unresolved; G4 NOT RUN for the required two native prototypes (one holder control and one mock policy control are insufficient); G5–G10 NOT RUN; G11 CONDITIONAL PASS because the journal exists but mirror and required reports remain incomplete. Next: implement a test-only generation/holder topology with actual LMI segments, verify publication and mutation behavior, and inspect figures and remaining reference source before final architecture acceptance.
