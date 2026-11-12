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

## Continuation — preflight

Resumed on clean branch `research/dli-cross-segment-20261112` at `960a35f37ab1cba3b49ed7404f9e4d22066f2dee`; verified the original `b5c195bd5248589f3c01b940fb8365b14256f838` is an ancestor. Rust 1.98.0, approximately 590 GiB disk free and 22 GiB available memory were observed. Existing isolated Cargo target and extracted texts are present; no cargo/rustc jobs were observed. `/mnt/c` remains `ro` in this command environment. New outputs go under `continuation-01`; existing evidence is preserved. Gate labels in previous entries refer to the earlier gate definitions; this continuation uses the revised definitions in the current task.

### Native topology build and normalization investigation

A test-only topology now builds two independent trained LMI segments with native `SegmentBuilder`, plus an appendable Plain segment. Its exhaustive per-level search uses nprobe=2 on two buckets and compares externally mapped point IDs and cosine scores to an independent nine-vector oracle, then reloads the learned segments. It is an independent-search baseline, not a DLI registry or global search. The test is pending compilation and execution.

First build failed: `torch-sys 0.18.1` requires PyTorch 2.5.1; the initially selected environment supplied 2.7.1. The diagnostic was the build script's explicit version error in `continuation-01/native-topology.log`. Found an existing `/home/nicoo/miniconda3/envs/lmi-rust-inspect` with CPU PyTorch 2.5.1; retrying without bypassing version checks in a separate log. No root cause beyond the environment mismatch is inferred.

The mock numerical check (two tests PASS, `continuation-01/budget-control.log`) shows a=3, level-1 probability 0.4 gives 1.2 with paper's one-based exponent and 0.4 with the reference's zero-based exponent, relative to buffer score 1. Both formulas multiply every learned level by the same constant factor, so learned-to-learned ordering remains the same, but buffer ordering changes. This verifies a mathematical difference, not which convention is intended or superior.

The 2.5.1 retry reached the new test but compilation failed with E0599: `external_id` requires `IdTrackerRead` in scope; the test imported a different trait. Corrected the import and reran the same bounded command; diagnostic evidence is `continuation-01/native-topology-retry.log`. No runtime behavior was tested by that failed build.

Native segment topology: after fixing the ID tracker trait import, the first run passed using independently trained two-bucket LMI indexes (seeds 17 and 29) and a Plain segment. All nine external IDs and cosine scores matched an independent oracle with both LMI paths attributed as learned. Each LMI segment reopened and returned the same results. Log: `continuation-01/native-topology-trait-fix.log`.

A subsequent controlled manual mutation extension initially failed to compile: `flush` and `delete_point` require distinct storage and non-appendable traits; fixes are documented by `native-topology-mutations.log` and `native-topology-mutations-retry.log`. After importing both, `native-topology-mutations-fixed.log` passed (1/1): explicit deletion of two old LMI copies and insertion of replacement IDs into Plain produced one result per ID with the expected scores, and the LMI segments reopened after flush. This is not shard-level automatic upsert, a registry, or crash recovery.

An additional test-only global selection control has been added and is awaiting execution: it uses actual model logits and postings, then `score_candidates` over selected offsets. Its experimental two-bucket-per-level topology and explicit normalization are not a faithful full geometric DLI hierarchy; it does not modify the production search path.

The final native global control passed (`continuation-01/native-global-final.log`, 1 test). For a requested candidate budget of 2, the chosen learned bucket contained 3 candidates; native scoring matched the independent fixture scores for each. This is genuine use of native routers, postings and candidate scoring, but remains test-only and does not prove global top-k, calibrated cross-level routing, a geometric level layout, filters, update versions or generation consistency. [Test matrix](TEST_MATRIX.md) records hashes, command and exact limits.

Revised gates for this continuation: G0 PASS; G1 and G2 CONDITIONAL PASS (figure inspection and complete call/recovery trace remain incomplete); G3 CONDITIONAL PASS (A lacks comparable native evidence); G4 CONDITIONAL PASS (actual segments and buffer built but no holder-level DLI registry); G5 NOT RUN; G6 CONDITIONAL PASS (test-only candidate scoring control, not a native coordinator); G7 CONDITIONAL PASS (manual segment operations only); G8–G10 NOT RUN; G11 CONDITIONAL PASS (mirror unavailable and handoff incomplete). No claim of a working temporal DLI is made.

## 2026-11-12 — Holder/view continuation

Verified clean starting HEAD `67e63b615a802f74b07b153598fb8a1efe6a80b9`. Existing fixture and evidence were inspected; approximately 586 GiB disk and 22 GiB memory available. Reusing the existing target. New logs: `continuation-02`.

Identity and synchronization investigation is recorded in [generation specification](DLI_GENERATION_AND_PUBLICATION_SPEC.md). The experimental adapter serializes capture and controlled holder replacement, pins native handles and fails closed after bypassing replacement. It rejects proxy members and does not join production optimizer publication. View generation is membership consistency, not point-in-time data visibility. New native holder integration tests are compiling; no result is claimed yet.

First holder test build failed with E0603 because the test imported the private `update::points` module. Source inspection showed public re-exports in `shard::update`; the import was corrected to that supported seam. The failed log is `continuation-02/test.log`, retry output is `continuation-02/retry.log`. This was a test import error, not a native contract failure.

The retry executed 2 tests: native update visibility passed, while the barrier query thread failed on missing ambient hardware-metrics scope. Qdrant's ambient scope is thread-local; adding an explicit unmeasured scope inside the spawned thread resolved it. Logs `retry.log` and `scope-fix.log` preserve diagnosis and confirmation. Final run: 2/2 PASS, 0.04s test execution (debug; no performance interpretation).

Native evidence: captured G routed before controlled holder replacement and scored afterward through old pinned Arcs; fresh capture sees G+1 and identical all-candidate oracle results. Stale view and physical tokens fail; a bypassing holder swap makes capture fail closed. Budget 2 returned 3 candidates, IDs 24/23/22, from actual LMI postings through native scoring. Separate native `upsert_points` and `delete_points` calls under the updates mutex demonstrate completed insert/upsert/delete visibility in previously captured handles. These are holder/update-helper tests without LocalShard, WAL, optimizer execution, pending updates during build or durable registry.

V0 PASS; V1 PASS for documented experimental contract; V2 PASS for test-only native registry; V3 CONDITIONAL PASS for cooperating controller transitions only; V4 CONDITIONAL PASS for direct native swap, not optimizer proxy lifecycle; V5 CONDITIONAL PASS for trusted test-generated route tokens and fixed query; V6 CONDITIONAL PASS for completed native helper operations, not pending replacement; V7 CONDITIONAL PASS for barrier search/replacement only; V8 NOT RUN as an established durable protocol; V9 checkpoint documentation/evidence being completed. Production expansion and Bentley–Saxe remain gated. Next task: integrate a test adapter with native proxy/delta publication, or establish a safe source-version rejection boundary, before testing writes during replacement.

## 2026-11-12 — End-to-end implementation attempt, M1

Verified clean HEAD `306635af179d36d330f70562079c8e13a86a6e84`; available memory approximately 22 GiB and disk 583 GiB. Existing target reused. New evidence directory `continuation-03`. Authorization now extends through production implementation subject to correctness gates; earlier scope limits in this journal are historical.

The real `execute_optimization` installs finalized proxies before calling `OptimizationStrategy::create_segment_builder`. A test strategy pauses at that boundary using bounded channels, then native update helpers upsert a source ID, delete another and insert a fresh ID while the source is proxied. Publication and cancellation both use native implementation. The registry intentionally fails closed after native replacement, then is rebound only for oracle verification; that rebind is not an atomic or durable registry protocol. Tests are pending execution.

M1 investigation encountered two test-only construction errors: `ProgressTracker` uses `new_for_test` and telemetry aggregation uses `OperationDurationsAggregator::new`; then a dummy CPU permit failed a native merge debug assertion. Following `SegmentOptimizer::optimize_for_test`, the worker now uses a real `ResourceBudget::new(1,1)` with an acquired IO permit. Failed logs `proxy.log` and `proxy-retry.log` are retained. Corrected `permit-fix.log`: all 4 holder integration tests PASS, including new native proxy publication and cancellation cases. Final oracle includes updated ID11, deleted ID12 and new ID99; other levels and buffer remain present. The pause is at builder creation after proxy installation, before copying source data. Late propagation/finalization writes are not tested.

M2 source review found that the native manifest is explicitly a superset-biased listing, not a topology transaction, and LocalShard restart scans directories and deduplicates source/destination point versions before constructing the holder. A hierarchy cannot safely derive exclusive logical membership from that listing alone. The candidate durable approaches and untested failure boundaries are recorded in [recovery matrix](DLI_RECOVERY_MATRIX.md). A source/output coexistence assertion is being executed to document the retained-preimage boundary. No production registry or loader patch is justified yet; M2 remains unestablished and dependent compaction/configuration/evaluation are gated.

Final M1/recovery-boundary run: 4/4 PASS, 0.04s debug runtime. It confirms native retained source and new output directories coexist after swap before safe flush. [Test matrix](TEST_MATRIX.md) records hashes and commands. [Final status](FINAL_IMPLEMENTATION_STATUS.md) classifies the implementation PARTIAL. No production DLI is exposed and no dynamic evaluation is justified. This session stops the affected M2 path because neither candidate durable registry scheme has an established logical commit/recovery contract; dependent implementation remains gated. Source inspection and executable evidence are sufficient to reject a naive manifest/scan patch, not to prove B impossible. Windows mirror remains unavailable under the read-only mount; canonical journal preserved.
