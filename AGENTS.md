# Qdrant research

Claude Code owns implementation. Codex may independently review committed changes.
When working in Codex, consult the upstream engineering conventions in CLAUDE.md before editing Rust, APIs or build settings.

## Research workflow

- Record the worktree, branch, HEAD, dirty paths, objective and observable acceptance criteria. One writer per worktree. Review a pinned commit in another checkout.
- Find the prior experiment and native call path first. State the invariant, ownership and failure cases before choosing the smallest maintainable change. Explain consequential trade-offs.
- Reproduce confirmed defects with deterministic regression fixtures. Exercise the normal entry point and failure/retry path. Report separately what compiled and what ran.
- Preserve immutable inputs, raw logs, failed runs and negative results. Report verified observations, hypotheses and open risks separately. Attribute every measurement, reference and test outcome to evidence.
- Search narrowly, keep full logs on disk, summarize relevant excerpts. Reuse artifacts only after matching source, inputs, flags and environment. Poll at meaningful intervals.
- Commit verified work in logical units. Reserve independent review and expensive reasoning for material risk. Review the actual diff. Resolve confirmed defects and update the existing experiment record before a fresh session.
- Keep credentials, local machine details, personal material and private research outputs outside tracked public files. Job submission and cancellation each require explicit task authorization.

## Engine checks

Trace search and mutation through collection, shard, optimizer/proxy segment and segment index dispatch. Establish which generation owns the point version during compaction, concurrent updates, deletion and reopening. Verify publication order, model reuse eligibility, WAL replay and fallback counters on the requested path.
Use focused Cargo tests for the affected crate and enable the features that compile the relevant instrumentation. List the selected tests first and verify a nonzero executed count. Use `cargo nextest run -p segment` with a relevant filter for segment work and the upstream `--profile perf` for benchmarks.
Match exact/plain, HNSW and IVF-style controls by corpus, precision, metric, filters, snapshot and timing boundary. Keep dimensions, budgets and selected configurations in experiment manifests.
