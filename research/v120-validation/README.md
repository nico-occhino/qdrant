# Qdrant v1.20 LMI scientific-validation checkpoint

Date: 2026-11-11. This checkpoint is **not** a v1.20 300K or 10.12M benchmark result. It records verified inputs, source contracts, local preflight, and unresolved gates. The machine-readable companion is [`validation-contract.json`](validation-contract.json).

## Source and numerical contract

The clean starting point was port `3790d4308e96a22684e0441038df02316b9ac75f`, based on official Qdrant v1.20.0 `222b2e44ab1c3f17e8246396ead6558e45041df7`. Historical committed LMI source is `9b12d583b1bf9df802b2cc420244bd3c24732d86`; the accepted 10.12M campaign also used a frozen overlay. The v1.20 `build.rs`, `build_plan.rs`, `config.rs`, `postings.rs`, `routing.rs`, `spherical_kmeans.rs`, `state.rs`, and `training.rs` matched that committed historical source byte-for-byte at preflight. `mod.rs`, `read_only.rs`, and `candidate_scoring.rs` had only v1.20 hardware-counter API adaptations before this checkpoint. This continuity does not prove equal storage, search output or latency.

For the historical 300K reference, the three original HDF5 files exist and their MD5s match the dated handoff. The corpus has 300,000 768-dimensional Float16/Cosine vectors; official external IDs begin at 1. The historical LMI uses 548 buckets, sample size 32,768, hidden dimension 512, 30 epochs, batch size 256, routing batch size 256, five k-means iterations, `nprobe=4`, and seed 42. The panel has 20 warmups and 9,980 measured queries. The original files and result path/hashes are in the JSON manifest. No accepted corpus or state was altered.

**Critical provenance distinction:** this historical 300K collection has `lmi_state.json` version **2**; both current committed v1.19 mainline and v1.20 port require version **3**. Its successful historical query result is a reference, not a same-source comparator. Before a frozen-router cross-version experiment, a disposable v2→v3 mapping must be explicitly designed and checked against the original version-2 source and binary representation, or a new version-3 build must be compared as a separate build-equivalence experiment. Directly relabelling the old metadata would not establish correctness.

Official recall remains overlap with unchanged SISAP23 ground truth. Native exactness requires an independent stored-vector oracle. Cross-version posting/candidate comparisons use stable external IDs; segment-local offsets may legitimately differ. Exact ties need a documented boundary policy. The query-score error tolerance cannot be chosen before observing actual Float32 scoring differences.

## OpenAPI gate — blocked by local generator dependency

`VectorParams.lmi_config` is present in Rust and `cargo run --locked --package qdrant --features service_debug --bin schema_generator` succeeded. The output contained `LmiConfig` and `lmi_config` and had SHA-256 `300c8299007c606687bd2cd4d6fa62b21a91fd4d75a4dbd3f4376dbc8b78f721`. The checked-in `docs/redoc/master/openapi.json` contains no LMI entry.

The required `tools/generate_openapi_models.sh` invokes `ytt`, Dockerized `schema2openapi`, `yq`, and Redocly. In this WSL distro, `ytt` and `yq` are absent and `docker info` reports that Docker Desktop WSL integration is disabled. Thus the repository's complete generator and `tests/openapi_consistency_check.sh` could not be run. No generated schema text was hand-edited or committed. The complete generator may expose additional upstream differences; the discrepancy is **not** certified as LMI-only.

## Learned-path observability — bounded internal improvement

The existing test-feature counters distinguish successful LMI routing from successful Plain fallback inside an LMI-configured segment. This checkpoint adds a test-feature-only counter for selected posting offsets after successful learned scoring. It is explicitly **pre-visibility**: a selected deleted/deferred offset may not be scored, and it is not candidate bytes or latency. The existing mixed-segment collection regression now asserts this counter advances while fallback stays zero. The shell regression asserts exact, filtered and non-default-parameter requests each use Plain fallback and select zero learned candidates. `hnsw_ef` does not control LMI `nprobe`; public search parameters other than defaults invoke fallback. The frozen config sets `nprobe` at index construction/opening. Public REST still lacks per-request path attribution and production candidate/scored counts.

No router, posting, scoring or result semantics were changed. The counter is not present without the `testing` feature. Timing instrumentation was not added because its boundaries and overhead have not yet been validated against HNSW/Plain.

## Executed commands and results

- `md5sum` of the three original 300K HDF5 files: all matched the historical handoff.
- `sha256sum` of historical 300K config, result file, state, router and postings: recorded in `validation-contract.json`.
- `cargo run --locked --package qdrant --features service_debug --bin schema_generator > /tmp/qdrant-v120-lmi-definitions.json`: **pass**, Rust schema contains LMI. Cached target `/home/nicoo/work/qdrant-v120-port-target`; two Cargo jobs; existing LibTorch environment.
- `cargo test --locked -p collection --features lmi-training --lib tests::lmi_lifecycle::native_segment_delta::native_lmi_mixed_segments_mutation_restart_and_optimizer -- --exact --nocapture`: **1 passed**, route/fallback/selected-offset observations through mutation, restart and optimizer replacement.
- `cargo test --locked -p segment --features lmi-training --test lmi_shell_v120 exact_filter_and_nondefault_params_use_plain -- --exact`: **1 passed**, three Plain fallbacks and zero selected learned candidates.
- `cargo check --locked -p segment -p collection`: **pass** without the training feature.
- `rustfmt +nightly --edition 2024 --check` on the three edited Rust files; `git diff --check`; `python3 -m json.tool` on manifest: **pass**. Whole-workspace rustfmt is not claimed; two untouched upstream files were known to fail it before this checkpoint.

## Gate decision and next bounded work

**BLOCKED for the specified 300K cross-version differential.** The verified HDF5 inputs are available, but the historical trained state is version 2 and source-equivalent version-3 model/postings have not been established. No same-input two-version per-query output, mismatch table, score deviation distribution or native-oracle recall was produced; none is inferred from the historical reference. OpenAPI consistency is separately blocked by the missing Docker toolchain. Internal learned/fallback observability improved, but public-service attribution and actual scored-candidate counts remain open.

The narrow next steps are: (1) make the repository's generator runnable in an isolated environment and commit only its verified output; (2) identify the exact version-2 serializer/source, then test a disposable state conversion or construct one version-3 state under a fixed protocol; (3) run the small synthetic paired comparison first; (4) run the 300K paired comparison with fixed queries, stable-ID postings/candidates, native scorer and independent exact oracle. Do not launch the 10.12M campaign or reinterpret the accepted 3 November results as v1.20 data.
