# Test matrix — initial research

| Evidence | Status | Scope and limitation |
|---|---|---|
| Global budget control (`prototypes/budget_control.rs`) | PASS | Mock-only, one deterministic test; whole-bucket budget 5 selects counts 4+3 and differs from fixed per-level probing. No Qdrant data, router or scoring. |
| Native holder replacement control | PASS | Native Qdrant holder and segment objects: new holder ID and retained old handle, using equivalent fixture contents. Not an optimizer, query, concurrent update or LMI test. |
| Plain buffer plus two LMI levels; oracle, mutations, reopen | NOT RUN | No native topology or coordinator built. |
| Concurrent optimizer replacement, reads and writes | NOT RUN | Native protocol source inspected only. |
| Failure boundaries and recovery | NOT RUN | Existing test source examined, not executed as DLI tests. |

Mock test: `rustc --edition 2024 --test research/dli/prototypes/budget_control.rs -o <experiment>/budget-control-test && <experiment>/budget-control-test`. Source SHA256 `6fe3230a9019fae0137f519559b52e1d5690e2960b5e8f61d6fc39e4fbb0a315`, binary SHA256 `94a6b2b14668b140ce1394f3f9cabfb77a9be2bd8e7e72a75e3977fa8c492854`; output `<experiment>/budget-control.log`. This is a semantic counterexample, not a database performance experiment.

Existing native optimizer controls: `cargo test --offline -p shard --lib optimize::tests::finish_optimization -- --nocapture`: 3 passed, including propagation of deletes and a post-swap failure recovery control. These are existing optimizer tests, not DLI or full crash tests; log `<experiment>/optimizer-controls.log`. Native test binary SHA256 `7b84bbe537e1708c0874387e61083c3e3f0024434455a677e487874ac0136b5b`. Holder control log `<experiment>/dli-holder-control.log`.

## Continuation: native topology and routing control

PASS, segment-level only: `lib/segment/tests/dli_native_topology.rs` constructs two independently trained two-bucket LMI segments (seeds 17 and 29, nprobe 2) and one Plain segment. Full independent probing matches an independently calculated nine-vector cosine oracle with learned-path counters. Explicitly deleting old LMI copies and inserting replacement IDs into Plain yields unique IDs and expected replacement scores. Each LMI segment is flushed and reopened; Plain segment and registry are not reopened. This is not a collection/shard update or optimizer test.

PASS, test-only global selection: real router logits and posting lists are ranked with the illustrative paper-style arity-two multiplier, including a buffer bucket, and selected offsets are scored by `candidate_scoring::score_candidates`. For budget 2 the selected single bucket contained 3 candidates, all mapped to its own external IDs and individually checked against the fixture's exact cosine scores. The test does not verify global top-k against an independent selection oracle, calibrated probabilities, filters, multi-segment visibility, concurrent publication or a geometric 2/4-bucket hierarchy. Output: `<experiment>/continuation-01/native-global-final.log`.

Command (from worktree): `CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=<experiment>/cargo-target LIBTORCH_USE_PYTORCH=1 PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH cargo test --locked --offline -p segment --features 'testing,lmi-training' --test dli_native_topology -- --nocapture`. Source SHA256 `242e6230ac9264740513a33c8d821a59119ce1227deadaddc52b40d2f289588a`; debug test binary SHA256 `ced25ebecd51709a1ef57070f3091f6859b943dc07493427882ad537bb59cffa`. PyTorch 2.5.1 CPU, Rust 1.98.0. Target directory total was 8.5 GiB including previous builds; no peak RSS was measured. The initial 2.7.1 environment failed the required version check; the logs and later trait-import diagnostics are preserved.

## Holder/view continuation (continuation-02)

`lib/shard/tests/dli_view.rs`: 2 tests PASS. Fixed nine-vector 2D cosine fixture, trained two-bucket LMI seeds 17/29, replacement seeds 43/53, one Plain buffer. Registry captures original native Arcs under registry/holder locks and checks UUID, dimension, distance and loaded model/postings. Barrier read/replacement retains old G handles, fresh capture observes G+1; invalid view/UUID routes and bypassed holder replacement fail closed. All-candidate results match fixture cosine scores. Budget 2 returned 3 points (24/23/22). Native helper upserts ID11 to [0,1], inserts ID99, deletes ID21; old view queried after completion sees the expected single replacement and deletion.

Scope: native holder objects and update helpers, test-only registry; no WAL, LocalShard, optimizer publication, pending-upsert replacement, persistence or crash test. Proxy members rejected. Replacement assumes no writes during its private build. Route inputs are trusted internal test data; arbitrary forged bucket/offset membership is not validated.

Reproduce with existing target and PyTorch 2.5.1 environment: `CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=<experiment>/cargo-target LIBTORCH_USE_PYTORCH=1 PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib cargo test --locked --offline -p shard --features 'testing,segment/testing,segment/lmi-training' --test dli_view -- --nocapture`. Final log: `<experiment>/continuation-02/scope-fix.log`; failed import and thread-scope logs preserved. Source SHA256 `74c2ad06f8da216c136131b36453335635639b85bd8704a436a1aba8d989c792`; binary SHA256 `dd0e1897e5fca3550a6cd9c0a9b4104d769871f171d4a58a1c7a5ce81a988544`. Built from parent 67e63b615 plus new test source. Debug results only; no peak RSS measured.
