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
