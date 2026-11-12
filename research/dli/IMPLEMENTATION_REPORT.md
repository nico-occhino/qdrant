# Implementation report — research controls only

Starting source 960a35f37ab1cba3b49ed7404f9e4d22066f2dee. No production code or public APIs are changed. `lib/segment/tests/dli_native_topology.rs` builds two independently trained LMI segments and a Plain segment, searches each with all learned buckets, maps offsets through the owning tracker, checks a small exact cosine oracle and reloads both LMI segments. The native test passed; see journal and test matrix for conditions and limitations. The test is segment-level and does not implement a registry, optimizer publication, concurrent updates or temporal compaction.

The separate `prototypes/budget_control.rs` is a mock policy and numerical-convention control. It is not native scoring or database evidence.

A test-only global bucket planner ranks actual LMI logits and scores selected native postings; the observed budget 2 selected 3 candidates in one bucket. It is research control code, not a native shard coordinator. There is no durable or concurrent DLI generation.

Holder continuation: `lib/shard/tests/dli_view.rs` adds test-only registry/controller, pinned view, generation-bound route tokens, controlled replacement and native update-helper tests. Two tests passed with deterministic barriers and actual trained LMI/native scoring. No production module/API changed. See generation specification for strict no-writes-during-build replacement precondition and unsupported proxy/publication/recovery states.
