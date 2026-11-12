# Recovery gate — source and executable evidence

Status: no durable DLI registry implemented. This is a gate assessment, not crash-consistency certification.

Native `execute_optimization` installs proxies, builds with stable source objects, propagates deltas, excludes updates for final propagation, flushes replacement and writes readiness version, swaps entries, registers source drop/ack pins and syncs the native manifest. Cancellation unwraps proxies after propagating changes back. The new tests exercise native publication and cancellation with writes while the source is proxied; they do not inject errors in all phases.

Critical source fact: `lib/shard/src/segment_manifest.rs::SegmentsManifest` explicitly defines a superset-biased listing which can include incomplete and obsolete segments. `LocalShard` startup in `lib/collection/src/shards/local_shard/mod.rs` scans physical directories, recovers pending proxy changes, deduplicates point versions and only then builds the holder/manifest. Thus the existing manifest is not an authoritative hierarchy commit record. Attaching a logical level to every discovered segment does not define which temporal generation wins when source and destination coexist.

| Boundary | Native mechanism observed | DLI evidence |
|---|---|---|
| Proxies installed, private builder paused | Native helper writes through COW/proxy semantics | Publication/cancellation tests pass |
| Private build cancellation | Native unwrap propagates pending changes | Mutation oracle passes; no output directory |
| Successful publication before safe flush | Old preimages retained with deferred cleanup and WAL ack pins | Native source/output coexistence control; see test log |
| Delta propagation failure | Native error path retains/unwraps according to propagation success | DLI-specific injection NOT RUN |
| Metadata commit between readiness and swap | No DLI transaction exists | NOT RUN |
| Restart before old-source cleanup | Loader may recover/deduplicate multiple physical generations | DLI logical precedence undefined |
| Durable hierarchy reopen | No durable DLI metadata | NOT RUN |
| Abrupt termination or power loss | Native assumptions require dedicated testing | NOT RUN |

Coupled-publication option requires durable hierarchy commit semantics plus a recovery rule compatible with native pending-change recovery, preimage retention and WAL replay. Merely extending the current manifest is insufficient. Derived descriptors need lineage, source/destination membership, committed generation precedence, overlap validation and a way to distinguish incomplete output from committed output while retaining source preimages for replay. Physical readiness is not itself a hierarchy commit. Both options require production loader/publication changes beyond the current registry adapter; no safe bounded metadata patch has been established.

Engineering decision: stop unsafe persistent-registry expansion at M2, retaining M1 tests. This is not evidence that B is impossible, and it is not permission to discard native preimages or turn the superset manifest into an exclusive source of truth. The next design must explicitly separate logical search membership from physical recovery dependencies, with error injection on both sides of the commit boundary, before compaction can proceed.
