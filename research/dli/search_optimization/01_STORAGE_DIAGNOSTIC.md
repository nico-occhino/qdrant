# Deterministic learned destination storage

Continuation26; baseline db01535f4, executed original10M baseline3393b4a3c. Fix checkpoint5fe7bd9cb. Existing uncommitted profiling work was retained and validated.

The maintenance worker cloned `sources.first().config()` and changed only the index. Plain/COW destinations explicitly use `InRamChunkedMmap`; an indexed Cold source uses `ChunkedMmap`. Consequently source ordering and previous COW ancestry could change the next learned destination's backend. This is accidental inheritance, not a model property.

The corrected worker captures collection vector placement before activation. It resolves `memory` before legacy `on_disk`, matching native optimizer precedence, and passes that policy into both native/private strategy instances. The builder overrides inherited dense storage placement. Explicit Cold stays Cold; default Cached stays Cached. Native appendable COW stays Cached. No existing persisted generation is converted, no vector population is forced into RAM, and no full ingestion was started.

| Frozen geometric member | Role | Vectors | Persisted placement |
|---|---|---:|---|
|44bee22d-72cb-4b49-a175-76b3e54d9dce|L5|6,000,000|Cold|
|c85e5ff2-942a-4557-943e-4a9573f245a4|L4|3,000,000|Cached|
|922d9a41-fcd2-43fa-9e94-9740ede00865|L2|750,000|Cached|
|d6e13095-ce83-4483-96a9-8e185853d397|L1|250,000|Cached|
|8c1ebbae-f349-499e-b3db-f27abfabe928|Plain|120,191|Cached|

All original100 geometric queries select the6M Cold member; therefore these traces alone cannot estimate a within-geometric Cold/Cached causal latency ratio. The earlier size-tiered association is relevant evidence, not that ratio. Cold is a native memory-placement policy, not proof of physical disk reads for each access. Task-local fault/CPU counters now verify a large major-fault access component (report04); exact file/device attribution and controlled Cold/Cached conversion remain unmeasured.

`dli_destination_storage_is_independent_of_source_order_and_reopens` passed12 builds: Cached/Cached, Cold/Cold and mixed sources, both source orders, both destination policies. Each target has four live IDs and reopens with the expected backend. The configured native fixture separately exercises Plain+learned source publication, concurrent native mutations, cancellation, repeated publication and clean recovery. These tiny tests prove policy stability, not10M placement performance. Log: `<admin>/experiments/2026-11-12-dli-architecture/continuation-26/trained-storage-test.log` (19 PASS,3 ignored).
