# Native quantized scoring checkpoint

Existing default-off learned-posting integration at caf5e2b46 is retained: native scalar shortlist, original rerank, complete member visibility resolution and exact fallback. Existing trained two-router/Plain tests passed25 tests,6 ignored; this does not validate quantized concurrent publication or persisted missing/stale artifacts.

This checkpoint improves the native65K comparison's accounting: S0 original scorer gets its own timer; S1 approximate top30 gets separate recall; S2 execution timer is captured **before** independent exact-score assertions. The previous wall_ms included those assertion reads and must not be presented as uncontaminated query latency. Quantized duration includes native ranking and candidate preparation; approximate and original counts are requested/scorer-input counts in this deletion-free65K fixture, not an instrumentation claim for arbitrary production calls.

A release run was launched with existing PyTorch2.5.1CPU environment/target, two build jobs, features testing,lmi-training,lmi-research. It encountered a substantial feature-specific dependency rebuild. It is a bounded4-query65K native segment experiment, not10M ingestion or neural training. Evidence: continuation32/release-quantized.log. Until completion and oracle assertions, **optimized results are pending**. No peakRSS, cold faults or original bytes-read claim is available. No policy is promoted from this unfinished measurement.

The retained10M active segments still have quantization disabled. No backfill was attempted. Host-independent process persistence is not verified; managed session76624 is not a user service. Previous user-systemd access was denied. No process is killed to save monitoring effort.
