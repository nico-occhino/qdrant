# Adaptive partition results — continuation35

Original serial native factorial remains running:128 and256 completed, fresh548 not complete at the single session check. No duplicate training launched. Results below compare completed128/256 against the **earlier continuation28 548 control**, not the unfinished replicate. Provisional symlinks and source identities are explicit under continuation35/provisional. Same65K pool,32768 samples,512hidden,30epochs,5teacher kmeans iterations, seed42,32 diagnostic queries800–831. These queries are reused diagnostics, not untouched final tests.

Coarse budget steps overstated work needed for95% coverage. A focused correction adds --budget-step250 and per-query rows to the existing offline analyzer, re-executed into **new** continuation35/refined directories without changing historical outputs or training again. Candidate work below is offline selected posting entries; it is not an instrumented native scorer count. Coverage is against an exact local Float64 normalized pool oracle.

|Outputs|Heldout assignment|Model bytes|Model work at ≥.90 coverage10&30|Model work at ≥.95|Teacher work at ≥.95|
|---:|---:|---:|---:|---:|---:|
|128|.88013|1,837,652|5283|9825|8598|
|256|.84744|2,100,308|3429|7450|6973|
|548 earlier|.80310|2,699,492|3354|5827|4344|

Occupancy min/max/CV:128=78/1860/.543;256=9/1015/.578;548=1/648/.741; all zero empty model buckets. Fewer outputs improve classifier agreement but worsen high-quality work here. At95% target the earlier548 model selects21.8% fewer entries than256, with28.5% more model bytes.128 is not the quality/work winner. At90%,256 and548 are close in work, so256 remains a credible smaller-model alternative. No statistical confidence claim from one seed/32 diagnostic queries.

Native fixture elapsed108.51s (128),188.39s (256),46.18s (older548). These are complete private-build test durations in different execution contexts, not comparable isolated teacher/MLP training times. Concurrent analyses/builds and source revisions constrain any cost interpretation. Router latency, native search latency and native full-pool recall are unmeasured for these new128/256 models.

**Decision:** reject128 as a scaling candidate on current evidence. Retain548 as current baseline and256 as a memory ablation; do not change physical hierarchy output policy or launch1M/10M. Final adoption awaits fresh548 replicate and a validation/test-separated native comparison. The geometry of larger buckets contributes even when the classifier agrees better; increasing classifier accuracy alone is insufficient.

Raw curves/per-query CSV and coverage-work.svg: continuation35/refined. New tool summarize_bucket_control.py retains actual metadata/configs and refuses missing cases. Do not silently substitute an old548 result into the completed continuation34 factorial.
