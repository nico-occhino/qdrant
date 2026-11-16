# Amortized costs — single-system projection only

No matched-recall comparison or break-even winner is established at10M. The geometric run achieved Recall10=.402 and Recall30=.391 at fixed configuration; static/size-tiered10M results are not yet available in this checkpoint. No original MUNI execution is measured.

`amortized.py` accepts measured maintenance wall sum, query mean, acknowledged writes, population and recalls. It rejects infeasible recall targets and cross-system recall spreads greater than one percentage point. CPU accounting requires separate measured CPU inputs and is not inferred from wall time.

The saved `continuation-25/amortized-input.json` and `amortized-projection.jsonl` illustrate only geometric DLI at target.4/.39. They use3540.055s job-wall sum,10,120,191 inserted vectors and3.374278s measured final-query mean. Under the explicit stationary-final-query approximation, projected maintenance+query seconds/query are approximately3.4093 at QPI=.01,3.3778 at.1,3.3746 at1,3.3743 at10/100. These are extrapolations, not billions of executed queries. The script also emits an acknowledged-write-inclusive additive cost proxy; overlapping operations mean this sum is not calendar elapsed time.

No selected low-QPI interleaved workload validates this projection. Earlier query costs differ materially: the model deliberately does not assume lifetime-stationary cost or interpolate unmeasured query checkpoints. A competitive service comparison requires recall-matched held-out budgets and measured static/size-tiered results. No QPI preference region or Qdrant-integration overhead ratio is supported yet.
