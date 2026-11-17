# Routing evidence retained

No new heuristic introduced. Prior12-query10M held-out slot208525 scores gives Recall10/30 .8417/.8500 and p50 93.2ms; T2 density196225 gives .9417/.9389 and123.1ms. Bucket coverage explains the quality gain at similar work; no latency superiority established. This is not enough queries for architecture selection, and warm/cache order confounds small timing differences. Keep calibrated density experimental, original scoring supported. Native scalar work can be studied independently on identical learned candidates before another allocation sweep.
