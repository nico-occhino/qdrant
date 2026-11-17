# Source comparison

Qdrant starting SHA caf5e2b4695772bc500b79bae4974fe321d64060, clean. Archives supplied under Windows codes directory were extracted read-only for study into continuation32/references; archive SHA256 and ZIP comments are recorded in archive-provenance.json. ZIP comments identify revisions; this is archive-level provenance, not independent Git-server authentication (archives contain no checked-out Git metadata). No third-party implementation was copied into Qdrant.

|Archive|ZIP revision comment|SHA256|
|---|---|---|
|DynamicLearnedIndex-python.zip|28dc7132e6141ad66c222175ecc7f9b96165fd62|86bc7e8a92b88b408bec9fa75f181e6b7d15f5f40466796eed96f55f3ea58a6f|
|DynamicLearnedIndex-rust.zip|4e756476ae9352fcef7896cf5ef551282751c32b|54b093a463d6b886b610a38aa02e076284f367a2123a264a073e5a214e12f950|
|Emma_DLI.zip|dde0e57427bb0bf8046dc18ec2f6adc22abea162|99d9e1c1fc6d7f217dd1bb5dd9d944f35145bb77a2260ac1d797556f27c7cb28|
|flat-knn-master.zip|33383935ab2c6f60e2821f6eb8296fd9191626e8|4372560e60949e3ec360042eee3a47191ad6287c482dcd9b2e29088d84cba806|
|rust-knn-benchmarks-master.zip|6e7d378debd3615324148265757a7a78c022331b|cfd8db1e0017a0f2215409685d3fadb4b399d35c02d185f66cc858aab22aa57c|
|rust-ml-frameworks-benchmarks-master.zip|195c271c760c79290f0774bd87c867364b72925e|2735f3aa0fddf9d51f8b2c211544fce2fe3f2cb762c0ab1fa2fc1f7b24a8bb36|

**Two Python references must not be conflated.** Emma_DLI is dde0e574…; DynamicLearnedIndex-python is28dc7132…. The former source headers explicitly say LGPLv3 and provide lgpl.txt. Rust DLI LICENSE is LGPLv3; flat-knn LICENSE is MIT. No LGPL code has been linked/copied. Behavioral reimplementation and any future dependency adoption still require license review.

|Mechanism|Emma Python (dde0…)|Plhák Rust (4e756…)|Qdrant|Experiment / integration cost|
|---|---|---|---|---|
|Global priority|dynamic_learned_index.py495–575: predict each model; MDN multiplies arity**zero-based level; buffer probability1|index.rs107–128 arity.pow(level_idx)*nonnegative probability, buffer1|slot weighting; population density; sharedT2 alternative|Same models/postings work frontier; low cost, already available|
|Candidate budget|Whole buckets until accumulated population reaches a **minimum**, not hard cap|Bucket ordering incorporates occupancy|Whole postings, overshoot measured|Compare actual scorer work, not nominal budgets|
|Output count|BentleySaxe._allocate_new_level77 and dynamic_learned_index.py134: arity**one-based level|index.rs159 arity^(level+1)|Frozen10M all548 outputs|Adaptive occupancy-based output count, moderate training cost|
|Sampling|sampling.py randomized without replacement; threshold100000 appears in supplied experiment configs|sampling.rs min(total,max(k*40,threshold))|32768 fixed reservoir in frozen models|Hold teacher fixed before testing classifier sampling|
|Compaction/retraining|Explicit Python strategy implementations|index.rs493–525: new level train; existing level retrain only when storage.size()==0|Native builder retrains selected destination|Model-preserving rebuild requires generation-safe reassignment; high cost|
|Bucket scoring|object_container.py110 Faiss knn on live bucket slice, local topk then merges|flat-knn/bucket storage hot/cold|Original native scorer, bounded materialization; optional native scalar shortlist|Native quantized same-posting evaluation highest priority|
|Durability/updates|Standalone object containers|Standalone hot/cold bucket ownership|WAL, proxy/COW, native physical UUIDs, complete visibility resolver|Standalone timing cannot be treated as DB integration timing|

**Benchmark defect:** rust-knn-benchmarks benches/knn_benchmark.rs defines DIM=768 but benchmark_faiss121–137 calls index_factory(8,"Flat",L2) on flattened data and query arrays. With10000*768 floats, this interprets960000 8D records and96 query vectors, rather than10000 768D records and one query. The same timed closure also creates/adds the Faiss index each iteration. Its plotted kernel comparison cannot justify a Faiss-versus-flat-knn speedup. This is verified source behavior, not a rerun result. Do not modify the immutable supplied archive to hide the defect.

These source differences plausibly explain allocation and maintenance disparities, but do not numerically attribute our10M costs to a specific mechanism. MUNI normalization is geometric weighting, **not temperature calibration**. Varying output counts changes its meaning; importing the multiplier alone does not reproduce its architecture.
