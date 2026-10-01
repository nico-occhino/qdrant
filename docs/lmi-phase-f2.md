# Phase F.2 — fixed-teacher routing diagnosis and bounded ablations

Repository base: `6e3cfdb0f03eb4ef09dc38334a8d41ef2f4a9fe0` (committed Phase F after a1987b504). No commit or push. The preserved Phase F artifacts were read-only inputs; every recorded baseline hash still matches.

## Findings and answers

**The 30-epoch baseline was substantially underfit. More training repairs most empty classes and changes the retrieval conclusion, but does not establish a general MLP advantage.**

1. **Why 24 empty buckets?** All 24 classes were represented in training (1–29 examples each, median 7), but none won a corpus top-1 prediction. The baseline got only 71.73% of training labels correct. Most empty classes are rare, and incorrect assignments are predominantly geometrically local. In the fixed-sample, fixed-seed intervention, going from 30 to 60 to 120 epochs changes empty counts from 23–24 to 9–10 to 1–2. Learning-curve prefixes match exactly across epoch budgets for each seed. This directly demonstrates that insufficient budget contributed substantially; it does not isolate a unique mechanism for every class. Empty classes comprise 10.21% of the sample versus 7.00% of the teacher corpus, so absence or collective undercoverage is not the explanation. They account for only about 19.6% of all corpus classification errors: empty buckets alone cannot explain the full approximation problem.
2. **Does more coverage/budget restore the teacher?** Yes, substantially but incompletely. At 2,048/120, training agreement reaches 99.71–99.80%, but common untrained-corpus agreement is only 79.48–79.87%. At 16,384/60 the common untrained-corpus agreement reaches 86.17–86.34%, with 1–2 empty buckets and validation agreement 87–90%. The final empty classes are tiny: class 28 (4 corpus vectors) persists in all these strongest runs; some seeds also miss class 45 (3 vectors) or 60 (1 vector). Larger samples bring more updates as well as coverage, so those contributions are not separately identified.
3. **Does better approximation reduce inflation?** Yes in the observed study. At nprobe=4 on the 100-query final test, the original MLP averages 14,494.21 candidates; the selected 4,096/60 models average 9,372.16–9,565.69. Largest corpus bucket size falls from 7,361 in the original model to 4,399–4,506 in the selected models (teacher maximum: 4,501). Better occupancy accompanies the reduction, but no claim is made that empty-bucket count alone mediates it.
4. **Can MLP match or beat centroid candidate efficiency?** There is a positive, bounded observation near 0.90 recall. Centroid uses 10,231.49 candidates at recall 0.906; seeds 42/43/44 use 9,565.69 / 9,500.48 / 9,372.16 at 0.901 / 0.907 / 0.910. That is 6.5–8.4% fewer candidates, with two seeds achieving slightly higher observed recall as well. Near the .95 target, MLP uses 16,464–16,824 versus 20,231 candidates, but its recall is 0.952–0.959 versus 0.961, so the quality difference must be retained. At the closest measured points to .99, MLP nprobe=32 uses 50,609–51,443 candidates at 0.994–0.996, versus centroid nprobe=16 at 38,856 and 0.993. MLP nprobe=16 offers lower work (28,515–29,150) at lower recall (0.976–0.985); the full curves make this discrete-budget tradeoff visible. There is no uniform dominance.
5. **Does the gain justify extra complexity?** This is a promising local candidate-work result, not yet a case for replacing centroid routing. The selected models cost 0.586–0.676 optimizer seconds and 0.876–1.075 instrumented training seconds, plus 3.581–3.884 seconds for diagnostic corpus classification/postings in this run. These exclude the fixed teacher's construction and native segment publication. The modest near-.90 gain, high-recall tradeoff, selection sensitivity and remaining uncertainty do not establish a general production benefit. Keep centroid/affine as the essential controls.
6. **What remains unexplained?** Rarity is associated with failures, but imbalance was not independently manipulated; rare-class geometry, optimization and finite-width approximation are not separately identified. Generalization remains imperfect despite almost-perfect training fit. Sample growth confounds coverage with update count. Seed replication shares one sample hierarchy and one teacher. The validation score is sensitive to missing bands on the coarse probe grid, and the test has only 100 queries already used by Phase F. No alternative losses, architectures, hidden widths, normalization, dropout, GPU or online learning were tried.

At nprobe=4, paired 2,000-resample query bootstraps give candidate-saving intervals excluding zero for all three selected models, but recall-difference intervals include zero: seed42 [-0.035, 0.0220], seed43 [-0.020, 0.020], seed44 [-0.021, 0.028]. These are exploratory percentile intervals for fixed trained models on this query sample; they do not cover teacher/training uncertainty or establish formal recall non-inferiority. Exact deltas, candidate intervals and seed 20260929 are in paired_bootstrap.json. Thus the slightly higher observed recall in two seeds is not a demonstrated statistical recall improvement.

The next bounded study should freeze these models, use a denser probe sweep around target bands, a larger genuinely new query set, and paired same-harness centroid/affine timing. Add a matched-update-count control before attributing sample-size gains to coverage. Diagnose the remaining rare-class errors before considering architecture changes. None of that follow-up was implemented here.

Low teacher top-1 agreement can coexist with good ANN recall because they measure different events. ANN probes multiple buckets and exactly scores their vectors; it need not reproduce the teacher's single winning label. MLP changes both query choices and corpus membership, sometimes losing teacher-retrieved neighbors and sometimes recovering others. At baseline nprobe=4 the losses (0.53) and gains (0.57) nearly cancel, but the larger candidate set costs work. The teacher's own partition is not exact-neighbor ground truth: exact Qdrant top-10 results are.

All conclusions are conditional on this LAION split, fixed 64-bucket teacher, 768→64→64 architecture and CPU training regime. Candidate savings are an observed local result; recall superiority, general latency improvement, and overall superiority of nonlinear routing are not established.

## Scope and experimental design

Same 99,780 indexed LAION vectors, Cosine metric, k=10, 64 saved f64 centroid teachers, 768→64→64 ReLU MLP, Adam 0.001, batch 256, CPU-only native Rust/tch training and Qdrant BatchFilteredSearcher scoring. The teacher is never reclustered. Centroid/affine postings and HNSW are unchanged. Baseline model and postings are checked point-for-point; all 2,800 baseline MLP recall/candidate observations are reproduced exactly. Timings are newly measured separately, never substituted into the preserved Phase F record.

The 20 warmup queries remain separate. The original 200 measured queries are divided into 100 validation and 100 final-test queries using Python Random seed 20260927. Test was withheld from F.2 configuration selection; it is not previously unseen data, since Phase F reported all 200 and the requested baseline diagnosis inspects them. Exact query indices and source rows are recoverable from protocol.json and the original dataset manifest. No model trains on either query subset.

Training uses nested corpus samples: the original sorted 2,048 offsets followed by a seed-20260928 permutation of the remaining corpus. Every prefix is sorted before training. Samples are shared across seeds 42/43/44; these replicate initialization/minibatch order, not sample-selection or teacher uncertainty. Stage 1 uses 2,048 samples × 30/60/120 epochs × three seeds (9 models). Stage 2 uses 4,096/8,192/16,384 samples at the validation-selected 60 epochs × three seeds (9 more). Hidden width stays 64. This is 18 models, not the 36-model Cartesian grid.

Selection was frozen before running models: at validation targets .90 and .95, accept measured points within ±.02, choose closest recall then candidate fraction, average candidate fractions across targets/seeds. Missing bands incur a cost of 1.0; ties prefer lower sample/epoch budget. This makes coarse nprobe coverage part of the operational selection score, not proof that a model cannot attain an unmeasured recall. Exact scores and missing bands are retained in epoch_selection.json and final_selection.json. Model selection never uses teacher accuracy alone. The chosen configuration is **4096 samples, 60 epochs**, tested with all three seeds after freezing selected_jobs.json. This is the protocol-selected configuration, not an established globally optimal model. The penalty strongly affects selection: 120 epochs at 2,048 samples misses two validation bands; 8,192 and 16,384 samples miss one each, despite better teacher fit.

Increasing sample size at fixed epochs increases both coverage and optimizer update count. That stage is a training-adequacy intervention; it cannot uniquely attribute improvement to coverage rather than extra updates. The fixed-sample epoch stage isolates training budget more directly.

## What the 24 empty baseline buckets mean

An empty MLP bucket means no corpus vector has that class as its largest native router logit. It does not mean the corresponding teacher class is absent or a hidden neuron is dead. 0 of the 24 empty classes have zero teacher-labelled training examples. Median training support is 7.0 examples for empty output classes versus 48.0 for active classes. Empty classes account for 10.21% of training labels and 7.00% of corpus teacher labels. This quantifies rarity and sample/corpus mismatch without asserting that imbalance alone caused the result.

Baseline native training top-1 agreement is 0.7173; remaining-corpus agreement is 0.6418; warmup agreement is 0.8500; measured-query agreement is 0.6150. The earlier 0.6364 pooled warmup and measured queries. Matrices and per-class tables distinguish these groups.

Of incorrect remaining-corpus assignments, 68.85% point to one of the teacher centroid's four nearest other centroids, and 83.96% to one of its eight nearest; mean destination-centroid rank is 4.96. This is descriptive geometry, not a size-adjusted causal test. Per-class tables retain teacher support, dominant predicted class/fraction, conditional entropy, number of merged teacher classes and MLP-bucket teacher entropy.

![Baseline confusion](../../lmi-phase-f2-data/baseline_confusion.png)

## Separating partition loss, approximation and candidate inflation

For each query the trace records exact Qdrant top-10 IDs/scores, both corpus labels, both full query rankings, and top-p membership. TT means teacher query routing + teacher postings; MT means MLP query routing + teacher postings; TM means teacher routing + MLP postings; MM is the actual MLP pipeline. MT/TM are counterfactual diagnostics, not serving baselines. They separate assignment effects without calling a hybrid a valid centroid index.

At nprobe=4 over the original 200 measured queries:

| Pipeline | Recall@10 from exact-neighbor membership | Mean candidates |
| --- | ---: | ---: |
| TT | 0.8975 | 10196.565 |
| MT | 0.8155 | 11628.375 |
| TM | 0.8440 | 11627.000 |
| MM | 0.9015 | 14503.615 |

Teacher routing misses 1.025 exact neighbors per query. Relative to TT, MM additionally loses 0.530 and recovers 0.570; losses and gains must both be counted. Along TT→MT→MM, query routing adds/removes 3419.715/1987.905 candidates, then changed corpus assignments add/remove 4658.695/1783.455. Their net changes sum to MM minus TT; gross additions alone are not net inflation. Reversing the counterfactual order gives another attribution, so this is a controlled accounting decomposition, not a unique causal mechanism.

All seven probe budgets and individual lost/gained neighbor identities are retained in baseline/trace.jsonl and baseline_decomposition.csv/json.

## All trained models

Accuracies below are native-router agreement with the fixed teacher. Curves retain post-epoch full-sample Torch loss/accuracy. A 2,048/30/seed42 run must reproduce the saved baseline router exactly; that assertion passed. Export parity checks still execute for each new model.

| Model | Train agreement | Remaining corpus | Validation agreement | Empty buckets | Final loss | Optimizer seconds | Instrumented training s | Classification/postings s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n16384_e60_s42 | 0.9946 | 0.8634 | 0.8700 | 1 | 0.094597 | 2.643786 | 4.007195 | 3.664726 |
| n16384_e60_s43 | 0.9949 | 0.8617 | 0.8900 | 2 | 0.092998 | 2.487680 | 3.791741 | 3.602115 |
| n16384_e60_s44 | 0.9942 | 0.8629 | 0.9000 | 2 | 0.097159 | 2.580773 | 3.890026 | 3.604322 |
| n2048_e120_s42 | 0.9980 | 0.7955 | 0.8300 | 1 | 0.127626 | 0.616428 | 0.901323 | 3.913875 |
| n2048_e120_s43 | 0.9971 | 0.7943 | 0.8300 | 1 | 0.129688 | 0.646735 | 0.952692 | 3.937618 |
| n2048_e120_s44 | 0.9971 | 0.7984 | 0.8400 | 2 | 0.126575 | 0.636664 | 0.930288 | 3.902483 |
| n2048_e30_s42 | 0.7173 | 0.6418 | 0.5500 | 24 | 1.629805 | 0.197336 | 0.471575 | 3.887041 |
| n2048_e30_s43 | 0.7285 | 0.6446 | 0.6300 | 23 | 1.633250 | 0.143182 | 0.212891 | 3.575892 |
| n2048_e30_s44 | 0.7251 | 0.6500 | 0.6200 | 24 | 1.647340 | 0.142689 | 0.212030 | 3.576021 |
| n2048_e60_s42 | 0.9209 | 0.7692 | 0.7800 | 10 | 0.599165 | 0.285586 | 0.422173 | 3.667283 |
| n2048_e60_s43 | 0.9268 | 0.7717 | 0.7600 | 10 | 0.589108 | 0.284157 | 0.419844 | 3.570796 |
| n2048_e60_s44 | 0.9268 | 0.7736 | 0.7900 | 9 | 0.590176 | 0.302944 | 0.447610 | 3.719012 |
| n4096_e60_s42 | 0.9700 | 0.8140 | 0.8100 | 6 | 0.278314 | 0.676110 | 1.074511 | 3.580645 |
| n4096_e60_s43 | 0.9714 | 0.8112 | 0.8200 | 6 | 0.276319 | 0.586103 | 0.875970 | 3.883598 |
| n4096_e60_s44 | 0.9719 | 0.8129 | 0.8500 | 5 | 0.275972 | 0.587982 | 0.880085 | 3.810300 |
| n8192_e60_s42 | 0.9882 | 0.8392 | 0.8300 | 3 | 0.154421 | 1.200597 | 1.827638 | 3.695515 |
| n8192_e60_s43 | 0.9874 | 0.8374 | 0.8900 | 4 | 0.156462 | 1.331572 | 1.995460 | 3.808962 |
| n8192_e60_s44 | 0.9879 | 0.8379 | 0.8500 | 4 | 0.152500 | 1.233583 | 1.866234 | 3.943661 |

The fixed-label trainer reuses the operational optimizer/export logic. Normal production calls have no observer. Instrumented training includes full-sample epoch diagnostics; optimizer seconds exclude those evaluations. Classification/postings also records top-8 predictions, so it is diagnostic construction cost, not a native optimizer-owned segment build. No clustering, segment copy/publication, HTTP scheduling or persistence-reopen time is included. Saved routers and native corpus assignments permit independent reanalysis. model_summary.csv also records whole-corpus agreement, occupancy percentiles and agreement on a common 83,396-vector cohort excluded from every training sample; the model-specific remaining-corpus complements change as samples grow.

![Epoch ablation](../../lmi-phase-f2-data/epoch_ablation.png)

![Training curves](../../lmi-phase-f2-data/training_curves.png)

## Final-test matched recall

Only measured points within ±.02 are shown; closest recall wins, then candidates. Missing points are shown explicitly. Test results did not change the configuration selection. Centroid/affine/HNSW reference timings are unchanged historical Phase F measurements subset to the same query IDs; F.2 MLP timings are current in-process common-harness measurements. Candidate counts and recall are deterministic comparison quantities here. Timing comparisons across runs are descriptive and not paired contemporary trials. HNSW candidate visits remain unavailable.

| Target | Model | nprobe / ef | Actual recall | Mean candidates | Median ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.90 | centroid | 4 | 0.9060 | 10231.5 | 3.527 |
| 0.90 | affine | 4 | 0.9060 | 10231.5 | 3.505 |
| 0.90 | hnsw | 16 | 0.9160 | unavailable | 0.156 |
| 0.90 | plain | — | no measured point in band | — | — |
| 0.90 | baseline | 4 | 0.9030 | 14494.2 | 4.995 |
| 0.90 | n4096_e60_s42 | 4 | 0.9010 | 9565.7 | 2.372 |
| 0.90 | n4096_e60_s43 | 4 | 0.9070 | 9500.5 | 2.239 |
| 0.90 | n4096_e60_s44 | 4 | 0.9100 | 9372.2 | 2.230 |
| 0.95 | centroid | 8 | 0.9610 | 20230.6 | 6.256 |
| 0.95 | affine | 8 | 0.9610 | 20230.6 | 6.348 |
| 0.95 | hnsw | — | no measured point in band | — | — |
| 0.95 | plain | — | no measured point in band | — | — |
| 0.95 | baseline | 8 | 0.9680 | 26681.8 | 8.697 |
| 0.95 | n4096_e60_s42 | 8 | 0.9580 | 16823.7 | 3.972 |
| 0.95 | n4096_e60_s43 | 8 | 0.9520 | 16464.1 | 3.872 |
| 0.95 | n4096_e60_s44 | 8 | 0.9590 | 16632.0 | 3.834 |
| 0.99 | centroid | 16 | 0.9930 | 38855.8 | 11.320 |
| 0.99 | affine | 16 | 0.9930 | 38855.8 | 11.126 |
| 0.99 | hnsw | 64 | 0.9890 | unavailable | 0.322 |
| 0.99 | plain | 0 | 1.0000 | 99780.0 | 21.479 |
| 0.99 | baseline | 16 | 0.9860 | 44152.4 | 12.742 |
| 0.99 | n4096_e60_s42 | 32 | 0.9950 | 51443.3 | 10.509 |
| 0.99 | n4096_e60_s43 | 32 | 0.9960 | 50608.9 | 10.221 |
| 0.99 | n4096_e60_s44 | 32 | 0.9940 | 51124.8 | 10.458 |

![Final test tradeoffs](../../lmi-phase-f2-data/test_tradeoffs.png)

Every trained model has validation Recall@10 and candidate counts/fractions across nprobe 1,2,4,8,16,32,64, two trials and 20 warmup queries per sweep. Router, preparation, Qdrant scoring and total harness p50/p95/p99 are in summary.csv/json; raw nanosecond observations remain per query/trial. Full-probe results must equal exact Qdrant results, including scores. HNSW was not rebuilt or tuned. This is not HTTP latency, and percentile digits do not imply physical clock accuracy. Three seeds and one split are insufficient for broad superiority claims.

F.2 uses the common native BatchFilteredSearcher over the freshly populated Plain segment, with explicit router-generated candidates; it does not invoke Plain search for approximate results. F used the same scorer over a built/reopened LMI target. Both configurations use float32 InRamChunkedMmap vector storage, but tracker ownership, process/cache state, compiled wrappers and timing order differ. Thus lower F.2 harness times must not be attributed solely to the model or treated as a contemporary centroid speedup. Baseline tracing also adds untimed work between calls. Candidate counts and exact-neighbor recall are the primary cross-run conclusions. The next latency claim would require paired same-harness timing of unchanged controls and frozen models.

## Reproduction and artifact ownership

Use the same CPU Torch environment as Phase F: Python 3.11.16, NumPy 2.2.4, Torch 2.5.1+cpu, tch 0.18.1; Rust 1.98.0, clang 21.1.8; Intel Core Ultra 9 285H, WSL2, CPU0 pinned, Torch/Rayon/OMP/OpenBLAS one thread. Release opt3/fat LTO/codegen-units1. Plotting uses the isolated matplotlib 3.10.8 / NumPy 2.4.6 directory, not operational Qdrant dependencies. All exact commands and exit codes are retained in verification/.

```bash
export PATH=/home/nicoo/.cargo/bin:/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export CXX=clang++ CXXFLAGS=-g0 CARGO_BUILD_JOBS=2
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LMI_PHASE_F_DIR=/home/nicoo/work/lmi-phase-f-data
export LMI_PHASE_F2_DIR=/home/nicoo/work/lmi-phase-f2-data
# Use a fresh F.2 directory for reproduction; create_new guards protect completed files.
python tests/lmi_phase_f2_prepare.py
LMI_PHASE_F2_MODE=baseline cargo test --release -p segment --features lmi-training --locked --lib phase_f2_study -- --ignored --nocapture --test-threads=1
python tests/lmi_phase_f2_analyze.py diagnose
python tests/lmi_phase_f2_analyze.py epoch_jobs
LMI_PHASE_F2_MODE=jobs cargo test --release -p segment --features lmi-training --locked --lib phase_f2_study -- --ignored --nocapture --test-threads=1
python tests/lmi_phase_f2_analyze.py sample_jobs
LMI_PHASE_F2_MODE=jobs cargo test --release -p segment --features lmi-training --locked --lib phase_f2_study -- --ignored --nocapture --test-threads=1
python tests/lmi_phase_f2_analyze.py select
LMI_PHASE_F2_MODE=test cargo test --release -p segment --features lmi-training --locked --lib phase_f2_study -- --ignored --nocapture --test-threads=1
python tests/lmi_phase_f2_analyze.py final
```

Baseline source binaries are not copied into this bundle. The original source hash/split manifest plus Phase F preparation reproduce them; baseline model and centroid hashes are frozen in protocol.json. F.2 owns separate model directories, curves, raw rank arrays, confusion tables, traces, selections and summaries. Corpus rank binary files are row-major uint8 [99780,8]; rows map to Phase F corpus offsets. Query rows 0–19 are warmup and 20–219 correspond to Phase F measured IDs 0–199. No Python model training/scoring is used.

## Changes and verification

- training.rs: extract the same optimizer/export implementation behind a fixed-label seam and optional epoch observer; normal trainer still clusters then passes the original labels with no observer.
- evaluation.rs / evaluation_f2.rs: test-only fixed-teacher diagnosis and ablations, exact baseline/export/full-probe guards.
- tests/lmi_phase_f2_prepare.py / lmi_phase_f2_analyze.py: frozen protocol, offline diagnostics and validation-only stage selection.
- docs/lmi-phase-f2.md: this engineering/scientific record.

- `analysis`: exit 0, 0.86 s; command `python tests/lmi_phase_f2_analyze.py final`.
- `baseline`: exit 0, 43.71 s; command `cargo test --release -p segment --features lmi-training --locked --lib phase_f2_study -- --ignored --nocapture --test-threads=1`.
- `compile`: exit 0, 653.8 s; command `cargo test --release -p segment --features lmi-training --locked --lib phase_f2_study --no-run`.
- `diagnosis`: exit 0, 0.48 s; command `python tests/lmi_phase_f2_analyze.py diagnose`.
- `epoch-plan`: exit 0, 0.07 s; command `python tests/lmi_phase_f2_analyze.py epoch_jobs`.
- `epochs`: exit 0, 146.71 s; command `cargo test --release -p segment --features lmi-training --locked --lib phase_f2_study -- --ignored --nocapture --test-threads=1`.
- `final-cluster-unit`: exit 0, 26.98 s; command `cargo test -p segment --features lmi-training --locked --lib lloyd_separates_clusters_and_obeys_cancellation -- --nocapture`.
- `final-collection-check`: exit 0, 11.77 s; command `cargo check -p collection --tests --features segment/lmi-training --locked`.
- `final-collection-unit`: exit 0, 29.93 s; command `cargo test -p collection --features segment/lmi-training --locked --lib lmi_config_ -- --nocapture`.
- `final-default-check`: exit 0, 8.6 s; command `cargo check --bin qdrant --locked`.
- `final-diff-check`: exit 0, 0.01 s; command `git diff --check`.
- `final-edge-check`: exit 0, 3.23 s; command `cargo check -p edge --locked`.
- `final-fmt`: exit 0, 3.32 s; command `cargo fmt --all`.
- `final-grpc-unit`: exit 0, 11.8 s; command `cargo test -p api --locked --lib lmi_configuration_defaults_roundtrip_and_validation -- --nocapture`.
- `final-lifecycle`: exit 0, 5.19 s; command `python3 tests/lmi_phase_e2_lifecycle.py --port 16733 --output /mnt/c/Users/nicoo/Documents/Codex/2026-09-21/referenced-chatgpt-conversation-this-is-an/work/phase_f2/final-lifecycle`.
- `final-lmi-tests`: exit 0, 22.66 s; command `cargo test -p segment --features lmi-training --locked --test lmi_candidate_scoring --test lmi_dummy --test lmi_phase_c --test lmi_phase_d --test lmi_phase_e -- --nocapture`.
- `final-segment-check`: exit 0, 7.39 s; command `cargo check -p segment --tests --features lmi-training --locked`.
- `final-snapshot`: exit 0, 4.67 s; command `python3 tests/lmi_phase_e_http_smoke.py --port 16633 --output /mnt/c/Users/nicoo/Documents/Codex/2026-09-21/referenced-chatgpt-conversation-this-is-an/work/phase_f2/final-snapshot`.
- `final-test`: exit 0, 45.0 s; command `cargo test --release -p segment --features lmi-training --locked --lib phase_f2_study -- --ignored --nocapture --test-threads=1`.
- `final-training-build`: exit 0, 41.55 s; command `cargo build --bin qdrant --features lmi-training --locked`.
- `sample-plan`: exit 0, 0.55 s; command `python tests/lmi_phase_f2_analyze.py sample_jobs`.
- `samples`: exit 0, 149.18 s; command `cargo test --release -p segment --features lmi-training --locked --lib phase_f2_study -- --ignored --nocapture --test-threads=1`.
- `selection`: exit 0, 0.78 s; command `python tests/lmi_phase_f2_analyze.py select`.

Tracked git diff --stat (untracked new files are listed separately in git-status.txt):

```text
 lib/segment/src/index/lmi_index/evaluation.rs |  3 +++
 lib/segment/src/index/lmi_index/training.rs   | 39 ++++++++++++++++++++++++---
 2 files changed, 38 insertions(+), 4 deletions(-)
```

Final verification: 36 existing LMI integration tests passed (2 candidate scoring, 1 dummy, 10 Phase C, 12 Phase D, 11 Phase E/E2). Six selected unit tests passed (clustering/cancellation 1, collection configuration 3, gRPC 1, centroid/affine 1). Four release harness invocations passed: baseline diagnosis, 9 epoch models, 9 sample-size models, and 3 frozen-model test evaluations. The latter reuse trained models; they are not three additional trainings. Collection/segment test compilation, edge check, default server check and training-enabled server build all passed with --locked. Both isolated HTTP lifecycle drivers passed, including restart/snapshot no-retraining, learned-path evidence, preserved state/results, deferred rebuilds, updates/deletions and mixed named indexes. Stable rustfmt warnings were the known nightly-only option warnings; the one unrelated import-format hunk was restored. Final git diff --check passes. No baseline hash changed, no dependency/lockfile/API/persistence-format changes, no commit or push.

The archive contains 32,200 measured per-query/per-trial observations: 2,800 original-model diagnostic rows, 25,200 validation rows from 18 trained models, and 4,200 final-test rows from the three selected models. Warmup is excluded from these counts. protocol.json freezes source hashes, split and sampling/selection rules. Epoch loss/accuracy curves, every model, corpus top-8 predictions, query rankings, confusion matrices, per-class metrics and full baseline neighbor traces are retained. Raw timing values are never replaced by aggregate values.

Final verification matrix (saved executions; not rerun during continuation):

| Check | Exit | Wall seconds |
| --- | ---: | ---: |
| final-cluster-unit | 0 | 26.98 |
| final-collection-check | 0 | 11.77 |
| final-collection-unit | 0 | 29.93 |
| final-default-check | 0 | 8.60 |
| final-diff-check | 0 | 0.01 |
| final-edge-check | 0 | 3.23 |
| final-equivalence | 0 | 0.67 |
| final-fmt | 0 | 3.32 |
| final-grpc-unit | 0 | 11.80 |
| final-lifecycle | 0 | 5.19 |
| final-lmi-tests | 0 | 22.66 |
| final-segment-check | 0 | 7.39 |
| final-snapshot | 0 | 4.67 |
| final-test | 0 | 45.00 |
| final-training-build | 0 | 41.55 |
