# Phase F.4: fixed-partition oracle headroom and retrieval-target MLP control

## Main result

At four probes, the fixed-partition oracle reaches Recall@10 0.9803. The existing corpus-label MLP reaches 0.8822, leaving 0.0981 mean recall headroom (95% exploratory paired query interval [0.0883, 0.1081]). This is substantial enough to justify testing an alternative query router while keeping the corpus partition fixed.

The retrieval-target MLP control uses the same 768 -> 64 ReLU -> 64 architecture and fixed postings as the existing MLP. It changes only the supervised query-routing target. Each training query contributes ten labeled copies, one for each exact top-10 neighbour bucket; this is empirical cross entropy on r_b(q)/10. Therefore any difference from the existing MLP cannot be attributed solely to prototype memory, which is not implemented in F.4.

## Experimental scope

The experiment fixes the canonical F.2 `n4096_e60_s42` corpus partition before F.4 observations. Its seed was predeclared rather than selected from F.3 test results. The indexed corpus has 96,403 live vectors after withholding every F.4 query row and exact normalized duplicates. Query split: 2,048 routing-training, 300 validation, 1,000 test and 20 warmups. The selected rows were excluded from F.2 training and F.3 queries but had been corpus rows in earlier experiments; this is fresh internal routing evaluation, not external validation.

One selected query has cosine similarity >= 0.9999 to an earlier training/query row, although it is not an exact normalized duplicate. The split was frozen before running, so it is retained and reported. Exact duplicate query rows were removed from the corpus. No query is permitted to retrieve itself.

## Four-probe comparison on the final test split

| Method | Recall@10 | Mean candidates | Corpus fraction | Oracle gap |
|---|---:|---:|---:|---:|
| oracle_p4 | 0.9803 | 8192.1 | 0.0850 |  |
| existing_mlp_s42_p4 | 0.8822 | 9215.5 | 0.0956 | 0.0981 |
| retrieval_target_mlp_s42_p4 | 0.9071 | 11233.2 | 0.1165 | 0.0732 |
| retrieval_target_mlp_s43_p4 | 0.9070 | 11207.9 | 0.1163 | 0.0733 |
| retrieval_target_mlp_s44_p4 | 0.9112 | 11123.5 | 0.1154 | 0.0691 |

## Oracle interpretation

The oracle knows the exact answer before choosing buckets. It is an upper bound on Recall@10 for any query-side method that selects four buckets from this exact frozen partition and then scores their contents exactly. It is not deployable and its candidate counts are only descriptive because equal-relevance bucket ties can be broken in multiple ways. The bound does not cover improvements that change the corpus partition, number of probes, scoring method, metric, or k.

For the existing MLP, 414 of 1,000 test queries have positive oracle headroom and 586 already meet the oracle ceiling. The maximum per-query gap is 1.0 Recall@10. This shows where a better query selector may help; it does not guarantee that a trainable selector can realize the bound.

## Retrieval-target control

| Model | Oracle minus practical recall at p=4 [95% CI] |
|---|---:|
| existing_mlp_s42 | 0.0981 [0.0883, 0.1081] |
| retrieval_target_mlp_s42 | 0.0732 [0.0655, 0.0811] |
| retrieval_target_mlp_s43 | 0.0733 [0.0660, 0.0813] |
| retrieval_target_mlp_s44 | 0.0691 [0.0617, 0.0767] |

The practical curves and validation-selected operating points are retained in the CSV/JSON artifacts. The validation policy chooses the smallest p reaching each target recall and is frozen before test; it must not be reselected from test curves. The p=4 oracle result remains the principal headroom question requested for this phase.

## Verification

- 452 F/F.2/F.3 source artifacts retained identical SHA-256 hashes.
- Fixed postings cover exactly 96,403 live offsets once; no corpus reassignment occurs.
- 480 native Qdrant scorer checks compare fixed-bucket membership recall to actual scored top-k results; full-probe warmups match exact canonical IDs and scores.
- 3 retrieval-target routers trained; Qdrant serving, persistence, configuration and existing learned postings were not changed.
- Full probe reaches all ten exact neighbours for every dense membership observation.

## Limits and next decision

F.4 is a controlled diagnostic, not a claim that the objective-controlled router generalizes beyond this dataset/split. The empirical soft target is derived from exact search and therefore expensive to create. The three seeds vary initialization and minibatch order only; they do not cover partition, query-split or dataset uncertainty. No HNSW, build, persistence, lifecycle or HTTP result is added.

Decision rule: if a retrieval-target MLP closes most of the oracle gap, objective choice is a stronger explanation than prototype memory. If material headroom remains after this control, prototype routing has a clearer rationale. Either result preserves the completed Qdrant integration and evaluates new routing ideas independently.

## Reproduction

Use a fresh F.4 output directory. Run `tests/lmi_phase_f4_prepare.py`, the ignored `phase_f4_oracle_and_objective_control` test with `--features lmi-training --locked`, then `tests/lmi_phase_f4_analyze.py`. Raw per-query oracle and router observations, targets, routers, selection, plots and source-preservation evidence are retained in the F.4 directory. No commit or push was performed.
