# Native LMI integration: code and reproducibility guide

This research branch integrates a static Learned Metric Index (LMI) into Qdrant. The pinned development revision is `9b12d583b1bf9df802b2cc420244bd3c24732d86`; the research working tree may contain additional uncommitted changes. Record the exact commit, overlays, compiler, configuration, and input manifests for every experiment.

## Reading the implementation

- `lib/segment/src/index/lmi_index/mod.rs`: Qdrant vector-index integration, learned candidate routing, scoring through Qdrant's candidate scorer, and fallback behavior.
- `routing.rs`, `postings.rs`, and `state.rs`: learned router, bucket postings, and persisted state metadata. `read_only.rs` implements the read-only serving path.
- `build.rs`, `build_plan.rs`, `training.rs`, and `spherical_kmeans.rs`: index preparation and training. Training is behind the `lmi-training` feature and has PyTorch/libtorch requirements; static serving does not require training.
- `lib/collection/src/tests/lmi_lifecycle.rs` and the `lmi_sisap23_*` test modules: collection lifecycle and research harnesses. `lib/segment/tests/lmi_static_119.rs` and `lmi_training_119.rs` cover segment-level behavior.
- `tools/lmi/sisap23_matched/README.md` and `recovery_stages.py`: matched-experiment workflow and recovery orchestration; consult the scripts and their source hashes before reproducing a run.

This is a static learned index: the trained index requires segment rebuild for vector updates. Do not infer implemented dynamic learned maintenance or a final comparative performance result from this branch.

## Checks

Run the targeted Python tests from the repository root:

```sh
PYTHONPATH=tools/lmi/sisap23_matched python3 -m unittest \
  tools.lmi.sisap23_matched.test_generation_publication \
  tools.lmi.sisap23_matched.test_recovery_stages \
  tools.lmi.sisap23_matched.test_validate \
  tools.lmi.sisap23_matched.test_plain_ssd_panel
```

Rust compilation and the SISAP23 integration tests require the pinned toolchain, appropriate features, and a compatible libtorch environment for training. The historical MetaCentrum preflights `25027255` and `25068770` record synthetic correctness and evaluator testing respectively; those results identify their deployed overlay sets and do not automatically validate later working-tree edits.

## Evidence and limitations

The full-scale LMI recovery job `25057641` formally published an accepted generation. HNSW-A job `25071582` published an accepted baseline. These are artifact-acceptance observations, not a completed matched LMI/HNSW/Plain quality–latency evaluation. The full dataset, model and collection files, build caches, credentials, and transient job outputs should not be committed. Experiment-specific absolute cluster paths in the historical orchestration are not portable; adapt them only while preserving the recorded scientific configuration and acceptance checks.
