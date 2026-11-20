# Local build environment (WSL)

The LMI training path (`lmi-training` feature, `tch` 0.18) links against libtorch from a
Python PyTorch install. `tch` 0.18 requires **torch 2.5.x**.

The durable environment is the conda env `lmi-rust-inspect` (torch 2.5.1+cpu, Python 3.11).
Do not point builds at a Python venv inside a temporary scratch directory; one such venv was
lost when its scratch directory was cleared.

```bash
V=~/miniconda3/envs/lmi-rust-inspect
export PATH=$V/bin:$PATH
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$V/lib/python3.11/site-packages/torch/lib
unset LIBTORCH LIBTORCH_BYPASS_VERSION_CHECK
```

Changing the torch environment rebuilds `torch-sys` and everything that depends on it.

## Regression suites

The LMI/DLI regression suites used before merging into `main`:

```bash
cargo test --offline --locked -p segment --features testing,lmi-training \
  --test lmi_filtered_v120 --test lmi_static_v120 --test lmi_shell_v120 \
  --test lmi_training_v120 --test dli_native_topology --test candidate_scoring_v120 \
  --test supervised_routing_v120
cargo check --offline --locked -p segment --features testing,lmi-research,lmi-training --tests
cargo test --offline --locked -p segment --lib --features testing,lmi-training lmi
cargo test --offline --locked -p collection --lib --features testing,dli-research -- lmi dli
cargo build --offline --locked --release --bin qdrant
```

For an upstream sync, also run the full `segment`, `collection` and `shard` lib suites
(`cargo test --offline --locked -p <crate> --lib --features testing`; add `dli-research` for
`collection`).

## Upstream syncs

`main` follows upstream **release tags** (not `upstream/dev`) by merge, never rebase:
experiment provenance files cite exact commit SHAs, and a merge keeps them valid.
