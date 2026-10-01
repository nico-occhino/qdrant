# Dynamic Learned Index implementation in Rust

A Rust implementation of a dynamic learned index for efficient search in unstructured data. The index uses a multi-level structure where each level contains a neural network model that learns to predict the most likely bucket containing the requested data. This enables fast approximate nearest neighbor search by leveraging learned models instead of traditional indexing.


## Run


If you want to build the `cli_dynamic_learned_index` workspace, you need to have hdf5 library installed on your system. You can install it via your package manager.


Project provides basic CLI interface to quickly run experiments and build the index.

```shell
# either run with candle model
cargo run -p cli_dynamic_learned_index --no-default-features --features candle

# or install the libtorch and link it, for example like this
cd py_dynamic_learned_index
uv sync
source .venv/bin/activate
cargo run -p cli_dynamic_learned_index
```

Or run via Docker

```shell
docker build -t dli .
docker run -it -v "$(pwd)/data/:/app/data/" -v "$(pwd)/configs/:/app/configs/" dli test
```

### Feature flags

The whole project is written in Rust and uses `cargo` as a build system. The project is divided into several crates:

- `dynamic_learned_index`: main library crate that implements the dynamic learned index
- `cli_dynamic_learned_index`: CLI crate that provides a command line interface to run experiments and build the index
- `py_dynamic_learned_index`: Python crate that provides a Python interface to the dynamic learned index
- `measure_time_macro`: Simple macro that enables logging time of the function through macro.


**dynamic_learned_index** crate:

- `measure_time`: enables time measuring macro, mostly for development purposes
- `tch`: enables the use of `tch-rs` library for model training
  - You need to have `libtorch` installed on your system to use this feature. Follow the installation instructions for `libtorch` from [tch-rs homepage](https://github.com/LaurentMazare/tch-rs).
- `candle`: enabled by default, enables the use of `candle` library for model training
- `mkl`: enables the use of `mkl` library for linear algebra operations, can be used with `candle` feature to speed up training on CPU
- `mix`: enables the use of mixed model of `tch` (training) and `candle` (predictions).

**Note** It's not possible to have only `tch`, `candle` or `mix` feature enabled at the same time. 

**cli_dynamic_learned_index** crate:

- `measure_time`: enables time measuring macro, mostly for development purposes
- `kmeans`: just for running benchmarks for kmeans implementation, not used in the main codebase
- `kentro`: just for running benchmarks for kentro implementation, not used in the main codebase

**py_dynamic_learned_index* crate:

- `measure_time`: enables time measuring macro, mostly for development purposes
- `mkl`: enables the use of `mkl` library for linear algebra operations, can be used with `candle` feature to speed up training on CPU
- `tch`: enables the use of `tch-rs` library for model training, can be used with `candle` feature to speed up training on CPU
- `candle`: enables the use of `candle` library for model training, can be used with `mkl` or `tch` features to speed up training on CPU

### Linking with Python

To link the Rust library with Python, we use `maturin` to build a Python package. This allows us to use the Rust code as a Python module.

Setup python environment

```shell
cd py_dynamic_learned_index
uv sync
source .venv/bin/activate
maturin develop
python -c "import py_dynamic_learned_index; print(py_dynamic_learned_index.__version__)"  # test installation
```

You can also run example python script:
```shell
# make sure your virtual environment is activated
python py_dynamic_learned_index/example.py
```

Building the wheel in release mode:

```shell
cd py_dynamic_learned_index
uv sync
# --strip flag is needed due to the linking with pytorch
maturin build --release --strip
uv pip install ./target/wheels/py_dynamic_learned_index*.whl
```

## Running experiments via CLI

### Dataset

```yaml
dataset:
  type: <type>
  value:
    path: <path>
    dataset_name: <dataset_name>
queries:
  type: <type>
  value:
    path: <path>
    dataset_name: <dataset_name>
ground_truth:
  type: <type>
  value:
    path: <path>
    dataset_name: <dataset_name>
```

`<dataset_name>`: name of the underlining dataset under h5

`<type>`: type of the dataset, can be `h5`

`<path>`: path to the dataset file relative from dataset config directory.

Default values can be found via `cli_dynamic_learned_index defaults dataset` command.

### Index

Example config:

```yaml
# Compaction strategy configuration
compaction_strategy:
  type: bentley_saxe              # Strategy for managing level transitions (only option)
  rebuild_strategy: no_rebuild    # How to rebuild during compaction: no_rebuild, basic_rebuild, or greedy_rebuild

# Level configuration (repeated for multiple levels if needed)
levels:
  # Neural network model configuration
  model:
    # Neural network layers (add more layers as needed, always adds output layer automatically)
    layers:
      - type: linear              # Linear transformation layer
        value: 256                # Output dimensions
      - type: relu                # ReLU activation layer
    
    # Model training parameters
    train_params:
      threshold_samples: 1000     # Minimum samples required before training
      batch_size: 8               # Samples per training batch
      epochs: 3                   # Number of training passes
      max_iters: 10               # Maximum iterations for clustering
      retrain_strategy: no_retrain  # When to retrain: no_retrain or from_scratch
    
    # Optional: path to pre-trained model weights
    weights_path: null            # Load model weights from file
    
    # Optional: enable model quantization for compression
    quantize: false               # Reduce model precision to int8
    
    # Optional: random seed for reproducibility
    seed: 0                       # Seed for random number generation
  
  # Storage configuration for this level
  bucket_size: 5000               # Maximum records per bucket before split

# Global index configuration
buffer_size: 5000                 # Size of write buffer before flushing to index
input_shape: 768                  # Dimension of input vectors
arity: 3                          # Fanout for compaction (max child nodes per parent)
device: cpu                       # Compute device: cpu or cuda (cuda requires tch feature)
distance_fn: dot                  # Distance metric: dot (for cosine similarity) or l2 (euclidean)
delete_method: oid_to_bucket      # Deletion strategy (only option currently available)
```

**Configuration Notes:**

- `levels`: Index levels inherit from previous level until matched. First level is always 0. Omit levels field to use defaults for all levels.
- `layers`: Add as many layers as needed. Neural network automatically adds output layer based on model parameters.
- `distance_fn`: Choose `dot` for normalized embeddings (dot product), or `l2` for euclidean distance
- `rebuild_strategy`: `no_rebuild` (fastest), `basic_rebuild` (retrains models), `greedy_rebuild` (optimizes splits)
- `retrain_strategy`: `no_retrain` (use existing models), `from_scratch` (train all models from beginning)

Default values can be found via `cli_dynamic_learned_index defaults index` command.

### Running experiments

To run experiments via CLI, you can use the `cli_dynamic_learned_index` binary. Example:

```shell
./target/release/cli_dynamic_learned_index experiment test_build data/k300 --force
```

## Rust API

The `dynamic_learned_index` crate provides a public API for building and querying indices programmatically from Rust code.

```rust
use dynamic_learned_index::{Index, IndexBuilder, IndexConfig, SearchParams, SearchStrategy};

// Build an index with f32 float type
let mut index = IndexBuilder::<f32>::default()
    .input_shape(768)
    .buffer_size(5000)
    .bucket_size(5000)
    .arity(3)
    .build()?;

// Insert data
index.insert(vec![1.0, 2.0, 3.0], 0)?;

// Search for k nearest neighbors
let query = vec![1.0, 2.0, 3.0];
let results = index.search(&query, 10)?;
index.delete(0)?;
```

## Python API

This crate provides a Python API available on PyPI: https://pypi.org/project/py-dynamic-learned-index/

Example usage can be found in [`py_dynamic_learned_index/example.py`](py_dynamic_learned_index/example.py) directory.

## Docker

To run the project in a Docker container.

```shell
docker build -t dli-cli --exclude py_dynamic_learned_index .
docker run -it --rm -v ${PWD}/experiments_data:/app/experiments_data -v ${PWD}/data:/app/data -v ${PWD}/configs:/app/configs dli-cli
```

## Profiling

Recommended approach to optimize code is to go through cycle of measuring, analyzing and optimizing.

### Measurment

The measurment focuses on measuring search time. There are two measurements:

- for 3 individual queries (good for debugging)
- for the whole query set (good for overall performance measurment)

```shell
# ensure that you have built the latest release version
cargo build --release

# we need to build the index and serialize it
./target/release/cli_dynamic_learned_index experiment profiler  data/k300/ \
  -i configs/example.yaml \
  --skip-validation \
  --start-from-one \
  --force \
  --skip-search \
  --output-dir index_dump

# run benchmark and save as a baseline
cargo bench -p cli_dynamic_learned_index --bench index_benchmarks -- --warm-up-time 5 --measurement-time 15 --save-baseline base

# run benchmark and compare with the baseline
cargo bench -p cli_dynamic_learned_index --bench index_benchmarks -- --warm-up-time 5 --measurement-time 15 --baseline base 
```

Results are in `target/criterion/report/index.html` file.

```shell
# open the report
open target/criterion/report/index.html
```

### Analyzing

```shell
# ensure that you have built the latest release version
cargo build --release

# we need to build the index and serialize it
./target/release/cli_dynamic_learned_index experiment profiler  data/k300/ \
  -i configs/example.yaml \
  --skip-validation \
  --start-from-one \
  --log-output stdout \
  --force \
  --skip-search \
  --output-dir index_dump

# run profiler
# profiler will load the index from `index_dump` directory
# starts benchmark
# searches 100 queries and generate flamegraph
# ends benchmark
cargo run -p cli_dynamic_learned_index --example profiler
```

The flamegraph is stored in `flamegraph.svg` file. You can open it in a web browser.

```shell
# open the flamegraph 
google-chrome flamegraph.svg
```

### Benchmarks for individual implementations

To run benchmarks, use the following command:

```shell
cargo bench -p cli_dynamic_learned_index
```

In basic case there is just candle model benchmark. To run tch model benchmark, enable `tch` feature:

```shell
# model benchmarks are in dynamic_learned_index due to the visibility
cargo bench -p dynamic_learned_index --features tch
```
