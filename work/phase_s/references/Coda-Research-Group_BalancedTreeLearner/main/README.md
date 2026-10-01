# BATL — Balanced Tree Learner

An implementation of **"Learning Balanced Tree Indexes for
Large-Scale Vector Retrieval"** by Li et al. (KDD '23), written from the paper
description rather than derived from the authors' code.

BATL is a learned index for approximate nearest neighbor search. It replaces the
hand-built routing structure of a k-ary tree with a transformer that predicts,
for a query vector, the path down to the leaf buckets most likely to contain its
neighbors. Buckets are kept balanced during training, so search cost stays
predictable regardless of how the data is distributed.

An index is an ensemble of independently trained trees. Training alternates
between fitting the routing model and reassigning vectors to buckets. Search
runs beam search over the ensemble, gathers candidates from the top-scoring
leaves, and reranks them by exact distance.


This is the implementation behind a bachelor thesis at the Faculty of
Informatics, Masaryk University. It is evaluated against published BLISS and LMI
results.

## Install

Python 3.10+ (3.11 is what CI and the cluster runs use). A conda environment is
the supported setup, locally and on HPC:

```bash
conda create -n batl python=3.11
conda activate batl
pip install -e ".[dev]"
```

If pip's `faiss-cpu` wheel gives trouble, install FAISS from conda-forge into the
same environment first:

```bash
conda install -c conda-forge faiss-cpu
pip install -e ".[dev]"
```

GPU environments are pinned separately: [`environment-gpu.yaml`](environment-gpu.yaml)
and [`environment-gpu-cu128.yaml`](environment-gpu-cu128.yaml) (CUDA 12.8, used for
the 100M-scale runs). [`environment.yml`](environment.yml) is the CPU environment.

Training runs on CUDA, MPS, or CPU — set `training.device` in the config, or
leave it `auto`.

## Datasets

No vector data ships with this repository. Download datasets yourself into
`experiments/data/` (gitignored) and point `dataset.path` in a config at them.

| `dataset.name` | Dataset | Source |
|---|---|---|
| `sift1m` | SIFT 128-d, 1M vectors | [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) |
| `glove100` | GloVe 100-d, angular | [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) |
| `deep1b` | Deep1B, 10M/100M subsets | [big-ann-benchmarks](https://big-ann-benchmarks.com/neurips21.html) |
| `bigann` | BIGANN (SIFT1B), 100M subset | [big-ann-benchmarks](https://big-ann-benchmarks.com/neurips21.html) |
| `spacev` | SpaceV, 100M subset | [big-ann-benchmarks](https://big-ann-benchmarks.com/neurips21.html) |
| `yandexti` | Yandex Text-to-Image, 100M subset | [Yandex Research](https://research.yandex.com/datasets/text-to-image-dataset-for-billion-scale-similarity-search) |
| `laion5b_subset` | LAION-2B-en CLIP768v2, 300k/10M/100M | [SISAP 2023 Challenge](https://sisap-challenges.github.io/2023/datasets/) |

The loader reads HDF5 (`.hdf5`/`.h5`), the BigANN binary formats (`.u8bin`,
`.i8bin`, `.fbin`, `.f32bin`, `.ibin`, `.i32bin`), `.fvecs`/`.ivecs`, and `.npy`.

## Usage

One YAML file describes one experiment: dataset, tree geometry, training
hyperparameters, and the evaluation sweep. See
[`experiments/configs/default.yaml`](experiments/configs/default.yaml) for an
annotated example, or the per-dataset configs under `experiments/configs/`.

Train an index:

```bash
python build.py experiments/configs/sift1m/sift1m_h2_paper.yaml
```

Sweep search over the trained index:

```bash
python search.py experiments/configs/sift1m/sift1m_h2_paper.yaml --num-leaves 10 20 40 80
```

Results (recall, QPS, distance computations, timings) are written to the
config's `experiment.output_dir` as JSON and CSV.

At 100M scale the trees of an ensemble are trained as separate jobs and merged
afterwards:

```bash
python merge_index.py tree_0.pkl tree_1.pkl tree_2.pkl tree_3.pkl --output ensemble.pkl
```

Both entrypoints take `--index-path` to read or write the index somewhere other
than the config default, and `--log` for verbose progress. `python build.py
--help` lists the rest.

### Analysis utilities

| Script | Purpose |
|---|---|
| [`compare_sweeps.py`](experiments/scripts/compare_sweeps.py) | Compare two search sweeps as recall-per-bucket curves |
| [`plot_curve.py`](experiments/scripts/plot_curve.py) | Render recall-vs-distance-computation CSV rows as SVG |
| [`scan_qps.py`](experiments/scripts/scan_qps.py) | Collect QPS measurements out of job logs and result JSON |
| [`estimate_runtime.py`](experiments/scripts/estimate_runtime.py) | Time a 1-cycle, 1-tree run of a config to size a job |
| [`run_faiss_hnsw_baselines.py`](experiments/scripts/run_faiss_hnsw_baselines.py) | FAISS IVFFlat and HNSW baseline curves |

## Repository layout

| Path | Purpose |
|---|---|
| `batl/` | Core library: model, tree, training, search, rerank |
| `batl/utils/` | Config parsing, data loaders, index I/O, metrics |
| `build.py`, `search.py`, `merge_index.py` | Command-line entrypoints |
| `experiments/configs/` | YAML experiment configs, grouped by dataset |
| `experiments/scripts/` | Analysis utilities and per-dataset cluster job scripts |
| `experiments/utils/` | Shared experiment helpers (e.g. LAION memmap preparation) |
| `experiments/data/` | Datasets — you create this; gitignored |
| `tests/` | pytest suite for the library and the experiment scripts |

## Running at scale

The 100M-scale experiments were run on [MetaCentrum](https://metavo.metacentrum.cz/),
the Czech national grid, under PBS Pro. The job scripts live under
`experiments/scripts/<dataset>/`: one script per tree for parallel ensemble
training, plus a merge-and-search script that combines the trees and runs the
evaluation sweep.

These scripts are records of how the thesis experiments were run, not a portable
harness. They hard-code cluster paths (`/storage/brno2/...`), a conda prefix, and
PBS resource requests, so adapt them to your own site before use.

## Development

```bash
make check     # ruff lint + pyright + pytest
make test      # pytest with coverage (gate: 85%)
make lint      # ruff check
make fmt       # ruff format
make smoke     # end-to-end build+search on synthetic data
```

The suite is 470 tests and runs in about 15 seconds — it uses synthetic vectors
throughout, so no dataset downloads are needed to run it.

## References

The method implemented here:

> Wuchao Li, Chao Feng, Defu Lian, Yuxin Xie, Haifeng Liu, Yong Ge, and Enhong
> Chen. **Learning Balanced Tree Indexes for Large-Scale Vector Retrieval.** In
> *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data
> Mining (KDD '23)*, pages 1353–1362, 2023.
> [doi:10.1145/3580305.3599406](https://doi.org/10.1145/3580305.3599406)

```bibtex
@inproceedings{li2023batl,
  author    = {Li, Wuchao and Feng, Chao and Lian, Defu and Xie, Yuxin and
               Liu, Haifeng and Ge, Yong and Chen, Enhong},
  title     = {Learning Balanced Tree Indexes for Large-Scale Vector Retrieval},
  booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining},
  series    = {KDD '23},
  pages     = {1353--1362},
  year      = {2023},
  publisher = {ACM},
  doi       = {10.1145/3580305.3599406}
}
```

The learned indexes it is evaluated against:

- Gaurav Gupta, Tharun Medini, Anshumali Shrivastava, and Alexander J. Smola.
  **BLISS: A Billion Scale Index Using Iterative Re-partitioning.** KDD '22,
  pages 486–495. [doi:10.1145/3534678.3539414](https://doi.org/10.1145/3534678.3539414)
- Matej Antol, Jaroslav Oľha, Terézia Slanináková, and Vlastislav Dohnal.
  **Learned Metric Index: Proposition of Learned Indexing for Unstructured
  Data.** *Information Systems* 100:101774, 2021.
  [doi:10.1016/j.is.2021.101774](https://doi.org/10.1016/j.is.2021.101774)
- David Procházka, Terézia Slanináková, Jozef Čerňanský, Jaroslav Oľha, Matej
  Antol, and Vlastislav Dohnal. **Scaling Learned Metric Index to 100M
  Datasets.** SISAP 2024, LNCS 15268, pages 266–273.
  [doi:10.1007/978-3-031-75823-2_22](https://doi.org/10.1007/978-3-031-75823-2_22)

## Contact

Questions or problems with the code: [536343@muni.cz](mailto:536343@muni.cz).
