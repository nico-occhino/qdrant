# Required packages
* faiss-cpu
* torch
* tqdm
* numpy
* pandas
* h5py
* matplotlib (optional for `plot.py`)

# Data
Must be placed in data folder inside project root.

Datasets:
* LAION2B - https://sisap-challenges.github.io/2024/datasets/#data
* agnews-mxbai - https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/agnews-mxbai-1024-euclidean.hdf5

# How to run
Install required packages either locally or create a Python env.

The main script is `eval.py`.

Run using:
```
python eval.py 
    --dataset laion2B
    --dataset-filename laion2B-en-clip768v2-n=300K.h5
    --queries-filename public-queries-2024-laion2B-en-clip768v2-n=10k.h5
    --ground-truth gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5
    --distance-metric inner-product
```

Other optional arguments:
* `--k-neighbors`
* `--n-probe`
* `--init-after-samples`
* `--insert-chunk-size`
* `--replay-memory-size`
* `--split-after-inserts`
* `--result-csv-file`

Optionally visualize the results by running `python plot.py`
