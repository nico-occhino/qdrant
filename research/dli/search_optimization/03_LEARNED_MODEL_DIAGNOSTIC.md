# Within-member bucket ranking

The reserved policy touches all four trained members: model-scope top10=.987 but bucket coverage=.643. Global selection cannot explain the remaining .344 gap. The following statistics use gold top30 neighbors that actually belong to each member; they are not teacher-label classification accuracy.

| Population | Mean max probability | Entropy (natural-log nats) | Gold bucket top1 | Top4 | Top8 |
|---|---:|---:|---:|---:|---:|
|6M|.837|.495|.652|.859|.928|
|3M|.829|.499|.595|.854|.935|
|750K|.836|.478|.670|.879|.949|
|250K|.820|.540|.571|.810|.893|

Predictions are concentrated (uniform548-way entropy≈6.306 nats). Concentration is not demonstrated calibration to nearest-neighbor relevance. These data suggest useful multi-probe gains, but additional buckets mean additional physical work. Neither a larger training sample nor temperature scaling has been shown to improve the Pareto frontier.

The teacher centers/labels needed to separate teacher quality from neural approximation are not in the final persisted index files (router, postings, state). Do not call the above numbers classification accuracy. The fixed32,768 sample represents0.546% of6M vectors; that alone does not prove statistical insufficiency. No model was retrained. Small future experiments should save held-out teacher assignments/centers, compare gold bucket rank with teacher rank, and only then vary sample size or training. No10M reconstruction is required for that test.

Raw: continuation26/geometric-routing/model-diagnostic.json and probabilities.npz; numeric identity mapping validated against native persisted format. All100 queries are diagnostic; fitting cross-model parameters on them and then reporting their recall would leak evaluation information.
