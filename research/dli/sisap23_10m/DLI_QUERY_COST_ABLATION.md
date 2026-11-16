# Query cost observations — ablation incomplete

The geometric runs use100 ordered2023 public queries, fixed budget2000,548 buckets, hidden512, and single-system processes. No warmup, validation/test tuning or frozen-topology routing-policy comparison has been executed. These are observed operating points, not recall-matched latency results.

| Population | Recall10/30 | p50/p95 ms | Vector evaluations | Learned/fallback |
|---|---|---|---:|---|
| 1M | .552/.5317 | 262.44/662.99 | 304124 | 100/0 |
| 3M | .595/.5703 | 1324.23/4742.18 | 890418 | 100/0 |
| 10,120,191 | .402/.391 | 3116.07/8778.63 | 1758567 | 100/0 |

At10M,337.428s end-to-end query time includes324.129s in the native scorer phase and12.552s candidate resolution; the remaining~.746s includes inference, planning, locks, aggregation and request overhead. The scorer phase includes physical vector access; it is not CPU-only arithmetic. Thus these measurements do not support multiple neural routers as the main cause of this run's high latency. Distinguishing storage access from arithmetic requires further profiling.

Correction to an intermediate commentary: the existing six-element accessor returns learned, fallback, selected postings, unique offsets, **scoring nanoseconds**, **resolution nanoseconds**. The fifth value was briefly misread as routing time; this report and analyze.py use the verified source order. At1M the final descriptor had two learned members, while at3M it had one, despite worse latency. This further limits a router-count-only explanation.

Scorer counters include Plain full-scan and iterator paths, attempts versus successful evaluations, and repeated work. Raw per-query JSON records latency, vector evaluations, scorer calls and hit count. The10M average is17,585.67 scored vectors per query versus requested budget2000, reflecting whole-bucket overshoot. No frozen-topology routing ablation, budget sweep, read-lock/writer-wait split, warm-cache repeat or MUNI query comparison is yet measured.
