# Native maintenance decomposition — first geometric 10M run

Source `3393b4a3c205c5cf3f7fc5f957f9a16e3f737599`. Raw `continuation-25/geometric-10120191/{native-events.jsonl,build-events.jsonl}`; per-build CSV and SVG produced by analyze.py.

40 native publications and40 successful model builds produced31.75M destination vectors for10,120,191 inserted points (3.1373x). Every build selected32,768 training samples and used30 epochs. Native job-wall sum3540.055s; ingestion elapsed3730.779s. These are distinct wall metrics, not CPU time.

| Measured cumulative stage | Seconds |
|---|---:|
| Clustering | 220.225 |
| MLP initialization | 8.264 |
| Epoch optimizer loop | 1706.693 |
| Export and numerical validation | .797 |
| Clustering + MLP/export, outer timer | 1936.178 |
| Sampling/live enumeration and sample reads | 6.428 |
| Corpus inference and posting construction | 536.877 |
| Exact posting coverage validation | .181 |
| Model/posting persistence | 3.221 |

Nested timers overlap: do not sum outer clustering/MLP timer with its component timers. Training's outer timer is54.7% of cumulative native job-wall sum; fixed sample/epoch work explains much of the approximately constant small-job cost. It does not explain large transitions by itself. The6M publication took484.913s, with49.979s training,102.781s routing/postings and13.767s final propagation/publication. Remaining time includes copying, storage preparation, permit waiting and other native builder work; it is not a measured copying-only timer.

The actual per-build eligible count and sample count are in events. Profile events record successful stages, not a count of all training attempts after failures. CPU time, distinct copy-only timer, actual deduplicated copy-input count and temporary allocation peaks remain unmeasured. Native failure events are separately retained; no failure occurred in the completed geometric campaign.
