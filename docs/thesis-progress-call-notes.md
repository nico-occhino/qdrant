# Thesis progress call notes -- Giovanni Bellitto

## Suggested 10--15 minute flow

**1. Start with the problem (about 2 minutes).** A vector database retrieves nearest vectors for embeddings. Exact search scores every vector; ANN indexes reduce work. The key thesis question is whether a learned candidate router can be integrated as a real Qdrant physical index while preserving database correctness.

**2. State the architecture (about 3 minutes).** Qdrant remains authoritative: vectors, IDs, deletions, filters, snapshots, segment lifecycle and final metric scoring stay inside the DBMS. LMI trains a KMeans teacher and a small MLP from Qdrant storage, makes learned bucket postings of Qdrant `PointOffsetType`s, and returns candidates. Qdrant validates/scorers them exactly. This avoids a duplicate vector database, Python in the request path, HDF5 operational storage and a custom metric engine.

**3. Explain the implementation result (about 2 minutes).** The index is configured normally, constructed by the optimizer, persisted as `lmi_state.json`, reopened without retraining, snapshot-restored into fresh storage, rebuilt into a new immutable segment after writes, and served through native and universal read-only paths. This is integration/lifecycle evidence, not a production-readiness claim.

**4. Explain the important control (about 1 minute).** KMeans nearest-centroid assignment is exactly affine: maximise `2 c_j^T x - ||c_j||^2`. Therefore a nonlinear MLP must be compared with both direct centroid and affine routing. They behaved identically in Phase F.

**5. Give the honest baseline result (about 2 minutes).** At Recall@10 around 0.90: centroid used about 10,197 candidates; baseline MLP used 14,504, roughly 42% more. HNSW was much faster in the in-process measurement (0.16 ms versus about 3.5--4.9 ms). The MLP had 24 empty buckets. Do not hide this: it motivated the diagnosis.

**6. Give the diagnosis and improved result (about 3 minutes).** All empty classes had training examples, but the baseline was underfit: 30 epochs gave 23--24 empty buckets, 60 gave about 10, and 120 gave 1--2. At 4,096 samples/60 epochs, selected before the final test, the MLP used 9,372--9,566 candidates versus centroid's 10,231 near recall 0.90: 6.5--8.4% less work. Candidate-saving uncertainty intervals exclude zero; recall-difference intervals include zero. At high recall the advantage does not persist uniformly.

**7. End with the research question (about 1 minute).** The question is not whether this MLP replaces HNSW. It is when learned routing is justified by the joint tradeoff among recall, candidate work, query latency, build/rebuild cost, index size and update frequency.

## Numbers worth saying aloud

| Point | Number |
| --- | ---: |
| Phase F corpus | 99,780 vectors, 768 dimensions |
| Baseline MLP empty buckets | 24 of 64 |
| Baseline MLP versus centroid candidates near recall 0.90 | 14,504 versus 10,197 |
| Improved MLP versus centroid candidates near recall 0.90 | 9,372--9,566 versus 10,231 |
| HNSW p50 near recall 0.90 | 0.16 ms |
| LMI/HNSW single-build observation | 5.27 s / 48.61 s |
| LMI/HNSW persisted auxiliary index | 1.22 MB / 3.86 MB |

## Likely supervisor questions

**Why an MLP if affine routing reproduces KMeans?** It is not needed to copy KMeans. The MLP is evaluated as a potentially useful learned deformation of the partition; centroid and affine are essential controls.

**Does better teacher accuracy prove better ANN?** No. Teacher agreement diagnoses routing. Recall@10 against exact Qdrant search measures ANN quality.

**Did the MLP beat HNSW?** No. HNSW has substantially lower measured query latency in this experiment. The current evidence concerns integration and candidate-work tradeoffs.

**Is the improved MLP result statistically decisive?** Candidate savings are supported on the 100-query internal holdout; recall differences are not. The test was withheld from F.2 selection but was already part of Phase F, so it is not an independent external test.

**What should happen next?** Use a denser probe sweep, larger external query set, paired same-harness timing, a matched-update-count sample experiment, repeated builds, and then larger/different datasets.

## Practical close

The project has moved beyond a toy neural index. It now has a database-owned lifecycle and a reproducible scientific comparison. The evidence is promising at one medium-recall operating point, candid about HNSW and high-recall limits, and clear about the next measurements needed before stronger claims.
