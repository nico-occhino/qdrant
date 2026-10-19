# SISAP23 10M: memory admission and independent recovery

## Scope and diagnosis

Base: `59f9550854db66d5c4ed5e5464a8c9132c07b6ce`.
New branch: `thesis/sisap23-lmi-memory-recovery`.
Incident: PBS `24723946.pbs-m1.metacentrum.cz`, reported exit 101.
Original root: `/storage/budejovice1/home/nico/lmi-metacentrum/results/sisap23-matched-recovered-20261016-01`.

**Verified in source:** the old experimental LMI guard used cgroup limit minus
total usage. Kernel cgroup usage includes filesystem cache. Consequently the
guard could reject a build despite reclaimable cache. The fixture ignored
`OptimizersStatus::Error` and waited up to 172800 seconds while the optimizer retried.

**Not established remotely:** the reported 490,192,896 available bytes alone does
not establish cache as the incident's cause. No incident memory.stat, allocation
limit, memory peak, or scheduler resource history has been inspected here. Genuine
memory pressure, parent limits and other processes remain possible.

The correction is scoped to experimental LMI build admission. Generic Qdrant memory
reporting and the fixed training/search parameters are unchanged:

- Bound by host MemAvailable and every visible ancestor cgroup hard limit.
  Treat memory.high as an additional conservative ceiling.
- Credit **50% of clean inactive file pages**, bounded by file usage and total usage,
  subtracting dirty, writeback, shared and unevictable counts. No anonymous memory,
  active file cache, swap or slab credit. Missing/malformed statistics give no credit.
  Unreadable limits/usage fail closed. v1 uses hierarchical total counters.
- Keep the existing estimated allocation budget plus **4 GiB reserve**:
  the specified 10M plan requires 6,538,220,945 bytes. This estimate is not a
  measured trainer peak and is not a reservation against concurrent allocations.
- Log host/cgroup limits, usage, available bytes, cache credit, memory.stat,
  memory.events and kernel peak where exposed. Recheck at the actual training
  allocation boundary. Admission failures have the `LMI_MEMORY_ADMISSION` prefix.
- Benchmark polling stops on any optimizer error, including when an indexed-count
  threshold was reached. No 48-hour retry loop for deterministic rejection.
  The production optimizer's retry policy is unchanged.
- Ingestion resources are closed before construction. Independent jobs use fresh
  processes and no corpus copy during index-only construction. Only our verified
  scratch copy receives fsync + POSIX_FADV_DONTNEED (best effort); no global cache
  dropping, original-data advice, memory override, or reduced sample.

Kernel references:
[Unified cgroup memory controller](https://docs.kernel.org/admin-guide/cgroup-v2.html)
and [v1 memory controller](https://docs.kernel.org/admin-guide/cgroup-v1/memory.html).

## Acceptance and data reuse

Never point a recovery operation at an original collection as its writable input.
The Python controller inventories the original, copies to newly allocated scratch,
checks both inventories, then opens **only that copy** with normal Collection::load,
optimizers paused and normal WAL replay enabled. Deferred/pending replay configuration
is rejected. The original is inventoried again before publication.

Acceptance requires all **10,120,191 live numeric IDs**, unique and in range;
forward/reverse mappings and versions; finite 768-dimensional vectors;
Float16/cosine configuration; and every live vector compared against the hashed
original corpus after native normalization/Float16 conversion (absolute tolerance
0.002, max observed discrepancy reported). Reopen twice and compare ID/vector
fingerprints. Recovered data has no learned/graph index to discard.

Normal optimizer cleanup can leave empty or entirely point-tombstoned source
segments. Every retained physical slot there must be covered by the point-deletion
mask and have no external ID; it cannot contribute search results. Nonempty
segments must have no deleted vectors/points. Full live-ID coverage remains mandatory.
Final search requires **one nonempty segment** containing the entire corpus;
empty/tombstoned appendable segments are explicitly checked, never silently searched.

Index construction opens a verified copy of accepted data, updates configuration
through native Qdrant APIs, and enables one optimizer. **No upserts occur.**
For HNSW, the inherited desired LMI config is explicitly disabled. Fixed parameters:

| Variant | Construction | Query effort |
|---|---|---|
| LMI | B=3162, sample=250000, hidden=512, epochs=30, KMeans=5, seed=42, both batches=256 | 1,4,8,16,30,40,64,100,160,256 |
| HNSW-A | m=16, ef_construct=100, index threads=1 | 16,32,64,128,256,512,1024 |
| HNSW-B | m=32, ef_construct=200, index threads=1 | same |
| Plain | existing verified exact results | full scan |

After successful construction, verify persisted graph/router parameters and coverage,
unchanged vectors and unchanged index files after normal reopen. Copy to new
persistent staging, reopen twice again, verify, fsync, then atomically publish with
Linux renameat2(RENAME_NOREPLACE). Existing generations are never replaced.
The namespace is immutable by protocol; external modifications are detected by hashes.

A directory, ingestion progress line, zero process exit, or partial marker is
insufficient. Accepted generations require `generation.json` plus all matching
file inventories and successful scratch/persistent reports.

If LMI recovery fails because data or WAL are genuinely incomplete, preserve the
failure report. Repeat **data recovery from the already verified Plain collection**
using `SISAP23_SOURCE_VARIANT=plain`. This disables inherited LMI configuration only
in an index-build copy and avoids reinsertion. If neither source can pass exhaustive
validation, a fresh isolated ingestion is necessary; there is no synthetic filling
of missing IDs, no inferred durability, and no automatic fallback.

## Timing and comparability

- Import completed Plain rows/chunks/panels byte-for-byte into a new campaign;
  independently check coverage and official recall. Original artifacts stay untouched.
- Plain evaluator provenance remains the base commit. New variants record the new
  reviewed commit and immutable index generation. The scorer/search loop is unchanged.
  This is explicitly a mixed-source campaign with preserved historical Plain timings,
  not a claim that Plain was remeasured.
- Query jobs require exactly
  `deimos11.meta.zcu.cz|AMD EPYC 9555 64-Core Processor|nfs` before and after execution.
  A different node or filesystem fails. If deimos11 is unavailable, wait or create a
  separately designed all-variant remeasurement; do not relax the fingerprint.
- Run timed variants sequentially, without concurrent benchmark/load jobs. A campaign
  lock rejects overlapping variant evaluations. This does not exclude other users'
  workloads; scheduler isolation and system load still need review.
- Keep the original query allocation (16 CPUs, 96 GB) and persistent NFS vectors,
  as in the old suite.
  HNSW graph remains Memory::Cached; LMI router/postings remain resident.
  Cached graphs can be evicted, so the policy is warm-cache comparability rather than
  identical residency guarantees. No global cache flushing. Preserve existing warmup,
  query order, 100-query checkpoints and three paired 1000-query panels.
- Native latency includes routing/traversal and scoring; LMI candidate count and HNSW
  ef are different effort measures, not interchangeable work counts.
- Compilation logs are separate. Native /usr/bin/time -v captures CPU and peak RSS
  for each executed fixture; cgroup peak includes charged filesystem cache and may
  include compilation and other processes. Neither is an isolated trainer-only peak.
  Index wall time, copy time, audit time and resource logs are separate. Reuse records
  `ingested_points=0` and `ingest_seconds=null`; 22401.5 s is preserved only as the
  historical observation, not a new comparable construction measurement.
- Build hosts may differ. Do not treat build times as a matched-host construction
  comparison without rerunning all relevant constructions on matched hardware.

## Manual Perian / MetaCentrum commands

No jobs were submitted by this patch. From your Perian terminal connect normally:

```bash
ssh nico@zenith.metacentrum.cz
```

On the frontend, first read the old job metadata and small logs (no vector scans):

```bash
qstat -fx 24723946.pbs-m1.metacentrum.cz
ORIGINAL=/storage/budejovice1/home/nico/lmi-metacentrum/results/sisap23-matched-recovered-20261016-01
cat "$ORIGINAL/suite.state"
find "$ORIGINAL/runs/lmi" -maxdepth 1 -type f -print
grep -hE 'LMI experimental sample|measured=|sisap23_ingest_points|Maximum resident|Elapsed|Exit status' "$ORIGINAL"/runs/lmi/*.log "$ORIGINAL"/runs/lmi/*.txt
cat "$ORIGINAL/runs/lmi/collection/config.json"
```

Return those outputs if the diagnosis remains uncertain. Scheduler history may have
expired; the terminated job's cgroup may no longer exist. Do not infer its old peak
from the current frontend cgroup. New probe/build logs capture the compute-node state.
Do not run recursive hashes, Rust builds or vector audits on the frontend.

Prepare an isolated clone, checking the resulting SHA against the delivered SHA:

```bash
export WORKSPACE=/storage/budejovice1/home/nico/lmi-metacentrum
export QDRANT_DIR="$WORKSPACE/qdrant-sisap23-memory-recovery"
test ! -e "$QDRANT_DIR"
git clone --single-branch --branch thesis/sisap23-lmi-memory-recovery https://github.com/nico-occhino/qdrant.git "$QDRANT_DIR"
export SISAP23_REVIEWED_COMMIT=$(git -C "$QDRANT_DIR" rev-parse HEAD)
printf 'Verify against delivered commit: %s\n' "$SISAP23_REVIEWED_COMMIT"
export SISAP23_ORIGINAL_ROOT="$WORKSPACE/results/sisap23-matched-recovered-20261016-01"
export SISAP23_RECOVERY_ROOT="$WORKSPACE/results/sisap23-independent-recovery-20261019-01"
export SISAP23_CAMPAIGN="$WORKSPACE/results/sisap23-independent-matched-20261019-01"
test ! -e "$SISAP23_RECOVERY_ROOT"
test ! -e "$SISAP23_CAMPAIGN"
PBS="$QDRANT_DIR/tools/lmi/sisap23_matched/pbs"
VARS=WORKSPACE,QDRANT_DIR,SISAP23_REVIEWED_COMMIT,SISAP23_ORIGINAL_ROOT,SISAP23_RECOVERY_ROOT,SISAP23_CAMPAIGN
PROBE=$(qsub -v "$VARS" "$PBS/49_recovery_probe.pbs")
printf '%s\n' "$PROBE"
```

The probe compiles the real collection crate with lmi-training, runs the exact fixed
plan admission check, and allocates no 10M dataset or trainer tensors. It validates
Rust 1.98.0, selected Python 3.12/PyTorch 2.5.1 CPU, g++, and protoc using the existing
common environment. Builds/TMP use PBS scratch. No version-check bypass.
If the site requires a versioned module, export PROTOBUF_MODULE and add its name to
VARS. Inspect persistent `job-logs` and `probes` before expensive construction.

Recover and initialize the preserved-results campaign; these do not depend on LMI
training succeeding:

```bash
RECOVER=$(qsub -W "depend=afterok:$PROBE" -v "$VARS" "$PBS/50_recovery_recover.pbs")
INIT=$(qsub -W "depend=afterok:$PROBE" -v "$VARS" "$PBS/53_recovery_init.pbs")
printf 'recover=%s init=%s\n' "$RECOVER" "$INIT"
```

PASS: recovery log prints ACCEPTED_GENERATION and that path contains generation.json.
Select exactly that path from the successful recovery log (do not select .staging):

```bash
grep -h '^ACCEPTED_GENERATION=' "$SISAP23_RECOVERY_ROOT"/job-logs/recover-*.log
export SISAP23_DATA_GENERATION=$(sed -n 's/^ACCEPTED_GENERATION=//p' "$SISAP23_RECOVERY_ROOT/job-logs/recover-$RECOVER.log")
test -f "$SISAP23_DATA_GENERATION/generation.json"
BUILDVARS="$VARS,SISAP23_DATA_GENERATION,SISAP23_VARIANT"
export SISAP23_VARIANT=hnsw-a
HA=$(qsub -v "$BUILDVARS" "$PBS/51_recovery_build.pbs")
export SISAP23_VARIANT=hnsw-b
HB=$(qsub -v "$BUILDVARS" "$PBS/51_recovery_build.pbs")
export SISAP23_VARIANT=lmi
LM=$(qsub -v "$BUILDVARS" "$PBS/51_recovery_build.pbs")
printf 'hnsw-a=%s hnsw-b=%s lmi=%s\n' "$HA" "$HB" "$LM"
```

These three jobs are independent. Each builds its index once from the same accepted
data. Do not launch query timings while build jobs occupy the evaluation node.
Each successful build prints its own ACCEPTED_GENERATION.

After builds and initialization succeed, select the corresponding accepted index
path and submit one evaluation at a time. This example uses HNSW-A:

```bash
export SISAP23_VARIANT=hnsw-a
export SISAP23_INDEX_GENERATION=$(sed -n 's/^ACCEPTED_GENERATION=//p' "$SISAP23_RECOVERY_ROOT/job-logs/build-$HA.log")
EVALVARS="$VARS,SISAP23_VARIANT,SISAP23_INDEX_GENERATION"
EA=$(qsub -W "depend=afterok:$INIT:$HA:$HB:$LM"   -l select=1:ncpus=16:mem=96gb:scratch_local=80gb:host=deimos11.meta.zcu.cz   -v "$EVALVARS" "$PBS/52_recovery_evaluate.pbs")
```

If LMI failed, omit LM from that dependency list after verifying its job has ended;
HNSW evaluation does not require LMI success. Submit HNSW-B afterok:EA using its own
variant/generation, then LMI afterok:EB when its build succeeds. If host selection is
rejected by PBS, ask the site for the correct host resource syntax; the runtime
fingerprint remains mandatory.

After all variant evaluations have successful completion markers:

```bash
qsub -v "$VARS" "$PBS/54_recovery_validate.pbs"
```

The final PASS is a new `validation.json` with validation=passed, full fixed-grid
coverage, both official/Plain reference recall and paired-panel metrics. There is
no valid final matched frontier while any variant is missing.

## Recovery of interrupted stages

- Recovery/build failure: retain .staging and logs. No publication occurs. Submit a
  new attempt with the same accepted input after resolving the error. Native training
  and graph construction are not checkpoint-resumable; only their input data is reused.
- Failed LMI data recovery: repeat stage 50 with
  `export SISAP23_SOURCE_VARIANT=plain` and add SISAP23_SOURCE_VARIANT to VARS.
  The original failed LMI directory is retained.
- Query failure: rerun stage 52 with the exact same campaign, commit, index generation
  and hardware. Valid completed 100-query chunks are reused; incomplete chunk temporary
  files are regenerated by the existing fixture. Full raw-result validation is mandatory
  before evaluation.complete.json. Completed Plain data is never rerun or overwritten.
- Failure during initial query-copy creation, leaving no evaluation.generation.json:
  fail closed. Preserve/rename only that incomplete **new campaign variant directory**
  after inspecting logs, then rerun. No automated deletion.
- Interrupted campaign initialization: preserve the partial new campaign and choose
  another fresh campaign name. Never repair or overwrite the original incident root.
- Commit changes require a new campaign/generation lineage; do not mix generations
  produced by different fixes under one accepted source pin.
- Duplicate same-variant jobs are not supported. Timed evaluation takes a nonblocking
  campaign lock. Use afterok dependencies to serialize it, never afterany for acceptance.

## Resource envelope and cost

PBS defaults: probe 16 CPU/32 GiB/50 GiB scratch/2 h; recovery 16/96/150/24 h;
build 16/128/200/72 h; evaluation 16/96/80/72 h; initialization 16/16/50/6 h;
validation 16/16/50/8 h. Scheduler units may be decimal GB. These are requested
budgets, not measured requirements or runtime predictions. Actual guard snapshots
remain authoritative; request more memory if admission still rejects.

One corpus/vector payload is about 15.54 GB before metadata/WAL. Allow several
hundred GB persistent free space for data generation, three index generations,
evaluation copies, retained staging and logs. Recovery performs multiple full scans,
hashes and copies; budget hours, potentially the 24-hour slot on slow NFS. Index
construction can take many hours or days; no measured 10M build duration exists here.
The fixture retains a 48-hour legitimate optimizer deadline within the 72-hour job.
Full query sweeps can also take days. Plain's completed expensive scans are retained.

## Local verification and remaining gate

Run lightweight Python and shell checks:

```bash
python3 -m unittest discover -s tools/lmi/sisap23_matched -p 'test_*.py'
for f in tools/lmi/sisap23_matched/pbs/*.sh tools/lmi/sisap23_matched/pbs/*.pbs; do bash -n "$f" || exit; done
cargo test -p segment --lib utils::mem --locked
cargo test -p segment --lib index::lmi_index::build_plan --locked
```

With the supported training environment, these small native tests need no SISAP data:

```bash
cargo test -p collection --lib --features lmi-training sisap23_ --locked
cargo test -p collection --lib --features lmi-training sisap23_reuse_vectors_without_reingestion --locked -- --ignored
cargo test -p collection --lib --features lmi-training sisap23_interrupted_final_batch_restarts_twice --locked -- --ignored
```

Local recovery/HNSW tests were compiled with only the fixture module's training
gate temporarily removed, then restored: local Python has no torch installation.
They do not prove LibTorch training compilation or actual 10M recovery. Stage 49 is
the opt-in cluster compilation/admission gate. No MetaCentrum job, remote data audit,
10M training, or new benchmark measurement was executed during this implementation.
\nLocal result: 26 focused Rust tests and 18 Python tests passed; all PBS/shell files\npassed bash syntax checks, and git diff --check passed. Expensive ignored fixtures\nwere not executed; only the small reuse and interrupted-WAL regressions were opted in.\n