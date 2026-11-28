# CLI in Qdrant: design note (in-segment mutable index)

**Status: DRAFT design, before code.** The author decided on 2026-11-28 (final-benchmark A5.2) on an in-segment
mutable index with an imbalance-trigger variant.
- **The algorithm.** [CLI_SHARED_SPEC.md](CLI_SHARED_SPEC.md).
- **The branch.** `research/cli-20261128`, worktree `~/work/qdrant-cli`, based on the final-benchmark head
  `ac09562a4`.
- **Isolation.** Nothing here touches the running final-benchmark jobs; they use the frozen staging
  `native-30271d364902`.

## 1. Facts this design rests on (source-checked, 2026-11-28)
- **F1. Index updates.** `VectorIndex::update_vector(id, Option<vector>)` is called on every upsert into an
  appendable segment (`segment/segment_ops.rs:71, 101, 124, 250, 372`).
  - **Who accepts it.** Plain indexes accept it. HNSW refuses it ("Cannot update HNSW index"), and so does trained
    LMI ("requires segment rebuild", `lmi_index/mod.rs:717`).
  - **What that means.** A mutable learned index is a *new index kind* behind this existing hook. No new write path
    is needed.
- **F2. Deletes never reach the vector index.** `delete_point_internal` marks the id tracker only (the
  vector-storage delete is deliberately disabled, `segment_ops.rs:650–692`).
  - **Consequences.** Search must filter dead offsets through the id tracker, as LMI already does. CLI bucket sizes
    must be live counts, taken by census (as `DliTriggerDecision` does), and empty buckets are removed at
    maintenance (spec §3).
- **F3. The segment flush persists four things, not the vector index.** In order: id mapping, vectors and payload,
  then id versions last (`segment/entry.rs:440–600`).
  - **Why this matters.** An appendable segment's plain index is rebuilt from storage on load. CLI's
    bucket assignments cannot be rebuilt from the vectors, because they depend on the routing history. CLI
    therefore needs its own flushed state.
  - **Recovery.** Persisted state is at `persisted_version`, and WAL replay re-applies operations above it.
- **F4. Existing pieces to reuse.**
  - **The router.** `MlpRouter` (Linear–ReLU–Linear), with its native forward pass and top-bucket ranking.
  - **Training.** `TrainedRouter::from_native` builds a trainable tch model from existing weights; that is the
    fine-tune start. `training::cluster` provides spherical k-means, and `train_labeled` is a fixed-label training
    seam.
  - **The imbalance rule.** `ImbalanceRetrain::POSTGRES_FROZEN`, from the uniform branch, is ported for CLI-I.
  - **Scoring.** The per-task scoring counter (`TaskScoringProfileGuard`) counts work.
- **F5. The maintenance pattern.** `DliMaintenance` (`collection_manager/dli_maintenance.rs`) shows the pattern: a
  collection-level owner with a worker, a running/stopped claim and test hooks before and after native steps.

## 2. Placement
- **The index kind.** A new `Indexes::Cli(CliConfig)` behind a `cli-research` feature (segment and collection),
  implemented by a new `CliIndex`. It is valid only on an **appendable** segment with mutable vector storage.
- **One CLI segment per local shard holds every point.** CLI is one flat index, so the collection is configured
  with a single segment (`default_segment_number = 1`).
  - **Optimizer exclusion.** Qdrant's optimizers (indexing, merge, vacuum, config mismatch) skip any segment whose
    vector index is `Cli`. That is one check in segment selection, feature-gated. Without it, an optimizer would
    rebuild the segment and destroy the CLI state.
- **Writes.** `update_vector(id, Some(v))` routes by argmax and records `assign[id] = bucket`. If `id` already had a
  bucket (an update), it is first removed from that bucket. `update_vector(id, None)` clears the assignment.
- **Search** (`VectorIndexRead::search`).
  - **Bucket choice.** Rank buckets by logit (ties by id), then take whole buckets until the live candidates reach
    the soft budget (spec §1).
  - **Scoring.** Score the live candidates through the normal raw scorer, so the existing counter measures work.
  - **Filters.** Out of scope (as for DLI): filtered requests take Qdrant's exact path.

## 3. State and persistence
- **The in-memory state:**
  - the router (version v);
  - `assign: Vec<u32>`, indexed by internal offset, with `u32::MAX` for unassigned;
  - `postings: Vec<Vec<u32>>` per bucket, derived from `assign`;
  - `meta`: H_init, |B|_last, b_max, the counters and the next bucket id.
- **On disk, in the segment's index directory:**
  - `router-<v>.bin`: immutable, written once (temp + fsync + rename) before it is used;
  - `cli-state-<g>.bin`: generation g, containing the router version v, the full `assign` table (4 bytes per
    offset, 4 MB at 1M) and `meta`, with a checksum. It is written temp + fsync + rename, then the directory is
    fsynced;
  - `CURRENT`: names g, replaced atomically last.
- **The flush hook.** A `CliIndex` flusher is added to `Segment::flusher` between step 2 (vectors and payload) and
  step 3 (id versions). It writes a new state generation that reflects all operations up to the version being
  persisted. Older generations and unused routers are removed only after `CURRENT` moves.
- **Load.** Read `CURRENT`, verify the checksum, load the router v, and rebuild `postings` from `assign`, which costs
  O(n). WAL replay then re-applies the later operations through `update_vector`. Those were routed with router v at
  the time, by §4's commit rule, so replay is deterministic.

## 4. Maintenance (spec §4) and its commit protocol
- **The owner.** A `CliMaintenance` (collection crate), modelled on `DliMaintenance`.
  - **When it runs.** In the benchmark the harness calls it after every plan batch (the thesis cadence). An
    insert-count cadence for production is out of scope and declared.
- **The four phases:**
  1. **Prepare (segment read lock).** Take a census of live bucket sizes. Evaluate the triggers (entropy, growth,
     or for CLI-I imbalance). Select the buckets (split or DEDRIFT). Copy their vectors and the replay sample. The
     lock is then released.
  2. **Train (no lock).** k-means, output-layer growth with weight copying, and fine-tuning from the current
     weights. Write `router-<v+1>.bin`.
  3. **Commit (the shard's updates mutex, then the segment write lock).**
     - **Collect.** Re-collect the members of the affected buckets, including points inserted since phase 1.
     - **Reassign.** Reassign them by argmax under router v + 1, and remove empty buckets.
     - **Swap.** Swap in the router, then rebuild `postings` for the affected buckets.
  4. **Persist (still under the updates mutex).** Force a segment flush, so the new generation, with router v + 1
     and the new `assign`, becomes `CURRENT` before any later operation is applied.
- **Why this is crash-consistent.**
  - **Maintenance is not a WAL operation.** It changes routing, not data.
  - **A crash before step 4's rename** reloads generation g (router v, old assignments) and replays the WAL with
    router v. That is exactly the state in which maintenance had not run, which is a valid CLI state; maintenance
    reruns at the next batch.
  - **A crash after the rename** reloads the post-maintenance state.
  - **No operation is ever applied under router v + 1 without v + 1 being durable first,** because phase 4 holds
    the updates mutex until `CURRENT` moves.
- **Concurrency.** Searches hold the segment read lock, so they see either the whole pre-commit state or the whole
  post-commit state. Training (phase 2) blocks nothing. The write-locked commit costs routing the reassigned points
  (at most the affected buckets) plus table updates.

## 5. Benchmark integration
- **The system.** `dli_phase3_workload` gains `DLI_P3_SYSTEM=cli`. One CLI collection replays the plan; after each
  batch the harness runs maintenance, flushes and copies the shard, as for the other arms.
- **The frontier.** A new `run_cli_frontier` searches the copied segment with the budget grid and the per-query
  scoring counter, and records rows with policy `cli_budget`. The timed path reuses `run_timed`.
  - **Analysis.** `analyze_final.py` gains the arm `CLI` → `cli_budget` and the variant `CLI-I`, by amendment.
- **The per-batch record** follows spec §7.

## 6. Gates (all before any pilot data)
- **G1 (unit).**
  - **Formulas.** Eqs. 2.1–2.4 known answers, including the audit-A4 edge (n < b_max).
  - **Output layer.** Growing copies the parent's row and bias; shrinking removes exactly the empty buckets'
    rows.
  - **Selection.** DEDRIFT selection on synthetic size vectors.
  - **Entropy** in bits on known logits.
  - **Imbalance.** The CLI-I rule is the frozen Q5 rule.
- **G2 (invariants after every maintenance).**
  - **Membership.** Every live point is in exactly one bucket, and the sizes sum to the live count.
  - **Routing.** Each reassigned point sits in its argmax bucket under the new router, and untouched points keep
    their bucket.
- **G3 (recovery).**
  - **(a)** A crash hook between commit and the `CURRENT` rename reopens to the pre-maintenance state plus the WAL.
  - **(b)** A crash after the rename reopens to the post-maintenance state.
  - **(c)** Reopening plus WAL replay gives an `assign` byte-identical to a never-crashed twin.
  - **(d)** A torn or checksum-failed generation is rejected, and the previous `CURRENT` is used.
- **G4 (concurrency).** Searches run concurrently with the commit, through a test hook as in `DliBeforeNativeHook`.
  Each returns results consistent with exactly one router version, checked against both oracles.
- **G5 (reference).** On a small fixture, the deterministic parts match `cli_ablation` run read-only: bucket count,
  split k, DEDRIFT selection given equal sizes, and the trigger decisions given equal entropies. Recall is compared
  within a tolerance only, because training differs.
- **G6 (local smoke).** Seed 20261202, W1 and W3, batches 0–2: gates G2 and G3(c) inside the run, plus monotone work.

## 6b. Staging the work (adopted from the author's study notes, 2026-11-28)
- **Stage A, the algorithm on a frozen single segment.** A pure in-memory CLI core (state machine, maintenance and
  search) over a segment's vectors. Persistence and concurrency are out of scope at this stage.
  - **What it must show.** Untouched buckets keep their assignments; split buckets are reinserted; model dimensions
    match the bucket directory; L_route is measured.
  - **Gates.** G1, G2 and G5 pass here first. Maintenance may rewrite the test segment's state wholesale. That is
    acceptable for verification, and is never reported as an incremental-maintenance cost.
- **Stage B, the in-segment lifecycle (§3–§4).** Persisted generations, the commit protocol, and optimizer
  exclusion, with gates G3 and G4.
- **The invariant both stages test.** A reader sees a complete (router, bucket directory, assignments) generation:
  all old or all new.
  - **Two failure kinds kept apart in the tests.** A mixed publication is a defect. Routing drift between the new
    router and the stored assignments of untouched buckets is CLI's intended approximation, measured by L_route.

## 7. Plan and cost (estimate)
| Weeks | Work |
|---|---|
| 1–2 | Stage A core (formulas, output-layer grow and shrink, triggers, DEDRIFT, fine-tune, search) on a frozen segment; G1, G2 and G5. Then `CliIndex` writes and the live census |
| 3 | Stage B: the persisted generations and their flusher, the commit protocol, optimizer exclusion; G3 and G4 |
| 4 | harness and frontier; G6; the b_max selection pilot (seed 20261202: {500, 1,000, 2,000} × W1–W3, 9 jobs); the CLI amendment frozen |
| 5 | primary: CLI and CLI-I × W1–W3 × seeds 20261201 and 20261205 (12 jobs) |

- **The target.** CLI primary data by mid-January 2027, within the final benchmark's timeline (§11).
- **Risks.**
  - **Host variation in fine-tuning.** The tch fine-tune is not bit-reproducible across hosts, as already seen for
    router training. Each arm runs on one node, and the cross-seed replication covers it.
  - **The commit's stall.** The write-locked commit blocks updates for the reassign time. It is measured and
    reported.
  - **Code churn.** Upstream changes on the research fork could break the feature-gated hooks.
