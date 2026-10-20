# Publication failure and manual recovery

Affected commit: 69216d5ada898feef7066c72da0514aeae2ead89.
Repair branch: thesis/sisap23-publication-repair, based exactly on that commit.
No cluster access, submission or artifact promotion was performed here.

## Evidence and execution flow

**Verified in source:** construct() validates inputs, inventories/copies/checks scratch
data, runs the scratch Rust fixture, hashes/copies/checks persistent staging, runs the
persistent Rust fixture, compares vector fingerprints, rechecks the original source,
computes the final persistent inventory, and writes generation.json.
publish_new() then fsyncs all staged files/directories before
renameat2(RENAME_NOREPLACE). Finally, the source is hashed yet again, even on failure.

**Reported by the operator, not independently inspected:** job
24808764.pbs-m1.metacentrum.cz ran on haldan13 with 16 CPUs/96 GB for 12h51m;
CPU time 28m36s; exit 1; traceback identifies renameat2 errno 22.
Scratch verification took 7m39s; persistent verification took 9h45m54s with
about 196 CPU seconds. Staging and approximately 6 MB of generation metadata survive.

**Inference:** unsupported RENAME_NOREPLACE is the leading explanation. Linux
documents EINVAL for unsupported filesystem rename flags. The literal flag used
here is valid (1). Mount type, capability, exact traceback and both path states still
need inspection. An NFS syscall failure does not universally prove no server-side
change. Sources: [rename(2)](https://man7.org/linux/man-pages/man2/rename.2.html),
[link(2)](https://man7.org/linux/man-pages/man2/link.2.html).

The persistent fixture audits every vector twice and performs two normal collection
restarts against NFS. Segment loading, mmap faults and storage metadata reads can
wait on remote I/O. The low CPU/wall ratio supports waiting as an explanation, but
does not identify NFS, fsync, locks, cache misses or congestion individually.
Scratch and persistent fixtures perform different work: their elapsed-time ratio
is not a filesystem benchmark.

## Publication protocol

The repair publishes a **small descriptor**, not the collection directory:

1. Review saved/embedded scratch and persistent reports, zero native exit codes,
   producer/job, point/configuration counts, full corpus audit, restart fingerprints,
   source inventory hash and metadata structure.
2. Existing-artifact promotion requires an explicit review token, publisher commit,
   compute allocation and confirmation that the old traceback occurred **after
   sync_tree completed**.
3. Probe small filesystem primitives before large I/O. Compute one complete read-only
   SHA-256 inventory of the retained collection and compare it with generation.json.
   Recheck the report token after that pass.
4. Reserve a new destination with mkdir, which fails if it exists. Copy/fsync only
   generation.json there. Write/fsync a complete prepared publication record and sync
   the new directory/parent.
5. Atomically create publication.json using a hard link that refuses an existing
   destination. Sync the directory/parent again. No rename fallback exists.
   Readers require this committed record and matching metadata hashes before resolving
   its pinned .staging/<name>/collection.

Staging is never moved, deleted, chmodded or rewritten. **Never garbage-collect a
referenced .staging directory.** Producer SHA stays unchanged; publisher SHA is separate.
No shared vector-file hard links or symlink directory aliases are created.

A partial new destination is rejected. Collision, link failure or lost acknowledgement
never triggers overwrite or automatic retry. Preserve and inspect partial destinations.
If publication.json exists despite an error, publication may have reached its visibility
point; confirm state/durability before reuse. Future consumers rehash the collection.
The protocol assumes source/staging are quiescent; it cannot freeze files against
another privileged writer.

Hard links and directory fsync must pass the target probe. Failure means stop, not
permission to use ordinary rename/replace.

## Read-only inspection now

These commands need no new checkout and read only small reports and filesystem metadata:

~~~bash
ROOT=/storage/budejovice1/home/nico/lmi-metacentrum/results/sisap23-independent-recovery-20261019-01
STAGE="$ROOT/.staging/recover-lmi-24808764.pbs-m1.metacentrum.cz-f46b380a2475"
FINAL="$ROOT/recover-lmi-24808764.pbs-m1.metacentrum.cz-f46b380a2475"
qstat -fx 24808764.pbs-m1.metacentrum.cz
findmnt -T "$ROOT" -o TARGET,SOURCE,FSTYPE,OPTIONS
ls -ld "$STAGE" "$FINAL"
ls -l "$STAGE"/{generation,scratch,persistent,scratch.exit,persistent.exit}.json
cat "$STAGE/scratch.exit.json" "$STAGE/persistent.exit.json"
cat "$STAGE/scratch.json" "$STAGE/persistent.json"
cat "$STAGE/scratch.resources.txt" "$STAGE/persistent.resources.txt"
tail -n 45 "$ROOT/job-logs/recover-24808764.pbs-m1.metacentrum.cz.log"
~~~

A missing FINAL is expected after the reported error; if it exists, stop and inspect.
Scheduler history may have expired. Do not run du, recursive hashes or Qdrant startup
on the frontend. Provide these outputs for review before publication. Confirm both
native exits are zero and the stack failed at renameat2, not inside sync_tree.

## Deploy only after manual approval of the patch

GitHub changes are not automatically deployed to HPC. Use a separate checkout:

~~~bash
WORKSPACE=/storage/budejovice1/home/nico/lmi-metacentrum
REPAIR="$WORKSPACE/qdrant-sisap23-publication-repair"
test ! -e "$REPAIR"
git clone --single-branch --branch thesis/sisap23-publication-repair https://github.com/nico-occhino/qdrant.git "$REPAIR"
REVIEWED=$(git -C "$REPAIR" rev-parse HEAD)
printf 'Compare with delivered repair SHA: %s\n' "$REVIEWED"
PY="$WORKSPACE/envs/py312-torch251/bin/python"
TOOL="$REPAIR/tools/lmi/sisap23_matched/generation_publication.py"
ROOT=$(realpath -e "$ROOT")
STAGE=$(realpath -e "$STAGE")
FINAL="$ROOT/recover-lmi-24808764.pbs-m1.metacentrum.cz-f46b380a2475"
~~~

Inspection remains read-only, including no Python bytecode writes:

~~~bash
"$PY" -B "$TOOL" inspect --staging "$STAGE" \
  --expected-producer 69216d5ada898feef7066c72da0514aeae2ead89 \
  --expected-job 24808764.pbs-m1.metacentrum.cz
~~~

PASS here means **metadata only**. Review the displayed token and reports; this does
not prove current collection bytes still match. Require all 10,120,191 points, zero
mapping/vector errors, Float16/cosine/768, two restarts, complete corpus comparison
and equal fingerprints. Check original root, source path and producer/job provenance.

## Tiny filesystem test: explicit mutation approval

This creates one unique small sisap23-fs-probe-* directory directly under ROOT,
touches no collection and retains its files for inspection:

~~~bash
"$PY" -B "$TOOL" capability --parent "$ROOT" --allow-create-probe
~~~

Require descriptor_primitives=passed (file sync, collision-safe hard link, directory
sync). Separately reported renameat2 errno 22 supports the proposed root cause.

## Existing-generation publication: wait for report review

Only after the reports/capability output have been reviewed and you explicitly approve,
use an existing compute allocation for the single full hash pass. If none exists,
request a small allocation manually under site policy; this patch submits nothing.
This operation needs only Python: no Rust compilation, Torch, WAL replay or training.
Its cost is one sequential file inventory plus metadata latency; no runtime guarantee.

Set the variables above again on that compute node and run:

~~~bash
test -n "$PBS_JOBID"
APPROVED_TOKEN='paste the exact reviewed token'
"$PY" -B "$TOOL" publish --staging "$STAGE" --final "$FINAL" \
  --expected-producer 69216d5ada898feef7066c72da0514aeae2ead89 \
  --expected-job 24808764.pbs-m1.metacentrum.cz \
  --approve-publication "$APPROVED_TOKEN" --publisher-commit "$REVIEWED" \
  --confirm-prior-sync-completed
~~~

The prior-sync flag requires the reviewed traceback and exact affected source.
Without that evidence, stop: this utility does not claim new durability without it.
Any changed report, payload, missing/extra file rejects publication.
Keep staging on every outcome. Never delete/replace a partial destination to retry.
A new destination name requires a fresh explicit decision after inspection.

PASS: a publication JSON event with status=complete; FINAL/publication.json pins the
producer and metadata hash; STAGE remains present. generation.json accepted=true
alone is insufficient.

For a later independently approved build with patched scripts, set:

~~~bash
export QDRANT_DIR="$REPAIR"
export SISAP23_REVIEWED_COMMIT="$REVIEWED"
export SISAP23_DATA_GENERATION="$FINAL"
export SISAP23_DATA_PRODUCER_COMMIT=69216d5ada898feef7066c72da0514aeae2ead89
~~~

Include the producer variable in manually prepared PBS -v lists. This explicitly pins
old data provenance instead of relabeling it as rebuilt. Old deployed code cannot
consume descriptors. This procedure does **not** authorize resubmitting stage 50.

## I/O audit and deliberately limited changes

| Affected operation | Decision |
|---|---|
| Prepared-input hash, source/corpus copies and copy hashes | Retain necessary input/transfer checks. |
| Source inventories before copy, after copy, before publication, finally | Remove only the redundant final scan. Success already checks the source; failure cannot accept data. A finally failure must not hide the publication exception. Source is only read. |
| Build checked_generation followed by immediate source inventory | Reuse the just-validated inventory; retain after-copy/pre-publication checks. |
| Persistent hash before Rust and after Rust | Retain: normal restart can legitimately change metadata, so these are distinct states. |
| Persistent Rust audits/restarts | Retain for new recovery. Skip for existing artifact only when exact prior proofs and all current bytes match. |
| Full staging sync | Retain for new builds. Existing promotion requires proof that the old barrier completed; no repeated fsync of the old collection. |
| Scratch fsync before fadvise | Retain for freshly copied pages before cache eviction advice. |
| New publication | Copy about 6 MB metadata plus a tiny marker, not the 10M data. |

JSON phases now report UTC, elapsed seconds and inventory file/byte counts, plus
capability, copy, compile/native, sync and publication status. PBS logs start before
environment setup and refuse overwriting a job log. Failure retains staging/scratch;
PBS may still remove scratch after exit, so persistent reports/logs remain essential.
Optional SISAP23_CLEAN_OWN_SCRATCH=1 deletes only this attempt's uniquely created direct
child of scratch after successful publication. It never cleans the full allocation
or persistent staging. No automatic scratch cleanup runs on failure.

## Local verification and limitations

~~~bash
python3 -m unittest discover -s tools/lmi/sisap23_matched -p 'test_*.py'
for f in tools/lmi/sisap23_matched/pbs/*recovery*; do bash -n "$f" || exit; done
git diff --check
~~~

Tests use tiny synthetic files: report proof, collisions, interruptions, partial
records, unsupported rename/link, NFS acknowledgement ambiguity, descriptor resolution,
producer pinning and cleanup containment. No production Rust/LMI/scoring code changes.
Target NFS capability and the actual staging reports/data remain unverified until
the operator supplies inspection evidence.

Local result: **40 Python tests passed**, including 22 publication tests and the
existing recovery/preservation/reference tests. All recovery shell/PBS files passed
bash syntax checks; git diff --check passed. Rust was not rebuilt because its files
are unchanged. This does not verify the MetaCentrum mount or surviving 10M files.
