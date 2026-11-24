---
name: metacentrum-pbs
description: Prepare or audit a bounded MetaCentrum PBS run when explicitly invoked.
---

Use the target branch's previous accepted preflight, PBS wrapper and failure records. Check Kerberos ticket validity with klist -s and an existing SSH master with ssh -O check <configured-alias>. Use the user's established alias and private control socket outside the repo. Keep ticket material out of logs.
Pin source bundle, offline dependencies, runtime and input checksums. Verify the transferred bytes and offline dependency resolution. Estimate CPUs, walltime, memory and SSD scratch from measured accounting, staging bytes, peak temporary files and output reserve. Bound workers and BLAS threads to the allocation.
Use the login node only for transfer, checks and scheduling. Stage to the allocated SSD scratch directory on a compute node. Run the smallest representative smoke test before scaling. Reuse validated artifacts only when identities and experiment boundaries match.
Create a new job-specific durable output directory exclusively. Preserve manifest, resource samples, stdout/stderr, validator result, hashes and exit status on success or failure. Check the existing wrapper's trap and copy-back behavior. Treat an absent completion record as incomplete.
Prepare or inspect by default. Submit only with explicit authorization for this job; check the returned job ID, resources, dependencies and durable handoff before reporting submission. Cancellation needs its own explicit authorization. Use bounded status checks; compare PBS CPU time, walltime and peak memory with requested resources and validate science gates independently of exit code.
