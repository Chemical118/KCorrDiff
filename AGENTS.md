# KCorrDiff — Agent Guidelines

## This is research code, not production

K-CorrDiff is a research project (0–6h heavy-precipitation nowcasting).
The priority is fast iteration on experiments, not operational hardening.
Code and reviews must optimize for a single researcher being able to
change a config, rerun on whatever GPUs happen to be free, and get results.

## Do NOT add unnecessary constraints

The following patterns have repeatedly blocked experiments and must not be
(re)introduced. When touching code that still contains them, remove them
rather than extending them.

1. **No sha256/hash gating.** Never pin configs, source trees, artifacts,
   manifests, or model configurations to an expected hash and refuse to run
   on mismatch. Hashes are acceptable only as cache keys / content-addressed
   storage names, or as purely informational metadata that is recorded but
   never compared.
2. **No GPU-count or device pinning.** Never require a specific
   `world_size`, GPU countr, or GPU model name. Derive the world size from
   the environment (`WORLD_SIZE`, `torch.cuda.device_count()`) and accept
   any positive value. A run configured for 2 GPUs must also work on 1.
   Batch-topology mismatches (global batch ≠ world × micro × accum) are a
   warning, not an error.
3. **No compression-ratio or byte-budget hard failures.** Observed
   compression ratios and artifact sizes may be logged, never enforced.
4. **No "unauthorized access" / tamper-detection ceremony.** No symlink
   refusal, no "unsafe path" checks, no `resolve(strict=True)` identity
   comparisons, no O_EXCL/hard-link atomic-publish rituals, no re-hashing
   of live source files mid-run, no launch-identity / provenance artifacts
   required to start a run.
5. **No preregistration enforcement.** Preregistration documents are
   documentation; code must not compare runtime choices against pinned
   preregistered values and refuse to proceed.
6. **No fail-closed schema pinning.** Config loaders may validate types and
   values they actually consume, but must not reject unknown keys, forbid
   optional keys from being absent, or hard-code "this value must never
   change" checks for tuning knobs (fold counts, batch grids, worker
   counts, etc.).

## What IS still worth checking

- Scientific-correctness invariants: train/test split leakage, duplicate
  sample IDs, fold assignments in range, importance weights consistent.
- Shape/dtype/units contracts on tensors — these catch real bugs.
- Not silently overwriting existing training outputs.

## Practical notes

- Tests should verify behavior and numerics. Do not write tests whose only
  purpose is to prove a refusal/gate fires.
- Keep `python -m pytest tests/ -q` green.
- Cluster/K8s conventions live in the workspace-level `AGENTS.md`
  (one directory above this repo).
