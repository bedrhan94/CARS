# CARS: research artifact for reproduction

This repository contains the code, frozen inputs, run records and analyses of the CARS study. CARS compares two ways of certifying solutions of sequences of sparse symmetric positive definite systems from three-dimensional finite element discretizations, at equal rigorous guarantees and equal certified accuracy:
- **certificate reuse:** an eigenvalue bound transferred from an anchor matrix, with the stored matrix bound bit for bit to its element model;
- **per-system verification:** Rump's criterion with a sparse Cholesky factorization (Intel oneMKL PARDISO).

It is meant for reviewers and readers who want to check or repeat the experiments. The manuscript and supplement are not part of this repository.

## Quick check

Windows, CPython 3.10, from the repository root:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r environment\requirements-lock.txt
set PYTHONDONTWRITEBYTECODE=1
.venv\Scripts\python reproduce.py
```

`reproduce.py` takes about seven minutes on the original machine. It prints `ALL_OK True` and writes `reproduce_result.json`. Some checks rewrite their own result files with new run times; the comparisons ignore timing fields, so `git status` will show these files as modified. Its steps are:

| Step | What is checked |
|---|---|
| 0 | every file listed in `MANIFEST_SHA256.txt` has its recorded SHA-256 |
| 1 | the decision rule of the pre-specified analysis, on synthetic cases |
| 2 | the accurate residual enclosure, against 2 000 exact rational cases; deliberately wrong variants are caught |
| 3 | every frozen held-out matrix, right-hand side and coefficient vector rebuilt from the exact hex inputs |
| 4 | replay of every shipped primary-arm record (CHOLMOD re-proof of the eigenvalue bound, recomputation of the bounds) |
| 5 | the frozen confirmatory analysis, re-run on all records and compared with `FINAL_DEV/phase10b/analysis10b.json` |
| 6 | an independent recomputation of the primary statistic |
| 7 | the analysis of every supplementary and post hoc experiment, re-run and compared with the shipped result |

## Contents

| Path | Content |
|---|---|
| `PHASE3B_DEV/`, `PHASE5_DEV/`, `PHASE6_DEV/`, `PHASE8A_DEV/`, `reference_impl/` | frozen matrix generator, reference assembly and transfer, runner, binding and independent-arithmetic modules |
| `PHASE10A_DEV/` | the seven compared methods (arms), the verifier, tamper and negative-control tools, the accurate residual enclosure and its exact test suite, development analyses |
| `PHASE4_HELDOUT/` | results of an earlier held-out study (seeds 100–104) |
| `reports/PHASE10B/` | the pre-specified analysis plan, its pre-data amendments, the code freeze (67 files), the environment record and the freeze record |
| `environment/` | `requirements-lock.txt` and the environment record |
| `FINAL_DEV/phase10b/` | confirmatory experiment: input generation, frozen held-out inputs (exact hex coefficients and right-hand sides, SHA-256 manifest), cell harness, all run records, independent replay, independent checks (exact residuals, Arb proofs), frozen analysis |
| `FINAL_DEV/stage0/`, `FINAL_DEV/phase10b_post/` | pre-freeze checks; symbolic-analysis diagnostic, budget and incident record, Arb attempt registry |
| `FINAL_DEV/simp/` | topology-optimization (SIMP) sequences: protocol, generator, frozen windows, runs, analysis |
| `FINAL_DEV/crossplatform/` | Linux (WSL2) software-stack check on the same hardware |
| `FINAL_DEV/release_tools/` | frozen-input loader, input check, subset replay, frozen-cell runner |
| `FINAL_DEV/audit/` | independent recomputation of the primary statistic |
| `FINAL_DEV/REVISION1/structural/` | post hoc: structural sparsity envelope for the symbolic analysis |
| `FINAL_DEV/REVISION2/compiled/` | post hoc: compiled binding and accurate residual enclosure, with an equivalence test |
| `FINAL_DEV/REVISION2/fallback/` | post hoc: corrected carried-shift fallback (shift-invert) and completion of the SIMP evaluation |
| `FINAL_DEV/REVISION2/checks/` | post hoc audit checks: Rump-constant share, absolute values of the exact suite, exact sample of accurate-bound records, coefficient-based negative control |
| `FINAL_DEV/REVISION3/fullc/` | post hoc: all compilable per-step operations compiled, alone and with the structural envelope; CG-loop overhead estimate |
| `FINAL_DEV/REVISION3/simpc/` | post hoc: SIMP sequences with compiled operations on both sides |
| `FINAL_DEV/REVISION3/large/` | post hoc, descriptive: one comparison at n = 104 544 on development data |
| `FINAL_DEV/REVISION2/RUN_NOTES_REV2.md`, `FINAL_DEV/REVISION3/RUN_NOTES_REV3.md` | execution records of the post hoc experiments |
| `MANIFEST_SHA256.txt` | SHA-256 of every file |

Each post hoc experiment has a protocol (`*_PROTOCOL.md`) and a freeze record (`PROTOCOL_FREEZE.json`) with the SHA-256 of its protocol and code, written before its timed runs.

## Re-running experiments

Timings depend on the machine. The study used one Intel Core i7-14700K under Windows 11, single-threaded on one performance core unless stated otherwise.

| Purpose | Command (repository root unless stated) |
|---|---|
| one confirmatory cell | `python FINAL_DEV/phase10b/run_cell10b.py --family E3-TS --size S3 --seed 200 --repeat 0 --mode ST --out <dir>`, then `python FINAL_DEV/phase10b/replay10b.py <dir>` |
| one SIMP cell (corrected fallback) | `cd FINAL_DEV/REVISION2/fallback`, then `python run_cell_simp_fb.py --frozen ../../simp/eval/pass_a --size S3 --inst 301 --window early --repeat 0 --out <dir>` |
| one full-compilation cell | `cd FINAL_DEV/REVISION3/fullc`, then `python run_cell_fullc.py --family E3-TS --size S3 --seed 200 --repeat 0 --out <dir>` |
| one SIMP-compiled cell | `cd FINAL_DEV/REVISION3/simpc`, then `python run_cell_simpc.py --frozen ../../simp/eval/pass_a --size S3 --inst 301 --window early --repeat 0 --out <dir>` |
| one large-size cell (about 12 min, 3.3 GiB) | `cd FINAL_DEV/REVISION3/large`, then `python run_cell_large.py --seed 0 --out <dir>`; then `python FINAL_DEV/phase10b/replay10b.py <dir>` (40–45 min) |
| kernel equivalence tests | `FINAL_DEV/REVISION2/compiled/test_kernels.py`, `FINAL_DEV/REVISION3/fullc/test_shift.py` |

Replay a SIMP cell with `python FINAL_DEV/simp/replay_simp.py FINAL_DEV/simp/eval/pass_a S3 <dir>`.

**Compiled experiments.** These need the same environment plus `numba==0.61.2` and `llvmlite==0.44.0`.

**Frozen code.** Re-running confirmatory cells requires the unchanged frozen code. The harness checks it against `reports/PHASE10B/PREREG_FREEZE_RECORD.json` and refuses any change.

## Limits

- **Solution vectors.** They are omitted (several GB), except those of the two primary arms in the single-thread S3 repeat-0 cells. Every record carries the SHA-256 of its solution vector, so the full replays and the kernel equivalence tests cannot be repeated from this repository alone. Their results are included.
- **Bitwise identity.** Certificates are bit-reproducible on the original platform in single-thread mode. Elsewhere the floating-point results of PARDISO may differ, and certificates must be re-verified rather than compared bit for bit.
- **Closed-source engine.** PARDISO is closed source. Its conformity with the rounding model of Rump's criterion is an assumption. It is cross-checked by CHOLMOD re-proofs and an Arb ball-arithmetic sample.
- **Third-party software** is installed from PyPI and not redistributed here; see `THIRD_PARTY_NOTICES.md`.

## Licence

No licence has been chosen yet. The material is provided for review and reproduction.
