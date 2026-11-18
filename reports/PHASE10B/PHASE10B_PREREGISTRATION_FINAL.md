# Phase 10B / E1: final pre-registration (frozen 15 Nov 2026, before any held-out input exists)

**Basis.**
- The Phase 10A draft (`reports/PHASE10A/PHASE10B_PREREGISTRATION_DRAFT.md`, kept unchanged).
- The pre-data amendments (`PHASE10B_PREDATA_AMENDMENTS.md`, A1–A11).

This document consolidates both. Where wording differs, this document and the amendments govern.

**Freeze.**
- SHA-256 of this file, the amendments, the code freeze, the environment record and the original draft are written to `PREREG_FREEZE_RECORD.json` before generation.
- Code aggregate: `f305dda539b6a51e42de5fd7862265a5233120dc9fa91eb742e3b76873924977` (`PHASE10B_CODE_FREEZE.json`).
- Every held-out process re-checks all frozen hashes at start and refuses to run on any change.

## 1. Question

On fresh held-out seeds, at equal certified accuracy, is gray-box certificate reuse (M2-A with the shared rigorous residual strategy) faster than an optimized per-system Rump/PARDISO verifier?

This is a **confirmatory** test. The development finding (Phases 5, 6, 10A) is that the per-system verifier is faster by roughly 10–20 %.

## 2. Methods

All seven frozen Phase 10A arms run in every cell (amendments A1, A2): A, B1, B, B9 (per-system); C, D, D2 (reuse).

**Primary pairs:**

| Family | Per-system P | Reuse Q |
|---|---|---|
| E3-TS | B9 | D |
| D3-TS | B | D2 |

**Common settings:**
- target τ = 1·10⁻⁸;
- PARDISO, mtype 2, with symbolic reuse;
- frozen PCG stopping rule;
- the shared frozen-first residual cascade (frozen bound, then the rigorous EFT bound) for the new arms.

## 3. Data

- **Families:** E3-TS and D3-TS, split `heldout` (variant parameters in A5).
- **Seeds:** 200, 201, 202, 203, 204; t = 0…20.
- **Sizes:** S3 (primary), S2 (secondary).
- **Inputs:** generated twice in separate processes, compared bit for bit, and frozen (exact hex coefficients and rhs, SHA-256 of every matrix, rhs and coefficient vector) **before** any benchmark (A7).
- **Never used:** seeds 100–104 (consumed in Phase 4) and S4.

## 4. Execution

**Machine.** i7-14700K (8 P + 12 E cores, 28 logical CPUs), 31.7 GB RAM, Windows 11 (build 26300), balanced power scheme. Python 3.10.0, NumPy 2.2.6, SciPy 1.15.3, MKL 2026.2.1 (PARDISO), python-flint 0.9.0, cholespy 2.2.0 (`PHASE10B_ENVIRONMENT.json`).

**Modes:**
- **ST (primary):** one thread (MKL/PARDISO, OpenMP, OpenBLAS), pinned to logical CPU 2 (a P-core), HIGH priority.
- **MT (secondary):** eight threads, pinned to logical CPUs 0, 2, …, 14 (one per P-core).

**Cells.**
- One fresh process per (family, size, seed, repeat, mode). All seven arms run in that process.
- Arm order is rotated by (repeat + seed) mod 7, after a symmetric untimed warm-up (every arm on development E3-TS S0 seed 0).
- 3 repeats.
- Every input is hash-checked against the frozen manifest before the warm-up.

**Stage order:** ST_S3, ST_S2, MT_S3, MT_S2. Each cell is replayed immediately (A9).

**Metric.** Measured certified wall-clock cost per 21-step sequence: the sum of all timed components, excluding artifact I/O (historical rule).
- Components: λ estimation, shift and Rump constant, factorization including PARDISO symbolic analysis, preconditioner set-up, structure set-up, binding, transfer, PCG solve, residual bound, other.
- End-to-end cost including I/O is reported as a sensitivity.

**Resources.**
- 6 h for the campaign (cells, replays and independent checks); 30 min per process; 16 GiB peak working set per process.
- Resource stops are reported as such.

**Hard stops:**
- an input mismatch;
- a second failure of a crashed cell (one repetition allowed, reported);
- any REJECTED or UNAVAILABLE replay record;
- a manifest mismatch;
- a failed exact residual check;
- a false λ claim.

## 5. Primary statistic

- **Unit:** (family, seed); 5 units per family. Repeats are technical replicates (A4).
- **Within-process ratio:** for each (family, seed, repeat), R = cost(P) / cost(Q). R > 1 favours reuse; R < 1 favours per-system verification.
- **Per unit:** the median of R over the 3 repeats.
- **Per family:** the geometric mean GM_f over the 5 units, with a two-sided 95 % t-interval on log R (df = 4).
- **Reported for every family:** the per-seed values, the repeat CV, the 90 % interval (equivalence descriptor) and the component breakdown.

## 6. Accuracy parity (applied first)

- A run (one arm, one cell) is **parity-valid** iff all 21 records are CERTIFIED, target-met (ε ≤ τ) and VERIFIED on replay.
- Speed is decisive for a family only if both primary arms are parity-valid in all 15 runs of that family (ST, S3).
- A per-system parity failure makes the family **INVALID**.
- A reuse parity failure makes it **REUSE ACCURACY-INFERIOR**; its speed is then descriptive only.

## 7. Decision rule (directional thresholds unchanged; categories per A3)

| Per-family category | Condition |
|---|---|
| REUSE ADVANTAGE | GM_f ≥ 1.2 **and** lower 95 % bound > 1 |
| PER-SYSTEM ADVANTAGE | GM_f ≤ 1/1.2 (≈ 0.833) **and** upper 95 % bound < 1 |
| NO ESTABLISHED PRACTICAL ADVANTAGE | parity holds, and neither condition above is met (this is **not** evidence of equivalence) |
| REUSE ACCURACY-INFERIOR / INVALID | §6, or any hard stop |

**Joint outcome:**
- both families in the same category among REUSE ADVANTAGE / PER-SYSTEM ADVANTAGE / NO ESTABLISHED PRACTICAL ADVANTAGE: that category;
- any INVALID: **INVALID**;
- otherwise: **MIXED**.

**Secondary descriptors (not decisive):**
- *directional*: the 95 % interval lies below or above 1;
- *practically equivalent within ±20 %*: the 90 % interval lies inside (1/1.2, 1.2), i.e. TOST at α = 0.05 per side.

**Permitted claims** are fixed in A3.5.

**No re-analysis** with other thresholds, units, arms, sizes or modes is allowed. Any deviation is reported as a deviation.

## 8. Validation (A8)

- Full replay of every record.
- 84 exact rational residuals (all seven arms; S2 and S3; ST and MT).
- Arb λ proofs:
  - mandatory at S2: per family, (seed 201, t = 0) and (seed 203, t = 20);
  - optional at S3: per family, (seed 202, t = 10), capped at 30 min.
- Determinism of certificates across repeats within each mode.

## 9. Secondary analyses (A10)

- MT and S2 statistics, all arms, all 12 pair ratios, counterfactual B, end-to-end with I/O, components, iterations, memory.
- Post-hoc best-versus-best ratios, labelled as such.
- A same-machine second ST S3 session (one repeat, index 3) after the campaign, as repeatability evidence only.

## 10. What must not happen

- Code changes after the freeze.
- Tuning of any arm after generation.
- Re-running cells to obtain a different result.
- Deleting or excluding outliers.
- Pooling development and held-out data.
- S4.
- Reuse of seeds 100–104 as untouched evidence.
