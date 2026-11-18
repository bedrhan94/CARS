# Compiled-kernel experiment: exploratory protocol (Revision 2)

**Status.** Exploratory and post hoc. The owner authorised this experiment in response to the second referee report (major comment 1), after all Phase 10B results were known.
- **Seeds.** It uses the frozen Phase 10B held-out sequences, seeds 200–204. These seeds are consumed; the data are not fresh held-out data.
- **Scope.** It cannot change, replace or reclassify any registered Phase 10B result. No pre-registration document is modified.
- **Freeze.** This protocol and the code in §8 are hash-frozen in `PROTOCOL_FREEZE.json` before any timed cell runs.

## 1. Question

Two components of the frozen methods run as interpreted Python with NumPy array operations:
- the lean **binding** of the reuse arms, which checks bit for bit that the stored matrix is the element-model assembly of the coefficients;
- the **accurate residual bound** (EFT bound) of the shared residual strategy, used by per-system and reuse arms.

At E3-TS S3 (Phase 10B, seed 200, repeat 0) the binding cost 1.0 s per sequence in arm D, and the residual component cost 0.37 s (B9), 0.99 s (D), 2.23 s (D2) and 2.24 s (B), against measured totals of 7–8 s.

The question: **do the registered speed comparisons change direction or category when both components are compiled?** Specifically:
- the E3-TS S3 per-system advantage (B9/D = 0.783 [0.696, 0.881]);
- the D3-TS S3 result (B/D2 = 0.893 [0.864, 0.923], no established practical advantage);
- the S2 results.

## 2. The compiled kernels (`compiled_kernels.py`)

- **Binding (`CompiledAssembler`).** A subclass of the frozen `FastAssembler`. It uses the same memoized SciPy summation order and the same products, sums left to right, applies the same zero-dropping rule and compares exactly. The comparison runs in one compiled pass without temporary arrays.
- **Accurate residual bound.** Row by row in CSR order. It uses the same TwoProduct splitting and the same cascaded TwoSum order, with the same γ constants. Every directed-rounding step is the same nextafter operation, and the same fallback branches apply. The final sum of squares uses the frozen NumPy routine.
- **Compilation settings.** Kernels are compiled with numba 0.61.2 / llvmlite 0.44.0, without fast-math (no reassociation, no FMA contraction), and cached on disk. They are loaded during the untimed warm-up.

**Equivalence**, checked before this protocol was frozen (`test_kernels.json`, ALL_IDENTICAL = true):
- every certified record of the EFT arms (B1, B, B9, D, D2) of all Phase 10B ST repeat-0 cells at S2 and S3 (2 100 records): both implementations return the same bound bit for bit, equal to the stored `R_hi`;
- synthetic stress cases covering every fallback branch;
- binding: all 21 steps of 20 held-out and 12 development sequences (data bit for bit; true matrices accepted by both; perturbed matrices and coefficients answered identically, and any acceptance confirmed by the frozen full assembly).

Consequently the compiled arms take the same decisions and produce the same certificates as the frozen arms. This is checked again on every timed cell (§6).

## 3. Arms (13 per cell)

| Method | Frozen arm | Compiled arm | Compiled component(s) |
|---|---|---|---|
| RPS\|P1F (per-system, frozen Phase 6 runner) | A | — | none used |
| RPS-EFT (per-system) | B1 | B1c | residual bound |
| RPS-OPT (per-system) | B | Bc | residual bound |
| RPS-OPT-S09 (per-system) | B9 | B9c | residual bound |
| RPS\|M2A (reuse, frozen Phase 6 runner) | C | Cc | binding |
| M2A-EFT (reuse) | D | Dc | binding, residual bound |
| M2A-EFT-S05 (reuse) | D2 | D2c | binding, residual bound |

## 4. Cells and execution

**Configurations.** ST mode. S3 (primary) and S2 (secondary). Both families, seeds 200–204, repeats 0–2: 60 cells.
- S3 cells run first, ordered by repeat; then the S2 cells.
- One fresh process per cell.

**Each cell.**
- Single thread, pinned to logical CPU 2 (a performance core), HIGH priority.
- Inputs verified against the frozen manifest before the warm-up.
- Symmetric untimed warm-up: every arm once on E3-TS S0, development seed 0.
- Arm order: the interleaved list A, B1, B1c, B, Bc, B9, B9c, C, Cc, D, Dc, D2, D2c, rotated by k = (repeat + seed) mod 13.

**Environment.** CPython 3.10.0 with the pinned lock file plus numba 0.61.2 and llvmlite 0.44.0, in a separate virtual environment. The project environment is not modified. Frozen and compiled arms run in the same process.

No other benchmark or Arb process runs at the same time.

## 5. Limits and stopping rules

- 90 min of cumulative cell wall time (benchmark budget); 30 min per process; 16 GiB peak memory per process. A process that hits a guard is killed and the plan stops; completed cells are analysed, and the incomplete part is reported.
- Every completed cell is replayed with the frozen Phase 10B replay (untimed, outside the budget).
- Hard stop on any REJECTED or UNAVAILABLE record, input mismatch or manifest mismatch.

## 6. Analysis (fixed now; `analyze_compiled.py`)

- **Validity.** An arm in a cell is valid only if all 21 steps are certified and meet τ = 10⁻⁸, and all its records are VERIFIED by the replay.
- **Identity.** For every frozen/compiled pair of arms in every cell, the replay's certificate views and the stored `R_hi` sequences must be identical. A pair that is not identical is reported, and no speed comparison is made with it.
- **Ratios.** Per-system cost / reuse cost (the frozen `measured` metric, I/O excluded), within a cell. Unit = seed (median over valid repeats). Geometric mean over seeds with two-sided 95 % and 90 % t-intervals (df = seeds − 1).
- **Contrasts.**

| Family | Primary exploratory contrast | Other contrasts |
|---|---|---|
| E3-TS | **B9c/Dc** | B9/D (frozen, re-measured), B1c/Dc, Bc/Dc, B9c/D, B9/Dc, A/Cc, A/C |
| D3-TS | **Bc/D2c** | B/D2 (frozen, re-measured), B1c/D2c, B9c/D2c, Bc/D2, B/D2c, A/Cc, A/C |

  Speed-ups of each compiled arm: B1/B1c, B/Bc, B9/B9c, C/Cc, D/Dc, D2/D2c.
- **Descriptive categories** (exploratory; same numerical thresholds as the registered rule):
  - reuse faster beyond 1.2 (GM ≥ 1.2 and lower 95 % bound > 1);
  - reuse faster, directional (lower bound > 1);
  - per-system faster beyond 1/1.2 (GM ≤ 1/1.2 and upper bound < 1);
  - per-system faster, directional (upper bound < 1);
  - otherwise, no directional difference.

  Margin check: whether the 90 % interval lies inside (1/1.2, 1.2).
- **Mechanism.** Mean components per arm (binding, residual, solve, factorizations, estimate), PCG iterations, accurate-bound calls, repeat CV.

## 7. Interpretation rules (fixed now)

- The primary exploratory contrasts at S3 (B9c/Dc at E3-TS, Bc/D2c at D3-TS) are reported next to the registered results, with their category, whatever it is.
- **If a category differs from the registered classification of the same family and size**, the manuscript states that the registered result depends on the interpreted implementation of these two components.
- **If the categories agree**, the manuscript states that compiling them does not change the conclusion, with the measured shift of the ratio.
- The S2 contrasts are reported in the same way as secondary results.
- The result is labelled exploratory and post hoc, is reported separately from the registered results, and does not change any registered classification.
- No projection replaces a missing measurement. Fusing the binding with the PCG matrix–vector product is not implemented; it is discussed only as a possible further reduction.

## 8. Frozen files

`compiled_kernels.py`, `test_kernels.py`, `test_kernels.json`, `run_cell_compiled.py`, `run_plan_compiled.py`, `analyze_compiled.py` and this protocol. Hashes are in `PROTOCOL_FREEZE.json`.
