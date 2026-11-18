# SIMP3D with a fixed carried-shift fallback: exploratory protocol (Revision 2)

**Status.** Exploratory and post hoc. The owner authorised this experiment in response to the second referee report (major comment 4), after the original SIMP3D results were known.
- **Supplementary.** The SIMP3D evaluation was supplementary and descriptive from the start (`FINAL_DEV/simp/SIMP_PROTOCOL.md`), and so is this completion.
- **Inputs.** It uses the frozen SIMP3D inputs, settings and per-window targets unchanged (`FINAL_DEV/simp/eval/pass_a`, `simp_settings.json`).
- **Scope.** It cannot change any Phase 10B registered result. The original SIMP3D records (12 of 30 planned runs) are kept and reported as they are.
- **Freeze.** This protocol and the code in §7 are hash-frozen in `PROTOCOL_FREEZE.json` before any timed cell runs.

## 1. Problem being fixed

The original evaluation stopped at its 90-minute budget after 12 of 30 runs. The cause was the carried-shift arms B (RPS-OPT) and B9 (RPS-OPT-S09) in the early windows of instances 301 and 302, through this chain:
1. The carried shift failed (Rump's shifted Cholesky did not complete).
2. The failed factorization reset the PARDISO session, so the previous preconditioner became stale.
3. The frozen code then estimated λ_min from a cold start: ARPACK, smallest algebraic, no shift-invert, 10⁴ restarts.
4. On these high-contrast matrices (E_min = 10⁻⁹) the cold Lanczos run did not converge. The step was UNVERIFIED, and the next carried shift failed again.

These arms took 280–490 s per sequence, against about 10 s for the others. In total B and B9 had 35 UNVERIFIED steps.

The carried shift never failed on the Phase 10B families (no `carry_failure` in any B or B9 record).

## 2. The fix (`fallback_runner.py`; arms Bf, B9f)

**One change.** Whenever a new λ_min estimate is needed after the first anchor of a sequence and no valid preconditioner exists:
- λ_min is estimated by the same five-step inverse iteration as the frozen estimator;
- the operator is a fresh PARDISO factorization of the unshifted matrix A, i.e. shift-invert with shift 0;
- the iteration starts from the last eigenvector estimate.

**Cost accounting.** The factorization uses the anchor session, whose pattern is unchanged, so no new symbolic analysis is needed. It is charged to the `estimate` component and counted as a factorization.

**If that factorization fails**, the frozen cold estimate is used, and the event is recorded.

**Unchanged:** the first anchor of a sequence, the anchor rule (σ = frac·λ̃, /5 retries), the carried shift, the shared residual strategy, the restoration rule, the timers and the accounting.

**Implementation.**
- `run_sequence_fb` is compiled from the source of the frozen `run_sequence_10a`, with the estimate block replaced by textual substitution.
- The full diff is stored in `check_fallback.json`.
- No frozen file is modified.

**Checks before the freeze** (`check_fallback.json`):
1. **Phase 10B identity.** On every ST S3 repeat-0 sequence (both families, seeds 200–204), the rebuilt B and B9 arms reproduce the certificate view of every stored record. The fix is inactive there.
2. **Forced-failure harness test** (development seed 0, E3-TS S1). The carried shift is made to fail genuinely at three steps.
   - The frozen code records the cold estimate.
   - The fixed code records `shift-invert`, certifies all 21 steps, and its records are VERIFIED by the frozen Phase 10B replay.

No SIMP3D evaluation data were used before the freeze.

## 3. Arms (7 per cell)

| Role | Arms |
|---|---|
| Per-system | A (RPS\|P1F), B1 (RPS-EFT), **Bf** (RPS-OPT, fixed fallback), **B9f** (RPS-OPT-S09, fixed fallback) |
| Reuse | C (RPS\|M2A), D (M2A-EFT), D2 (M2A-EFT-S05) |

The frozen B and B9 arms are not re-run; their failures are documented from the original runs.

## 4. Cells and execution

**Cells.** 5 instances (301–305, S3) × 2 windows (early, late) × 3 repeats = 30 cells, ordered by repeat. Each cell is the frozen SIMP cell (`run_cell_simp.py`) with Bf and B9f in place of B and B9:
- inputs hash-checked before the warm-up;
- single thread, pinned to logical CPU 2, HIGH priority;
- the same symmetric untimed warm-up;
- rotated order k = (repeat + instance) mod 7.

**Environment.** The project environment of the original SIMP runs (CPython 3.10.0, pinned lock file).

**Replay.** Each cell is replayed immediately with the frozen SIMP replay (`replay_simp.py`), untimed and outside the budget.

No other benchmark or Arb process runs at the same time. (The original SIMP runs overlapped with an Arb proof process on another core.)

## 5. Limits and stopping rules

- 90 min of cumulative cell wall time; 30 min per process; 16 GiB peak memory.
- Frozen SIMP settings must be unchanged.
- Hard stop on any non-VERIFIED record.
- A stop leaves the completed cells for analysis, and the incomplete part is reported.

## 6. Analysis (fixed now; `analyze_simp_fb.py`) and interpretation

- **Per window.** Unit = instance (median over valid repeats). GM over the 5 instances with a 95 % t-interval (df = 4).
- **Validity.** An arm in a cell is valid if all 21 steps are certified and target-met and all its records are VERIFIED. A ratio uses only cells where both arms are valid. Exclusions are reported.
- **Pairs.** Every per-system arm against every reuse arm. **B9f/D and Bf/D2** are the analogues of the Phase 10B primary pairs.
- **Fallback events.** Carried-shift failures, fallback kinds and the `estimate` cost.
- **Categories.** The registered thresholds are applied descriptively, with no decision role.

**Interpretation rules (fixed now).**
- **Practical recommendation.** The manuscript's recommendation for carried shifts follows this result:
  - if Bf/B9f certify all steps in all runs, the recommendation is to carry the shift *with* a shift-invert fallback, and the frozen B/B9 failures are reported as a defect of the frozen fallback, not of the carried shift;
  - if they still fail, the recommendation against relying on carried shifts for high-contrast sequences is kept.
- **Labelling.** The comparison with the reuse arms is reported descriptively, labelled post hoc.

## 7. Frozen files

`fallback_runner.py`, `check_fallback.py`, `check_fallback.json`, `run_cell_simp_fb.py`, `run_plan_simp_fb.py`, `analyze_simp_fb.py` and this protocol. Hashes are in `PROTOCOL_FREEZE.json`.
