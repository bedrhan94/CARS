# Full-compilation experiment (with structural envelope): exploratory protocol (Revision 3)

**Status.**
- Exploratory and post hoc. The owner authorised it in response to the third referee report (items A and C), after all earlier results were known.
- Seeds 200–204 are consumed. The experiment cannot change any confirmatory result.
- This protocol and the files in §7 are hash-frozen in `PROTOCOL_FREEZE.json` before any timed cell.

## 1. Questions

**A.** In Revision 2 only the operations that reuse needs were compiled (binding, accurate residual enclosure). The construction of the shifted matrix for Rump's criterion, which the per-system arms repeat at every step, remained interpreted (0.18 s per sequence at E3-TS S3). How does the comparison change when it is compiled as well?

**C.** How does the comparison at E3-TS S2 change when the compiled operations are combined with the structural sparsity envelope (Revision 1), which removes the repeated symbolic analyses of the per-system arms?

## 2. Compiled operations (`compiled_full.py`)

- **Level `partial`.** The Revision 2 kernels: binding and accurate residual enclosure.
- **Level `full`.** `partial` plus three kernels:
  - an exact symmetry check in one linear pass;
  - the diagonal positions;
  - the shifted copy, with the values copied and the diagonal updated by dn(a_ii − s̄); index arrays shared.

  `rump_anchor_once` is rebuilt from the frozen source with only the shift block replaced. The Rump constant and everything else stay frozen.
- **Equivalence before the freeze** (`test_shift.json`, ALL_IDENTICAL true). All checks were bit for bit:
  - 3 986 symmetry and diagonal cases on all held-out S2/S3 and development S0/S1 matrices, including five perturbations of every matrix;
  - 672 shifted copies;
  - 12 end-to-end anchors (returned fields and PARDISO solves).

## 3. Arms (6 per cell)

| Family | `partial`, frozen session | `full`, frozen session | `full`, envelope session |
|---|---|---|---|
| E3-TS | B9c, Dc | B9f, Df | B9fs, Dfs |
| D3-TS | Bc, D2c | Bf, D2f | Bfs, D2fs |

Sessions are the instrumented frozen session or the Revision 1 envelope session; both count analyses and factorizations.

## 4. Cells and execution

- **Cells.** E3-TS and D3-TS at S3 (primary) and S2; seeds 200–204; repeats 0–2; 60 cells. All S3 cells come first, ordered by repeat.
- **Execution.**
  - Fresh process per cell; single thread on logical CPU 2; HIGH priority.
  - Frozen-input check, then a symmetric warm-up.
  - Arm order [Pc, Qc, Pf, Qf, Pfs, Qfs], rotated by (repeat + seed) mod 6.
- **Environment.** The Revision 2 Numba environment.
- **Load.** No concurrent benchmark.
- **Limits.**
  - 90 min of cell time, 30 min per process, 16 GiB.
  - Every cell is replayed with the frozen Phase 10B replay (untimed); hard stop on any non-VERIFIED record.

## 5. Analysis (`analyze_fullc.py`)

- **Validity.** As in Revision 2.
- **Ratios.** Unit = seed (median over repeats); GM with 95 % and 90 % t-intervals.
- **Identity.** Pc/Pf and Qc/Qf must have identical certificate views and R_hi.
- **Primary contrasts.**
  - **A:** Pf/Qf (B9f/Df at E3-TS; Bf/D2f at D3-TS) at S3.
  - **C:** Pfs/Qfs at E3-TS S2.
- **Other contrasts.** Pc/Qc (Revision 2 level, re-measured), Pfs/Qf, and the speed-ups Pc/Pf, Qc/Qf, Pf/Pfs, Qf/Qfs.
- **Descriptive categories.** Thresholds of the pre-specified rule; margin check.
- **CG loop.** After the cells, `cg_overhead.py` (untimed) estimates the interpreted per-iteration overhead of the conjugate gradient loop. It measures the time outside the matrix–vector products and preconditioner solves on E3-TS and D3-TS S3 matrices. The overhead per sequence of each arm is then reported as overhead per iteration × iteration count. This is an estimate, labelled as such, not a timed comparison.

## 6. Interpretation rules (fixed now)

- **A.** The manuscript reports Pf/Qf as the comparison with all operations compiled that could be compiled without changing a certificate, together with its category. The headline number in the abstract is replaced by this result, whatever its direction. The estimated CG-loop overhead is reported as the remaining known asymmetry, without adjusting any ratio.
- **C.** The manuscript reports Pfs/Qfs at E3-TS S2 in place of the category of the compiled S2 row.
- **Labelling.** All results are exploratory and post hoc; no confirmatory classification changes.

## 7. Frozen files

- `FULLC_PROTOCOL.md`
- `compiled_full.py`
- `test_shift.py`, `test_shift.json`
- `run_cell_fullc.py`, `run_plan_fullc.py`
- `analyze_fullc.py`
- `cg_overhead.py`
