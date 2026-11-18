# SIMP3D with compiled operations: exploratory protocol (Revision 3)

**Status.**
- Exploratory and post hoc. The owner authorised it in response to the third referee report (item B).
- It uses the frozen SIMP3D inputs, settings and per-window targets, and it changes no confirmatory result.
- This protocol and the files in §6 are hash-frozen before any evaluation cell. Its smoke test used the development pilot only.

## 1. Question

In the Revision 2 completion both sides used the interpreted binding and residual enclosure. The question: how do the ratios of the corrected carried-shift arms to the reuse arms change when all compiled operations of the full-compilation experiment are used?

## 2. Arms (8 per cell)

| Role | Frozen operations | Compiled operations (level `full`) |
|---|---|---|
| Per-system, shift-invert fallback (Revision 2) | Bf (RPS-OPT), B9f (RPS-OPT-S09) | Bfc, B9fc |
| Reuse | D (M2A-EFT), D2 (M2A-EFT-S05) | Dc, D2c |

The compiled operations are binding, accurate residual enclosure and shift construction (`REVISION3/fullc/compiled_full.py`).

## 3. Cells and execution

- **Cells.** 30 cells (5 instances × 2 windows × 3 repeats), ordered by repeat.
- **Execution.** The frozen SIMP cell procedure, with the order [Bf, Bfc, B9f, B9fc, D, Dc, D2, D2c] rotated by (repeat + instance) mod 8.
- **Environment.** The Revision 2 Numba environment.
- **Replay.** Every cell is replayed immediately with the frozen SIMP replay (untimed).
- **Limits.** 90 min of cell time, 30 min per process, 16 GiB; hard stop on any non-VERIFIED record.
- **Load.** No concurrent benchmark.

## 4. Analysis (`analyze_simpc.py`)

- **Per window.** Unit = instance (median over repeats); GM with a 95 % t-interval.
- **Pairs.** B9f/D and Bf/D2 (frozen), B9fc/Dc and Bfc/D2c (compiled), and the speed-ups.
- **Ratios without the shared estimate.** The four pairs are also computed with the first-step eigenvalue estimate subtracted from both costs. This estimate is a cold computation common to all arms; the subtraction answers the referee's request.
- **Identity.** The frozen and compiled versions of each arm must store identical R_hi and λ_lo sequences.
- **Categories.** Descriptive only.

## 5. Interpretation (fixed now)

- **Main text.** It reports the compiled pairs B9fc/Dc and Bfc/D2c as the comparison of the corrected carried shift with reuse, replacing "about twice as fast" by the measured factor, with the cold-start-free ratios next to them.
- **Labelling.** The result is exploratory and supplementary.

## 6. Frozen files

`SIMPC_PROTOCOL.md`, `run_cell_simpc.py`, `run_plan_simpc.py`, `analyze_simpc.py`.
