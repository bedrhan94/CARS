# Revision 3: run notes (post hoc experiments)

**Written for** the authors and a future auditor. Times are local workstation time (16 November 2026; times after midnight fall on 17 November).

## Before the freeze

- **19:41.** Preservation snapshot taken (`preserve/before.json`).
- **Shift kernels** (`fullc/compiled_full.py`).
  - **Symmetry check.** The first version used a binary search per entry and was slower than the frozen code (8.3 against 5.7 ms per call on an S3 matrix); it was replaced by a linear-merge check (1.1 ms).
  - **NaN on the diagonal.** The first run of the equivalence test then found one difference per matrix: a NaN on the diagonal. The frozen check compares A with its transpose, so a diagonal NaN makes it fail; the kernel had skipped the diagonal. The kernel was corrected to reproduce this.
  - **Final test.** The test was re-run: ALL_IDENTICAL (`fullc/test_shift.json`).
- **Feasibility probe at n = 104 544** (development seed 0, `large/probe_large.json`). The first version of the probe reused a preconditioner after a later anchor had replaced its factorization (stale view); the probe was corrected.
- **Input manifest of the large size** (`large/inputs_manifest_large.json`). Generated twice per seed; the two generations are identical.
- **Smoke tests** (development data only): all three runners, their replays and the three analyses.
- **19:57.** Protocols frozen (`fullc/`, `simpc/` and `large/PROTOCOL_FREEZE.json`).

## Execution

The three plans ran sequentially in one chain, so no timed cells overlapped:

| Step | Times | Result |
|---|---|---|
| Full-compilation plan | 19:57–20:20 | 60 cells, no stop |
| Its replays | until 20:39 | all VERIFIED |
| Analysis and CG-overhead estimate | after 20:39 | untimed |
| SIMP plan, each cell replayed immediately | 20:39–22:03 | 30 cells (49.1 budget minutes), no stop; 5 040 / 5 040 VERIFIED |
| SIMP analysis | 22:03 | untimed |
| Large-size cells | 22:03–22:41 | 3 cells (37.2 budget minutes; 742–745 s each), no stop |
| Large-size replays (untimed, CHOLMOD re-proofs) | 22:41–00:47 | 378 / 378 VERIFIED (2 403, 2 494 and 2 686 s) |
| Large-size analysis | 00:47 | untimed; no invalid run; identity 6/6 |

- **Load during timed cells.** While cells ran, the only other processes were:
  - text editing and short patch scripts (under a second each);
  - reads of finished analysis files;
  - lightweight watcher loops (`sleep` and `grep` on log files).

  No table generation, LaTeX build or other benchmark ran.
- **After the last timed cell.** From about 00:03, while the untimed large-size replays were still running, table generation and LaTeX builds ran. The replays only verify records; load does not affect their result.
- **One watcher bug.** Its stop condition matched "why=None"; it ended early, and the plan was not affected.

## After the runs

- **Generator fixes.** The table generators needed two fixes before their first full run:
  - a missing record load for the equivalence-test macros;
  - a loop variable that shadowed the file-writer function.

  Neither affects any number.
- **Protocol rule C.** The category of the E3-TS S2 row of Table 4 is taken from the comparison with the envelope.
- **Documents.** Section and table numbers in the documents were written as label tokens and resolved from the .aux files of the final build (`docs/resolve_refs.py`).
