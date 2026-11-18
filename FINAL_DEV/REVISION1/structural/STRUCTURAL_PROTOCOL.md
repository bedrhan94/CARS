# Structural-envelope PARDISO experiment: exploratory protocol (Revision 1)

**Status.** Exploratory and post hoc. The experiment was authorised by the owner's revision brief after the Phase 10B results were known.
- **Seeds.** It uses the frozen Phase 10B held-out sequences, seeds 200–204. These seeds are consumed: the data are not fresh held-out data.
- **Scope.** It cannot change, replace or reclassify any registered Phase 10B result. The Phase 10B pre-registration is not modified.
- **Freeze.** This protocol and the code listed in §8 are hash-frozen in `PROTOCOL_FREEZE.json` before any timed cell runs.

## 1. Question

Does the E3-TS S2 secondary result of Phase 10B persist when both compared strategies can reuse a stable structural sparsity envelope? That result was B9/D = 1.230 [1.191, 1.271], with reuse faster.

Phase 10B background (exploratory diagnostic, `FINAL_DEV/phase10b_post/symbolic_diag.json`):
- The frozen PARDISO session re-analyses whenever an incoming matrix stores an entry outside the pattern analysed so far.
- Because exact cancellations are not stored, at E3-TS S2 this happened at 20–21 of 21 steps for the per-system arms and 4–5 times for the reuse arm.
- One analysis cost about as much as one numerical factorization (27 against 26 ms).

## 2. The structural envelope

For a mesh with element free-dof lists `mesh.dofs`, the envelope E is the set of all pairs (i, j) of free degrees of freedom that belong to a common element, including the diagonal.
- **Value-independent.** It is computed from the connectivity only, never from coefficient values or benchmark outcomes (`struct_session.envelope`).
- **No change to the frozen code.** `StructSession` subclasses the frozen session and only pre-seeds its union pattern with E. Since every stored matrix lies in E, the session analyses once per sequence, and again only after a factorization error, exactly as before.
- **Explicit zeros.** Entries of E that a matrix does not store are passed to PARDISO as explicit zeros, as the frozen session already does for its union pattern.
- **Charged cost.** The envelope is built inside the first factorization call of each sequence, so its cost is charged to the structural arms.

**Equivalence**, checked before this protocol was frozen (`check_envelope.json`, ALL_EQUIVALENT = true):
- For all 315 matrices (E3-TS S2 and S3, D3-TS S2, seeds 200–204, t = 0–20), the stored pattern lies in E.
- The values handed to PARDISO reconstruct each matrix bitwise.
- No matrix stores an explicit zero.

The envelope adds at most 286 explicit zeros at E3-TS S2 and 510 at S3. At D3-TS S2 it equals the stored pattern.

Every certificate refers to the original stored matrix, whose SHA-256 is in the record, and is re-verified against it by the frozen replay.

## 3. Arms (frozen Phase 10A methods; only the PARDISO session class differs)

| Family | Frozen session (instrumented, unchanged behaviour) | Structural envelope |
|---|---|---|
| E3-TS | B9 (RPS-OPT-S09), D (M2A-EFT) | B9s, Ds |
| D3-TS | B (RPS-OPT), D2 (M2A-EFT-S05) | Bs, D2s |

The instrumentation times PARDISO phases 11 (analysis) and 22 (numerical factorization) and counts them in both session classes.

## 4. Cells and execution

**Configurations.**
- E3-TS S2: main diagnostic.
- E3-TS S3: size control.
- D3-TS S2: fixed-pattern sanity check.

Each configuration uses seeds 200–204 and repeats 0–2, giving 45 cells. Cells run in a fresh process each, ordered by repeat (all configurations of repeat 0 first).

**Each cell.**
- Single thread, pinned to logical CPU 2 (a performance core), HIGH priority.
- Inputs verified against the frozen manifest before the warm-up.
- Symmetric untimed warm-up: every arm once on E3-TS S0, development seed 0.
- Rotated order, k = (repeat + seed) mod 4.

No other benchmark or Arb process runs at the same time.

**Measurements per arm and cell:**
- certified cost (the frozen `measured` metric, I/O excluded) and all frozen components;
- number and time of PARDISO analyses;
- number and time of numerical factorizations;
- envelope construction time;
- factor fill (`nnz_factor`);
- PCG iterations, accurate-residual-bound calls and re-anchorings;
- peak working set;
- fingerprints.

## 5. Limits and stopping rules

- 90 min of cumulative cell wall time (benchmark budget); 30 min per process; 16 GiB peak memory per process. A process that hits a guard is killed and the plan stops; completed cells are analysed.
- Every completed cell is replayed with the frozen Phase 10B replay (untimed, outside the benchmark budget).
- Hard stop on any REJECTED or UNAVAILABLE record, input mismatch or manifest mismatch.

## 6. Analysis (fixed now; `analyze_struct.py`)

- **Validity.** An arm in a cell is valid only if all 21 steps are certified, all meet τ = 10⁻⁸, and all its records are VERIFIED by the replay. Invalid arms are excluded from every ratio involving them and reported, not treated as slow observations.
- **Ratios.** Per-system cost / reuse cost, within a cell. Unit = seed (median over valid repeats). Geometric mean over seeds with a two-sided 95 % t-interval (df = seeds − 1).
- **Pairs:**

| Family | Pairs |
|---|---|
| E3-TS | **B9s/Ds (primary exploratory contrast: both sides with the envelope)**, B9/D (frozen pair, re-measured), B9s/D, B9/B9s, D/Ds |
| D3-TS | **Bs/D2s (primary exploratory contrast)**, B/D2, Bs/D2, B/Bs, D2/D2s |

- **Descriptive categories** (exploratory; same numerical thresholds as the registered rule, applied to the primary exploratory contrast):
  - reuse faster beyond 1.2 (GM ≥ 1.2 and lower bound > 1);
  - reuse faster, directional (lower bound > 1);
  - per-system faster beyond 1/1.2 (GM ≤ 1/1.2 and upper bound < 1);
  - per-system faster, directional (upper bound < 1);
  - otherwise, no directional difference.
- **Mechanism.** Analyses per sequence, time per analysis and per numerical factorization, envelope cost, fill, components, iterations, repeat CV.

## 7. Interpretation rules (fixed now)

- If B9s/Ds at E3-TS S2 is not "reuse faster", the manuscript reports prominently that the S2 reuse advantage of the frozen implementations disappears when both sides keep a stable structural pattern.
- If it remains "reuse faster", the manuscript reports that equally prominently.
- Either way, the result is labelled exploratory, is reported separately from the registered results, and does not change any registered classification.
- If the experiment cannot be completed validly within the limits, the limitation is reported. No projected value is substituted.

## 8. Frozen files

`struct_session.py`, `run_cell_struct.py`, `run_plan_struct.py`, `analyze_struct.py`, `check_envelope.py` and this protocol. Hashes are in `PROTOCOL_FREEZE.json`.
