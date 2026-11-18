# Large-size experiment (n ≈ 10⁵): exploratory protocol (Revision 3)

**Status.**
- Exploratory, post hoc and descriptive. The owner authorised one experiment at about 10⁵ unknowns in response to the third referee report (item D), lifting for this experiment only the earlier restriction to sizes S2 and S3.
- It uses development data: E3-TS geometry and development parameters (ν = 0.3, α = 0.7), development seeds 0–2. It makes no held-out claim.
- This protocol, the input manifest and the files in §7 are hash-frozen before any timed cell.

## 1. Question

The manuscript's structural argument predicts that at larger sizes the factorization grows faster than binding and residual work, so the balance may move towards reuse. How do the per-system and reuse arms compare at n = 104 544 (N = 32), and how do their components scale from S2 and S3?

## 2. Feasibility (untimed probe, development seed 0; `probe_large.json`)

| Item | Value |
|---|---|
| Stored entries | 7.96 M |
| Factor entries | 101 M |
| One anchor factorization | about 5 s |
| Cold first-step estimate | 12.7 s |
| PCG solve | 0.8 s (12 iterations) |
| Accurate residual bound (frozen code) | 0.6 s |
| Peak working set | 2.2 GiB |

## 3. Inputs

`prepare_large.py` generated each sequence twice; both generations hash identically. It wrote the SHA-256 of every matrix, right-hand side and coefficient vector to `inputs_manifest_large.json`. Each cell verifies its inputs against this manifest before the warm-up.

## 4. Arms (6 per cell)

| Operations | Session | Per-system | Reuse |
|---|---|---|---|
| frozen | frozen | B9 (RPS-OPT-S09) | D (M2A-EFT) |
| compiled (level `full`) | frozen | B9f | Df |
| compiled (level `full`) | structural envelope | B9fs | Dfs |

## 5. Cells, execution and limits

- **Cells.** Seeds 0, 1, 2; one repeat; one fresh process each.
- **Execution.** Single thread on logical CPU 2, HIGH priority; symmetric warm-up on E3-TS S0; order rotated by seed mod 6.
- **Limits.** 90 min of cell time, 45 min per process, 16 GiB peak memory. A stop leaves the completed cells for the analysis, and the stop is reported.
- **Replay.** Every completed cell is replayed with the frozen Phase 10B replay, including CHOLMOD re-proofs (untimed); hard stop on any non-VERIFIED record.
- **Load.** No concurrent benchmark.

## 6. Analysis (`analyze_large.py`) and interpretation

- **Per seed.** Ratios per seed and their GM with a 95 % t-interval (df = 2), reported with the per-seed values. No category is applied, because three development seeds with one repeat are too few for an inferential statement.
- **Pairs.** B9/D, B9f/Df, B9fs/Dfs, B9fs/Df, and the speed-ups.
- **Scaling.** Per-step costs of factorization, binding, residual and PCG, plus PARDISO analyses, factor size and memory. They are compared with the same compiled arms at S2 and S3 from the full-compilation experiment.
- **Identity.** B9/B9f and D/Df must store identical R_hi.
- **Reporting.** The manuscript reports the result as a single descriptive data point at about 10⁵ unknowns on development data, whatever its direction, and states what it cannot show (one family, three seeds, one repeat, one machine).

## 7. Frozen files

- `LARGE_PROTOCOL.md`
- `probe_large.py`, `probe_large.json`
- `prepare_large.py`, `inputs_manifest_large.json`
- `run_cell_large.py`, `run_plan_large.py`
- `analyze_large.py`
