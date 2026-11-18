# Phase 10B: pre-data amendments to the Phase 10A draft (15 Nov 2026)

**Status.** Every item below was written before any held-out input (seeds 200–204) was generated or inspected.

- A workspace scan (`FINAL_DEV/stage0/heldout_seed_scan.json`) found no file name and no JSON/JSONL record that refers to seeds 200–204: 59 280 files walked, 3 978 JSON files scanned, 0 hits.
- The draft `reports/PHASE10A/PHASE10B_PREREGISTRATION_DRAFT.md` is kept unchanged; its SHA-256 is recorded in the freeze record.
- These amendments clarify and complete the draft. They do not change the primary pairs, the statistic, the directional thresholds or the parity rule.

## A1. Number of arms (six versus seven)

**The draft itself names all seven arms** (§2, §4). The stale "six arms" wording is elsewhere:

- the docstring of the frozen `PHASE10A_DEV/methods/run_cell10a.py` ("Six arms run in this process", followed by a list of seven);
- `reports/PHASE10A/IMPLEMENTATION_AND_MATHEMATICAL_PARITY.md` (heading "The six arms", and "paired cell: six arms").

Both were written before arm B9 was added and were not updated.

**What actually executes is the `ARMS` list, which has seven entries:** A, B1, B, B9, C, D, D2. The Phase 10B worker `FINAL_DEV/phase10b/run_cell10b.py` runs exactly these seven. The pre-freeze smoke test confirmed seven arms × 21 records = 147 records per cell, all VERIFIED (8 cells, ST and MT).

**Primary pairs (unchanged from the draft, §8).** Both were chosen from the Phase 10A development S3 diagnostic, before any held-out data existed.

| Family | Primary per-system | Primary reuse | Development rationale (Phase 10A, S3, ST) |
|---|---|---|---|
| E3-TS | B9 (RPS-OPT-S09) | D (M2A-EFT) | fastest per-system arm (3.925 s); fastest target-valid reuse arm (4.752 s; D2 5.637 s) |
| D3-TS | B (RPS-OPT) | D2 (M2A-EFT-S05) | fastest per-system arm (1.171 s); fastest reuse arm (1.365 s; D 1.472 s) |

Choosing the faster development variant on each side favours reuse, because the reuse arm is the fastest available reuse variant.

## A2. Arm semantics, verified from the frozen code

| Arm | Code path | λ_min certificate | Shift policy σ | λ estimate | Residual bound | Reuse |
|---|---|---|---|---|---|---|
| A | `runner6.run_sequence_5(P1F, RPS)` | Rump 2006 bound I, every step | 0.9 λ̃; σ/5 retries | every step | frozen R_hi | none |
| B1 | `runner10a` `RPS-EFT` | same | 0.9 λ̃ | every step | shared cascade (frozen bound, then EFT bound) | none |
| B | `runner10a` `RPS-OPT` | same | carried σ from the last certified step; on failure a new estimate and σ = 0.5 λ̃ | only on carried-shift failure or target restoration | shared cascade | none |
| B9 | `runner10a` `RPS-OPT-S09` | same | as B, with σ = 0.9 λ̃ | as B | shared cascade | none |
| C | `runner6.run_sequence_5(M2A, RPS)` | Rump anchor plus M2-A min-θ Loewner transfer with assembly discrepancy ζ | anchors 0.9 λ̃ | at anchors | frozen R_hi | gray-box reuse |
| D | `runner10a` `M2A-EFT` | as C | anchors 0.9 λ̃ | at anchors | shared cascade | as C |
| D2 | `runner10a` `M2A-EFT-S05` | as C | anchors 0.5 λ̃ | at anchors | shared cascade | as C |

**Common to all arms:**
- the PARDISO engine, mtype 2, symbolic analysis once per sequence;
- the PCG stopping rule rtol = min(10⁻¹², max(10⁻¹⁶, ½ τ λ^lo / ‖b‖));
- target τ = 1·10⁻⁸ (`0x1.5798ee2308c3ap-27`).

**Common to the five new arms:** the target-restoration rule (one re-anchor per step).

## A3. Statistical interpretation (correction of draft §7)

1. **Renaming.** "NO PRACTICAL DIFFERENCE" is replaced by **NO ESTABLISHED PRACTICAL ADVANTAGE**.
   - A geometric mean between 1/1.2 and 1.2, or an interval that crosses a threshold, means only that neither directional criterion was met. It is not evidence of equivalence.
   - The directional thresholds are unchanged (GM ≥ 1.2 with lower 95 % bound > 1; GM ≤ 1/1.2 with upper 95 % bound < 1).
2. **Per-family categories and joint outcome** (decision logic made explicit; consistent with draft §6–§7):
   - Per family:
     - **INVALID**: a hard stop, a missing run, or a parity failure of the per-system primary arm.
     - **REUSE ACCURACY-INFERIOR**: a parity failure of the reuse primary arm. Draft §6 already classifies such an arm as accuracy-inferior; its speed becomes descriptive only.
     - **REUSE ADVANTAGE**, **PER-SYSTEM ADVANTAGE**, or otherwise **NO ESTABLISHED PRACTICAL ADVANTAGE**.
   - Joint:
     - if both families have the same category among the last three, that category;
     - any INVALID gives **INVALID**;
     - otherwise **MIXED**, reported per family.
3. **Formal equivalence (secondary descriptor; never a decision category).**
   - Two one-sided tests at α = 0.05 each, implemented as: the two-sided **90 %** t-interval of the GM (df = 4) lies entirely inside **(1/1.2, 1.2)**.
   - **Margin justification:** ±20 %, log-symmetric, is the practical-relevance threshold used for every directional decision since Phase 4. Below it, a difference is comparable to the cross-session timing shift observed on this machine (11–13 %, Phase 4) and to plausible implementation differences (NumPy versus compiled kernels).
   - The descriptor is reported as "practically equivalent within the pre-specified ±20 % margin". It is never reported as "no difference".
4. **Directional descriptor (secondary).** Whether the 95 % interval lies below 1, above 1, or includes 1.
5. **Claims attached to outcomes** (replacing the last bullet of draft §7):

| Outcome | Claim that may be made |
|---|---|
| PER-SYSTEM ADVANTAGE | On both held-out families, optimized per-system certification is faster than certificate reuse by a practically relevant margin (≥ 20 % in GM, interval below 1) |
| NO ESTABLISHED PRACTICAL ADVANTAGE | The experiment did not establish a practically relevant speed advantage for either strategy. Statements of equivalence only where the TOST descriptor holds; directional statements only where the 95 % interval excludes 1 |
| REUSE ADVANTAGE | Certificate reuse is faster by a practically relevant margin on both families |
| MIXED | Per-family statements only |
| INVALID | No confirmatory speed claim. Validity findings are reported |

## A4. Statistical unit and repeats (clarification)

- The independent unit is **(family, seed)**: five per family.
- The three timing repeats are **technical replicates**. They enter only through the per-seed median of the within-process ratios. They are never treated as independent samples, and no interval uses them as such.
- Families are decided separately. Any pooled description comes after the per-family results and has no decision role.

## A5. Held-out split semantics (disclosure)

`Family3D(…, split='heldout')` activates the held-out variants of `pilot_configs.json`:

| Family | Development | Held-out (Phase 10B) |
|---|---|---|
| E3-TS | ν = 0.3, a = 0.7 | ν = 0.35, a = 0.6 |
| D3-TS | aspect 3, a = 0.7, S2 N = 9, S3 N = 12 | aspect 4, a = 0.6, S2 N = 8, S3 N = 10 |

- This is the held-out distribution of the Phase 4 protocol. Seeds 100–104 of it are consumed and are not used.
- **Seeds 200–204 are fresh draws from a known variant distribution, not a new distribution.**
- The primary arms were selected at development S3 (D3-TS N = 12, aspect 3). They are applied unchanged to held-out S3 (N = 10, aspect 4).

## A6. Code basis

- **Phase 10B code freeze:** `reports/PHASE10B/PHASE10B_CODE_FREEZE.json`, aggregate `f305dda539b6a51e42de5fd7862265a5233120dc9fa91eb742e3b76873924977`, 67 files. (A first freeze, `ee3b3440…`, was superseded before any held-out data existed, when the second-session stage of A10 was added to `run_plan10b.py` and `analyze10b.py`.)
  - It includes every CARS module the workers import, discovered at runtime, the Phase 10B scripts, the Phase 10A frozen manifest files, and the generator data files.
- **Phase 10A record-keeping defect.** `PHASE10A_DEV/analysis/analyze10a.py` differs from the Phase 10A freeze (`8516e851…`) by one line: the list of analysed stages was widened 13 s after that freeze.
  - Reverting that line reproduces the frozen hash exactly.
  - All method, verifier, EFT and dependency files match the Phase 10A freeze. The analysis script is not used in Phase 10B.
- **Generator import path.** The generator imports `fem3d_ref.py` and `m2_ref.py` from `PHASE3B_DEV/work/reference_impl/`, not from `CARS/reference_impl/` as listed in the Phase 10A manifest. The two copies are byte-identical; the Phase 10B freeze hashes the imported copies.
- **Record label.** The frozen runners write `phase = 'PHASE10A-DEV'` into every record. Phase 10B records are identified by `split = 'heldout'`, by seed and by the cell files. The label is not edited.
- **New Phase 10B scripts.** These are wrappers with identical method semantics:
  - `run_cell10b.py`: thread mode, held-out guard, input pre-verification;
  - `replay10b.py`: the frozen `verify10a`, plus a manifest check;
  - `independent10b.py`, `run_plan10b.py`, `analyze10b.py`;
  - `gen_inputs10b.py`, `compare_inputs10b.py`.
- **Validation of the new scripts on development data before the freeze:**
  - generation reproduced the Phase 10A manifest at E3-TS / D3-TS S0 seed 0;
  - smoke: 8 cells × 147 records VERIFIED, with MKL reporting 8 threads in MT;
  - the decision logic passed 7/7 synthetic cases (`test_decision10b.py`).

## A7. Input freezing (completes draft §3)

- **Generation:** two complete passes in separate processes (`gen_inputs10b.py`), compared file by file (`compare_inputs10b.py`). Any difference is a hard stop.
- **What is frozen** for every (family, size, seed, t):
  - the coefficient vector c_t, as exact hexadecimal strings and as binary `.npy`;
  - the SHA-256 of the canonical matrix, of the rhs and of the coefficients, plus n and nnz;
  - the rhs, as exact hexadecimal, once per family and size (checked identical over t and seeds).
- **Matrices are not stored.** They are a deterministic function of the frozen mesh, the element factors (hash-checked) and c_t, through the frozen assembler. Every cell regenerates them and checks all 21 matrix, rhs and coefficient hashes against the frozen manifest **before** the warm-up (exit 3 and hard stop on mismatch). The replay checks them again.

## A8. Validation sample (completes draft §9)

- **Replay.** Every record of every cell (ST and MT) through the frozen verifier:
  - CHOLMOD cross-engine re-proof of every distinct Rump anchor;
  - M2-A transfer and ζ recomputation;
  - bitwise EFT recomputation.

  Any REJECTED or UNAVAILABLE record, or a manifest mismatch, is a hard stop.
- **Exact residuals** (fmpq; unchanged `independent5.exact_residual_sq`):
  - random sample `random.Random(10_2026)`: per arm (all seven), per size (S2, S3), 4 records from ST and 2 from MT, giving 84 records;
  - any R² < exact ‖b − Ax‖² is a hard stop.
- **Arb λ proofs** (unchanged `independent5.arb_congruence_pd`) of the largest λ^lo claimed by any arm in any mode:
  - **Mandatory:** S2, per family, (seed 201, t = 0) and (seed 203, t = 20).
  - **Optional:** S3, per family, (seed 202, t = 10), one process each, 30-min cap. A capped run is reported as UNAVAILABLE and is not a hard stop. It addresses the open item "no independent λ re-proof at S3".
  - An inconclusive Arb result (not proved within its precision) is reported as unresolved for that matrix. It is treated as a hard stop for any claim about that matrix until resolved.

## A9. Resources and execution order (completes draft §4)

- **Budget:** 6 h for the registered campaign (cells + replays, ledger `budget10b.json`; independent checks are added to the same total); 30 min per process.
- **Peak working-set ceiling: 16 GiB** per process (new, polled watchdog). A resource stop preserves completed cells and is reported as such, never as a result.
- **Stage order:** ST_S3 (primary), ST_S2, MT_S3, MT_S2. Within a stage: repeat → seed → family. Each cell is replayed immediately.
- **Crashes:** a crashed cell (non-zero exit other than an input mismatch) is repeated once, into `<cell>_retry`. The repetition and the reason are reported; a second failure is a hard stop. No cell is re-run for any other reason.

## A10. Secondary analyses (fixed now, none decisive)

- **MT, S2, both modes:** the same pair statistics and categories, reported descriptively.
- All seven arms' costs, all 12 per-system/reuse pair ratios, counterfactual B, end-to-end cost including artifact I/O (a sensitivity; the principal metric excludes I/O by the historical rule), component breakdown, PCG iterations, EFT calls, factorizations, repeat CV, peak memory, fingerprints and certificate determinism across repeats.
- **"Best-versus-best" ratios** (as reported in Phase 10A) are labelled post-hoc and descriptive.
- **Same-machine repeat session (repeatability, not replication).** After the registered campaign and its analysis are complete, one additional ST S3 session (all seven arms, seeds 200–204, one repeat with index 3, a new process per cell) is run and reported as a second session on the same machine and boot. It cannot change the primary outcome. It is **not** an independent-hardware replication; no second machine is available.

## A11. Development versus confirmatory evidence

- Phase 10A development numbers are never pooled with Phase 10B results.
- If Phase 10B differs from development, the held-out result is reported as the confirmatory result and the conclusions are revised accordingly.
