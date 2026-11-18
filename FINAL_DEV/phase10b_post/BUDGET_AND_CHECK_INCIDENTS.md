# Phase 10B: budget ledger correction and independent-check incidents (recorded 15 Nov 2026, 18:05)

## 1. Ledger over-charge (bookkeeping bug; the ledger file is left unchanged)

- `run_independent10b.py` started its seven check processes in parallel. After all had ended, it charged **each** process the elapsed time since its own start, measured at the end of the whole batch (1 809.5 s). Every process was therefore charged about 30 min, whatever its true run time.
- `budget10b.json` thus over-states the independent checks: 7 × 1 809.5 s = 3.52 h.

**Actual run times, from file timestamps** (all processes started 17:31:49):

| Process | Ended | Run time | Outcome |
|---|---|---|---|
| E: exact-residual sample | 17:35:54 | 4.1 min | 84/84 valid |
| L: E3-TS S2 seed 201 t 0 (mandatory) | 18:00:19 | 28.5 min | **proved** |
| L: D3-TS S2 seed 201 t 0 (mandatory) | 17:52:34 | 20.7 min | crashed: FLINT "Unable to allocate memory" |
| L: E3-TS S2 seed 203 t 20 (mandatory) | 17:52:34 | 20.7 min | crashed: same |
| L: D3-TS S2 seed 203 t 20 (mandatory) | 17:52:35 | 20.8 min | crashed: same |
| L: E3-TS S3 seed 202 t 10 (optional) | 18:01:39 | 30 min (cap) | UNAVAILABLE (cap, as registered) |
| L: D3-TS S3 seed 202 t 10 (optional) | 18:01:39 | 30 min (cap) | UNAVAILABLE (cap, as registered) |
| **Sum** | | **≈ 2.58 h** | |

**Corrected campaign total so far:**

| Item | Hours |
|---|---|
| Registered cells and replays | 2.079 |
| Same-machine second session | 0.245 |
| Independent checks | 2.58 |
| **Total** | **≈ 4.90** |

The 6 h limit is respected. The ledger as written (5.84 h) is also below 6 h.

## 2. Arb crashes (infrastructure, not mathematics)

- **Cause.** Three of the four mandatory Arb proofs aborted with memory-allocation failures. The ball-arithmetic congruence of an S2 matrix needs several GiB, and six such processes ran at the same time on a 31.7 GB machine. The crash comes from the parallel scheduling chosen for the checks; it says nothing about the claims being checked.
- **Remedy.** As with crashed cells, the pre-registration allows a crash to be repeated once, with the repetition reported. The three crashed mandatory proofs are repeated **once, one at a time**, after the SIMP evaluation. Their run times are added to the corrected total above.
- **Optional S3 proofs.** They reached their registered 30-min cap and are reported as UNAVAILABLE. They are not repeated.

## 3. Outcome of the single repeat, and a post-registration completion run (recorded 15 Nov 2026, 19:45)

**Single registered repeat** (sequential, pinned to the efficiency cores at below-normal priority, concurrently with the SIMP evaluation on CPU 2; log `phase10b/independent/rerun_crashed_arb.log`):

| Proof | Run time | Outcome |
|---|---|---|
| D3-TS S2 seed 201 t 0 | 30.8 min | cap reached (exit 124): UNAVAILABLE |
| E3-TS S2 seed 203 t 20 | 26.4 min | **proved** (max claim / LAPACK λ₁ = 0.900) |
| D3-TS S2 seed 203 t 20 | 31.2 min | cap reached (exit 124): UNAVAILABLE |

- **Registered status.** E3-TS: both mandatory proofs proved. D3-TS: neither mandatory proof completed within the registered limits.
- **Interpretation.** A capped run is not an inconclusive proof: no Arb result exists for these two matrices. Under the registered rule for unresolved Arb results, no claim of Arb-level confirmation is made for them. Their certificates remain verified by the frozen replay verifier and by the CHOLMOD re-proof.
- **Cause of the cap.** The D3-TS S2 matrices have n = 6 727, against 6 084 for E3-TS S2. The congruence is dense and scales with n³, a factor of about 1.35. On efficiency cores the E3-TS proof already needed 26.4 min, so the cap was too tight for D3-TS. This was not anticipated when the 30-min limit was registered.

**Budget, corrected** (sum of process run times):

| Item | Hours |
|---|---|
| As of section 1 | 4.90 |
| Three repeats (1 848 + 1 584 + 1 872 s) | 1.47 |
| **Total** | **≈ 6.37** |

- **Overrun.** This exceeds the registered 6 h limit by about 0.37 h. I should have checked the remaining budget before starting the repeats. The overrun is disclosed as a deviation.
- **What it affects.** Only validation runs; the registered cells and replays (2.08 h) were complete long before.
- **Wall-clock time.** The independent checks ran in parallel or concurrently with other work, so the campaign's wall-clock time is lower than this sum.

**Post-registration completion run (DEVIATION, not part of the registered plan).**
- **Runs.** The two D3-TS proofs, one at a time, alone, pinned to performance core CPU 4 at normal priority, 75-min cap each (`phase10b_post/complete_d3_arb.sh`; outputs `*_postreg.json`, log `independent/complete_d3_arb.log`).
- **Rationale.** These are correctness checks of certificates that are already verified. Their outcome cannot change any cost ratio or decision category. A failure, that is, a claim that Arb cannot prove, would be reported as a false-certificate finding.
- **Reporting.** The result is reported separately from the registered validation, labelled as post-registration, and its run time is not counted against the registered budget.

**Outcome of the post-registration completion run** (log `phase10b/independent/complete_d3_arb.log`):

| Proof | Run time | Result | Largest claim / LAPACK λ₁ (arm, mode) |
|---|---|---|---|
| D3-TS S2 seed 201 t 0 | 20.2 min | **proved** | 0.900 (B9, ST) |
| D3-TS S2 seed 203 t 20 | 19.6 min | **proved** | 0.901 (B1, MT) |

- **Cause of the earlier cap hits.** Alone on a performance core, each proof needed about 20 minutes. The earlier cap hits therefore came from running on the efficiency cores alongside the SIMP evaluation.
- **Status.** The results are reported as post-registration evidence, separately from the registered validation, in which D3-TS has no completed Arb proof.
