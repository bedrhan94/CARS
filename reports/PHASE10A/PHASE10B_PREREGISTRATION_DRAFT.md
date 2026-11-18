# Phase 10B / E1: pre-registration DRAFT (15 Nov 2026). NOT FROZEN, NOT EXECUTED

**Status.**
- This is a draft for the owner's review. It becomes binding only when the owner approves it and it is frozen by SHA-256 *before* any held-out input is generated.
- **No held-out seed (200–204) has been generated or inspected.**
- Nothing in this draft was changed after looking at held-out data, because none exists.

## 1. Question
On fresh held-out data, is gray-box certificate reuse (M2-A, with the same rigorous residual strategy) faster than an optimized per-system Rump verifier (RPS-opt) at equal certified accuracy?

This is a **confirmatory** test of the development finding. Development evidence (Phases 5, 6 and 10A, frozen protocols) shows the per-system method is at least as fast; §8 records the expected direction honestly.

## 2. Methods (frozen code: `PHASE10A_DEV/FROZEN_CODE_MANIFEST10A.json`, aggregate SHA-256 `8516e85196b508851174f8f4ca688d232286bae6f28884949de062b8dbf654de`)

| Role | Arm | Definition |
|---|---|---|
| **Primary per-system** (per family, §8) | B9 (E3-TS) / B (D3-TS) = RPS-opt | Rump 2006 + PARDISO (symbolic reuse); carried shift (fail closed), σ = 0.9 λ̃ (B9) or 0.5 λ̃ (B); shared cascade residual strategy; shared target restoration |
| **Primary reuse** (per family, §8) | D (E3-TS) / D2 (D3-TS) = M2-A + EFT | the faster **target-valid** M2-A + EFT variant on development data, which favours reuse |
| Secondary (reported, not decisive) | A = frozen RPS, C = frozen M2-A, B1 = RPS-EFT, and the non-primary B/B9 and D/D2 | as in `IMPLEMENTATION_AND_MATHEMATICAL_PARITY.md` |

All arms use **τ = 1·10⁻⁸** (as in Phases 5–6 and 10A).

## 3. Data
- **Families:** E3-TS and D3-TS, with the frozen generator and configurations, held-out split.
- **Seeds:** **200, 201, 202, 203, 204** (never generated before); t = 0…20.
- **Sizes:** **S3 (primary)**, S2 (secondary).
- **Input freezing** (as in Phase 4):
  - coefficient arrays and matrix / rhs SHA-256 for every (family, size, seed, t) are generated **twice**, in separate processes, and must be bit-identical;
  - they are stored as hex and binary with hashes;
  - all runs read or verify the frozen inputs.
- **Generation happens only after the owner approves this document and it is frozen.**

## 4. Execution protocol
- **Machine and environment:** the Phase 10A machine (i7-14700K, Windows 11, venv as in Phase 10A).
- **Thread modes:**
  - **ST (primary):** one thread pinned to logical CPU 2;
  - **MT (secondary):** 8 threads on logical CPUs 0, 2, …, 14.
- **Cells:** one fresh process per (family, size, seed, repeat, mode); all seven arms in that process; symmetric untimed warm-up; arm order rotated by (repeat + seed).
- **Repeats:** 3 per cell.
- **Artifacts:** solution vectors written (I/O excluded from costs, historical rule).
- **Budget:** ≤ 6 h total, ≤ 30 min per process. A budget stop is reported as such and is never treated as a result.
- **Hard stops:** any non-zero exit, any REJECTED or UNAVAILABLE replay record, any failed exact check, any input-hash mismatch.

## 5. Primary statistic
- **Unit:** (family, seed). 5 units per family.
- **Cost:** measured certified wall-clock cost A per 21-step sequence (all components except artifact I/O).
- **Ratio per (unit, repeat):** R = cost(primary per-system) / cost(primary reuse), computed within one process. Per unit: the **median** over the 3 repeats.
- **Across units:** geometric mean GM_f per family, with a two-sided 95 % t-interval on log R (df = 4).
- **Reported alongside, not decisive:** counterfactual B, component breakdown, MT ratios, per-seed values, timing CV.

## 6. Accuracy-parity rule (applied before any speed comparison)
- A run is **parity-valid** iff all 21 records are CERTIFIED **and** target-met (ε ≤ τ) **and** VERIFIED on replay.
- The speed comparison is reported as decisive only if **both** primary arms are parity-valid in **every** run.
- A primary arm that misses parity anywhere is classified **accuracy-inferior** for that family, whatever its speed. This is reported first, and the speed ratio becomes descriptive only.

## 7. Decision rule (directional thresholds fixed now; 1.2 is the threshold used throughout Phases 4–6)

| Outcome | Condition (per family f, both families required unless stated) |
|---|---|
| **REUSE ADVANTAGE** | parity in both families, GM_f ≥ 1.2 **and** the lower 95 % bound > 1.0 in **both** families |
| **PER-SYSTEM ADVANTAGE** | parity in both families, GM_f ≤ 1/1.2 (≈ 0.833) **and** the upper 95 % bound < 1.0 in **both** families |
| **NO PRACTICAL DIFFERENCE** | parity in both families, and 0.833 < GM_f < 1.2 in both families |
| **MIXED** | any other combination (reported per family) |
| **INVALID** | any hard stop (§4) or a parity failure of the per-system arm |

- The scientific claim supported by each outcome is fixed now. "Certificate reuse offers no practical speed advantage over an optimized per-system certified verifier on these families" is supported by **PER-SYSTEM ADVANTAGE** or **NO PRACTICAL DIFFERENCE**. Any other claim needs **REUSE ADVANTAGE**.
- **No re-analysis with other thresholds, units or arms** is allowed after data are generated. Any deviation is reported as a deviation.

## 8. Primary arms and expected direction (from Phase 10A development data; fixed before any held-out data)

**Rule.** For each family, the primary per-system arm is the faster of {B = RPS-opt σ 0.5, B9 = RPS-opt σ 0.9}, and the primary reuse arm is the faster **target-valid** of {D = M2A-EFT σ 0.9, D2 = M2A-EFT σ 0.5}. Both are chosen on the development S3 diagnostic, the size of the primary statistic. Choosing the faster reuse variant favours reuse.

| Family | Primary per-system | Primary reuse | Development S3 ratio (per-system / reuse) | Development S2 ratio (best / best) |
|---|---|---|---|---|
| E3-TS | **B9** (3.925 s) | **D** (4.752 s; D2 5.637 s) | 0.827 [0.731, 0.936] | 0.904 [0.814, 1.004] |
| D3-TS | **B** (1.171 s) | **D2** (1.365 s; D 1.472 s) | 0.858 (one seed) | 0.786 [0.657, 0.940] |

- **Expected direction:** the per-system method is faster, by roughly 10–20 %. Under §7 the development data would fall in NO PRACTICAL DIFFERENCE (E3-TS S2) or PER-SYSTEM ADVANTAGE / MIXED depending on interval widths.
- Frozen M2-A (C) is excluded as a primary arm because it is **not target-valid** at E3-TS S3 (50/63 on development data).

## 9. Validation (all mandatory)
- Full replay of every record (verify10a; CHOLMOD cross-engine re-proof of every distinct Rump anchor; bitwise EFT recomputation).
- An exact-residual sample (≥ 4 records per new arm per size).
- Arb λ proofs on ≥ 2 matrices per family (S2).
- Input-manifest check.
- Determinism report across repeats.

## 10. What must not happen
- Reuse of seeds 100–104 as untouched evidence.
- Any code change after freezing (aggregate hash checked at start).
- Tuning of any arm after held-out generation.
- S4.
- Re-running held-out cells to obtain a different result. A crash may be repeated once, and the repetition is reported.
