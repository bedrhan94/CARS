# SIMP3D supplementary evaluation: protocol (written before any SIMP sequence was generated)

**Purpose.** Test whether the Phase 10B conclusion depends on the synthetic coefficient path of E3-TS and D3-TS. The test uses stiffness sequences whose coefficient evolution comes from an independently specified process: 3D SIMP compliance minimisation.

**This evaluation is supplementary.**
- It is not part of the Phase 10B pre-registration and has no decision role for the confirmatory result.
- It uses the frozen Phase 10A arms unchanged (code aggregate `f305dda5…`); only the family wrapper and the cell runner are new.

## 1. Problem

| Item | Setting |
|---|---|
| Objective | minimise the compliance f·u subject to K(x)u = f, mean(x) = v_f, 0 ≤ x_e ≤ 1 |
| Material interpolation | E_e = E_min + x_e³(1 − E_min), E_0 = 1 |
| Sensitivity filter | radius r_min = 1.5 element widths (Sigmund 2001) |
| Update | optimality criteria, move limit 0.2, damping exponent 0.5, bisection on the multiplier to relative width 10⁻⁶; starting from the uniform density x = v_f |
| Discretisation | E3 mesh N³ on the unit cube, clamped at x = 0; element factor Z_ref_elas_h1 (ν = 0.3); frozen float assembly; coefficient c_e = E_e/N |
| Load | the uniform downward body load of the E3 families (design-independent) |
| Iterations | 61 |

**Windows** (21 consecutive iterations each = one sequence):
- *early*: iterations 1–21;
- *late*: iterations 41–61.

**Instances (units):**
- **Development pilot:** v_f = 0.50, N = 12 (S2). Never part of the evaluation.
- **Evaluation:** v_f ∈ {0.20, 0.25, 0.30, 0.35, 0.40} (instances 301–305), N = 16 (S3 size).

## 2. Settings fixed by rule from the pilot (rules fixed now)

1. **E_min.** E_min = 10⁻⁹ (the default of the reference codes) if arm A certifies all 21 steps of both pilot windows at that value. Otherwise E_min = 10⁻³, and the failure at 10⁻⁹ is reported as a finding on the certifiability of high-contrast SIMP systems.
2. **Target.** τ = 10⁻⁸ (absolute, as in Phase 10B) if arm A is target-met at all pilot steps. Otherwise, a scale-aware target per instance: τ_i = 10⁻⁸ · ‖u_i‖₂, where u_i is the displacement of the first iteration of the window computed by the optimiser. This corresponds to a relative accuracy of 10⁻⁸ and is written to `simp_settings.json` as exact hex before the evaluation cells run.
3. Nothing else may be changed after the pilot.

## 3. Execution

- **Inputs.** Generated twice in separate processes and compared bit for bit (all coefficient hex files and all matrix, rhs and coefficient SHA-256). Every cell re-checks them before its warm-up.
- **Cells.** One process per (instance, window, repeat): ST, pinned to CPU 2, all seven arms, rotated order, warm-up as in Phase 10B, 3 repeats. Every cell is fully replayed with the frozen verifier.
- **Budget:** ≤ 1.5 h for the evaluation cells and replays; 30 min per process; 16 GiB per process.

## 4. Analysis (descriptive)

- Per window, pair (B9 vs D) and (B vs D2): the Phase 10B primary pairs for E3-TS and D3-TS.
- Unit = instance; median over repeats; GM over the 5 instances; 95 % t-interval (df = 4).
- All seven arms' costs, parity, factorisations, refreshes, binding and PCG costs.
- **No decision category is applied.** Statements are descriptive and labelled as supplementary.
