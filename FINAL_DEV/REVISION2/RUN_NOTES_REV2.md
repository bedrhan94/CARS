# Revision 2: run notes (post hoc experiments and checks)

**Written for** the authors and a future auditor. Times are local workstation time (16 November 2026).

## Compiled-kernel experiment (`compiled/`)

- 00:38:21 protocol frozen (`compiled/PROTOCOL_FREEZE.json`) after the equivalence test (`test_kernels.json`, ALL_IDENTICAL true) and a development smoke test (E3-TS S1, seed 0).
- 00:38:30 plan started; no other Python process was running (plan log). 60 cells completed, 46.7 of the 90 budget minutes; every cell replayed, all records VERIFIED, no input or manifest mismatch.
- The first version of the equivalence test reported 32 "failures" that were defects of the test, not of the kernels. In each, both implementations accepted a one-ulp change of one coefficient that rounding had absorbed. The test was corrected before the freeze, to require identical answers and to confirm any acceptance with the frozen full assembly, and then re-run.

## SIMP completion with the corrected fallback (`fallback/`)

- 00:38:21 protocol frozen (`fallback/PROTOCOL_FREEZE.json`) after the pre-freeze checks (`check_fallback.json`, ALL_OK true) and a development smoke test on the SIMP pilot instance.
- 17:01:42 plan started (after the compiled experiment had finished). 18:30:40 plan done: 30 of 30 cells, 45.2 of 90 budget minutes, every cell replayed immediately, all records VERIFIED.
- **Concurrency notes (disclosed).**
  - The plan's start-up check found two other Python processes. They were the analysis script of the compiled experiment, launched in the same moment: one interpreter plus its launcher, which ran for about two seconds while the first SIMP cell was still verifying its inputs, before its warm-up and timed arms.
  - During the replay of cell 305_early_r0 and possibly the beginning of cell 305_late_r0, a directory-size scan of the compiled-experiment output (disk metadata only, about two minutes) ran on another core.
  - No other benchmark, Arb or exact-arithmetic process ran during the plan.

## Post hoc checks (`checks/`)

Run after the SIMP plan had finished (18:30), so that they could not perturb its timings:
- `rump_constant_share.py`
- `eft_exact_abs.py` (reproduces the frozen suite exactly)
- `negctl_coeff.py` (ALL_OK true)
- `exact_eft_sample.py`

The first run of `exact_eft_sample.py` was stopped and repeated. Its solution-file check compared the record's hash with a hash of the raw vector bytes, but the records hash the `.npy` file, so every comparison failed. The check was corrected before the run used for the results. The exact residuals themselves do not depend on this check.
