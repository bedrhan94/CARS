"""Unit test of the frozen decision logic (analyze10b.primary) on synthetic cost ratios (no real data).
Each case sets the per-(seed, repeat) ratio of the primary pair and checks the per-family category and joint outcome."""
import math, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze10b as AN                                              # noqa: E402
import common10b as K                                                # noqa: E402


def cells_for(ratios, parity_fail=None):
    cells = {}
    for fam, (P, Q) in K.PRIMARY.items():
        for i, seed in enumerate(K.HELDOUT_SEEDS):
            for rep in (0, 1, 2):
                r = ratios[fam][i] * (1 + 0.01 * (rep - 1))
                arm = lambda a, cost: dict(arm=a, measured_s=cost, parity=not (parity_fail and parity_fail == (fam, a, seed, rep)))
                cells[('ST', 'S3', fam, seed, rep)] = {P: arm(P, r), Q: arm(Q, 1.0)}
    return cells


CASES = [
    ('per-system advantage', {'E3-TS': [0.70, 0.72, 0.75, 0.71, 0.73], 'D3-TS': [0.6, 0.65, 0.62, 0.7, 0.66]}, None,
     ('PER-SYSTEM ADVANTAGE', 'PER-SYSTEM ADVANTAGE'), 'PER-SYSTEM ADVANTAGE'),
    ('reuse advantage', {'E3-TS': [1.4, 1.5, 1.45, 1.38, 1.42], 'D3-TS': [1.3, 1.35, 1.32, 1.4, 1.29]}, None,
     ('REUSE ADVANTAGE', 'REUSE ADVANTAGE'), 'REUSE ADVANTAGE'),
    ('small directional', {'E3-TS': [0.88, 0.9, 0.89, 0.87, 0.9], 'D3-TS': [0.88, 0.9, 0.89, 0.87, 0.9]}, None,
     ('NO ESTABLISHED PRACTICAL ADVANTAGE',) * 2, 'NO ESTABLISHED PRACTICAL ADVANTAGE'),
    ('GM below 0.833 but CI includes 1', {'E3-TS': [0.5, 1.2, 0.6, 1.1, 0.7], 'D3-TS': [0.6, 0.65, 0.62, 0.7, 0.66]}, None,
     ('NO ESTABLISHED PRACTICAL ADVANTAGE', 'PER-SYSTEM ADVANTAGE'), 'MIXED'),
    ('reuse parity failure', {'E3-TS': [0.7] * 5, 'D3-TS': [0.6] * 5}, ('E3-TS', 'D', 202, 1),
     ('REUSE ACCURACY-INFERIOR', 'PER-SYSTEM ADVANTAGE'), 'MIXED'),
    ('per-system parity failure', {'E3-TS': [0.7] * 5, 'D3-TS': [0.6] * 5}, ('D3-TS', 'B', 200, 0),
     ('PER-SYSTEM ADVANTAGE', 'INVALID'), 'INVALID'),
]


def main():
    ok = True
    for name, ratios, pf, cats, joint in CASES:
        out, j = AN.primary(cells_for(ratios, pf))
        got = (out['E3-TS']['category'], out['D3-TS']['category'])
        good = got == cats and j == joint
        ok &= good
        print('PASS' if good else 'FAIL', name, got, j, {f: (round(v['stats']['gm'], 3), [round(x, 3) for x in v['stats']['ci95']],
                                                           v['descriptors']) for f, v in out.items()})
    # equivalence descriptor: tight ratios around 0.95 -> equivalent; wide -> not
    out, _ = AN.primary(cells_for({'E3-TS': [0.94, 0.95, 0.96, 0.95, 0.94], 'D3-TS': [0.5, 1.5, 0.7, 1.3, 0.9]}))
    eq = (out['E3-TS']['descriptors']['practically_equivalent_tost90'], out['D3-TS']['descriptors']['practically_equivalent_tost90'])
    ok &= eq == (True, False)
    print('PASS' if eq == (True, False) else 'FAIL', 'equivalence descriptor', eq)
    print('ALL_PASS' if ok else 'SOME_FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
