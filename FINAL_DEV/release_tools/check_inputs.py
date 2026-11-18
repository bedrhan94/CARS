"""Rebuild every frozen held-out matrix from the exact-hex coefficients (FrozenFamily) and compare the SHA-256 of the
matrix, rhs and coefficient vector with the frozen manifest. Usage: python FINAL_DEV/release_tools/check_inputs.py"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, platform, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
sys.path.insert(0, str(CARS / 'PHASE6_DEV' / 'lean6')); sys.path.insert(0, str(HERE))
import runner6                                                        # noqa: E402,F401  (sets up the frozen import paths)
from frozen_family import FrozenFamily                               # noqa: E402


def main():
    t0 = time.time(); inp = CARS / 'FINAL_DEV/phase10b/inputs/pass_a'
    man = json.loads((inp.parent / 'inputs_manifest10b.json').read_text())['manifest']
    res = {}
    for key in sorted(man):
        fam, size, seed = key.split('|')
        ff = FrozenFamily(inp, fam, size, int(seed))
        for t in range(ff.T + 1):
            ff.get(t)
        res[key] = dict(steps=ff.T + 1, mismatches=ff.mismatches)
        print(key, 'OK' if not ff.mismatches else f'MISMATCH {ff.mismatches[:3]}', flush=True)
    bad = {k: v for k, v in res.items() if v['mismatches']}
    out = dict(platform=platform.platform(), python=platform.python_version(), checked=len(res) * 21,
               keys_with_mismatch=len(bad), details=bad, seconds=time.time() - t0)
    (HERE / 'check_inputs_result.json').write_text(json.dumps(out, indent=1))
    print('CHECKED', out['checked'], 'MISMATCHED_KEYS', len(bad))


if __name__ == '__main__':
    main()
