"""Phase 10A input freeze: SHA-256 of every development matrix, right-hand side and coefficient vector used in this
phase (E3-TS / D3-TS, sizes S0-S3, dev seeds 0-2, t = 0..20; SAFE-S / SAFE-P S0 seed 0), generated BEFORE any timed run.
Every run record is later checked against this manifest (analysis). Held-out seeds are never generated.
Usage: python freeze_inputs10a.py -> inputs_manifest10a.json"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
CARS = HERE.parent
sys.path.insert(0, str(CARS / 'PHASE6_DEV' / 'lean6'))
import runner6 as R6                       # noqa: E402
from cars import _reference_m0 as v        # noqa: E402
from cars_m2.families import Family3D      # noqa: E402


def main():
    t0 = time.perf_counter(); man = {}
    plan = [(f, s, seed) for f in ('E3-TS', 'D3-TS') for s in ('S0', 'S1', 'S2', 'S3') for seed in (0, 1, 2)]
    plan += [('SAFE-S', 'S0', 0), ('SAFE-P', 'S0', 0)]
    for fid, size, seed in plan:
        assert seed in (0, 1, 2)
        fam = Family3D(fid, size, seed=seed, split='dev')
        rows = []
        for t in range(fam.T + 1):
            A, b, meta = fam.get(t)
            A = R6.ensure_canonical(A)
            rows.append(dict(t=t, matrix_sha256=v.matrix_sha256(A), rhs_sha256=v.vec_sha256(b),
                             coeff_sha256=meta.get('coeff_sha256'), n=int(A.shape[0]), nnz=int(A.nnz)))
        man[f'{fid}|{size}|{seed}'] = rows
        print(fid, size, seed, rows[0]['n'], flush=True)
    out = dict(manifest=man, seconds=time.perf_counter() - t0, note='dev seeds 0-2 only; generator PHASE3B_DEV Family3D (frozen)')
    (HERE / 'inputs_manifest10a.json').write_text(json.dumps(out, indent=0))
    print('DONE', round(out['seconds'], 1))


if __name__ == '__main__':
    main()
