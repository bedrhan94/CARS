"""Revision 3: input manifest of the large-size experiment (LARGE_PROTOCOL.md), written before the protocol is frozen.
E3-TS geometry with N = 32 (generator size label S5, n = 104 544), development split (parameters nu = 0.3, alpha = 0.7),
development seeds 0, 1, 2; for every step t = 0..20 the SHA-256 of the canonical matrix, the right-hand side and the
coefficient vector (frozen hashing functions). Each sequence is generated twice in this process and must hash identically.
Writes inputs_manifest_large.json."""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b'))
import run_cell10b as RC                                             # noqa: E402
R6 = RC.R6; v = RC.v
from cars_m2.families import Family3D                                # noqa: E402

SEEDS = (0, 1, 2)


def rows_of(seed):
    fam = Family3D('E3-TS', 'S5', seed=seed, split='dev')
    rows = []
    for t in range(fam.T + 1):
        A, b, meta = fam.get(t); A = R6.ensure_canonical(A)
        rows.append(dict(t=t, n=int(A.shape[0]), nnz=int(A.nnz), matrix_sha256=v.matrix_sha256(A), rhs_sha256=v.vec_sha256(b),
                         coeff_sha256=meta['coeff_sha256']))
    return rows


if __name__ == '__main__':
    t0 = time.time(); man = {}
    for s in SEEDS:
        r1 = rows_of(s); r2 = rows_of(s)
        man[f'E3-TS|S5|{s}'] = dict(rows=r1, identical_regeneration=r1 == r2)
        print(s, r1[0]['n'], r1[0]['nnz'], r1 == r2, flush=True)
    out = dict(manifest=man, all_identical=all(m['identical_regeneration'] for m in man.values()),
               generated_at=time.strftime('%Y-%m-%d %H:%M:%S'), seconds=time.time() - t0)
    (HERE / 'inputs_manifest_large.json').write_text(json.dumps(out, indent=1))
    print('all_identical', out['all_identical'])
