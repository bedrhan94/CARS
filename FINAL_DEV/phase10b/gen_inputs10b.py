"""Phase 10B input generation (one complete pass per process).

For every (family, size, seed, t) of the plan, the frozen generator (PHASE3B_DEV Family3D, split 'heldout' for seeds
200-204) is run and the following are written to <out_dir>:
  - coefficient vectors c_t as exact hexadecimal strings (JSON) and as little-endian binary (.npy),
  - the right-hand side as exact hexadecimal strings (once per family/size; checked identical for every t and seed),
  - SHA-256 of the canonical matrix, the rhs and the coefficients, plus n and nnz.
Two passes in separate processes must agree bit for bit (compare_inputs10b.py) before any benchmark.
Usage: python gen_inputs10b.py <out_dir> heldout|devtest"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common10b as K                                               # noqa: E402
sys.path.insert(0, str(K.CARS / 'PHASE6_DEV' / 'lean6'))
import runner6 as R6                                                # noqa: E402
from cars import _reference_m0 as v                                 # noqa: E402
from cars_m2.families import Family3D                               # noqa: E402


def plan(kind):
    if kind == 'heldout':
        return [(f, s, seed, 'heldout') for f in K.FAMILIES for s in K.SIZES for seed in K.HELDOUT_SEEDS]
    if kind == 'devtest':                       # harness test of the generator path on development data only
        return [(f, 'S0', 0, 'dev') for f in K.FAMILIES]
    raise SystemExit('kind must be heldout or devtest')


def main():
    out = Path(sys.argv[1]); kind = sys.argv[2]
    if kind == 'heldout':
        K.check_freeze()
    out.mkdir(parents=True, exist_ok=True); (out / 'coeff_npy').mkdir(exist_ok=True)
    t0 = time.perf_counter(); man = {}; rhs_hex = {}
    for fid, size, seed, split in plan(kind):
        K.seed_guard(seed, split)
        fam = Family3D(fid, size, seed=seed, split=split)
        rows = []; chex = []
        for t in range(fam.T + 1):
            A, b, meta = fam.get(t)
            A = R6.ensure_canonical(A)
            c = np.ascontiguousarray(meta['c'], dtype='<f8')
            np.save(out / 'coeff_npy' / f'{fid}_{size}_{split}_{seed}_t{t}.npy', c, allow_pickle=False)
            chex.append([float(x).hex() for x in c])
            key_b = f'{fid}|{size}'
            bh = [float(x).hex() for x in b]
            if key_b not in rhs_hex:
                rhs_hex[key_b] = bh
            elif rhs_hex[key_b] != bh:
                raise SystemExit(f'RHS differs across t/seed for {key_b}')
            rows.append(dict(t=t, n=int(A.shape[0]), nnz=int(A.nnz), matrix_sha256=v.matrix_sha256(A),
                             rhs_sha256=v.vec_sha256(b), coeff_sha256=meta.get('coeff_sha256'),
                             coeff_npy_sha256=K.sha256_file(out / 'coeff_npy' / f'{fid}_{size}_{split}_{seed}_t{t}.npy'),
                             positive_offdiag=int(meta['positive_offdiag'])))
        man[f'{fid}|{size}|{seed}'] = dict(split=split, N=int(fam.N), spec={k: fam.spec[k] for k in sorted(fam.spec)},
                                            element_data_sha256=fam.factor_hash, rows=rows)
        (out / f'coeff_hex_{fid}_{size}_{split}_{seed}.json').write_text(json.dumps(chex))
        print(fid, size, seed, split, rows[0]['n'], rows[0]['nnz'], flush=True)
    (out / 'rhs_hex.json').write_text(json.dumps(rhs_hex))
    res = dict(kind=kind, manifest=man, generated_at=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(),
               seconds=time.perf_counter() - t0, numpy=np.__version__)
    (out / 'generation.json').write_text(json.dumps(res, indent=0))
    print('DONE', round(res['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
