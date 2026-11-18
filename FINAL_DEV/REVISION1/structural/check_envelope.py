"""Revision 1: numerical equivalence of the structural-envelope representation (untimed, run once before the cells).

For every matrix A_t of every sequence used in the structural experiment (frozen held-out inputs, verified against the
frozen manifest), with E the structural envelope of the mesh (struct_session.envelope):
  1. the stored pattern of A_t is contained in E (the frozen PardisoSession._embed succeeds);
  2. the value array that the frozen session hands to PARDISO (A_t.data at the embedded positions, explicit zeros
     elsewhere) reconstructs A_t exactly: as a CSR matrix on the pattern E with explicit zeros removed it is bitwise
     identical to A_t (indptr, indices, data);
  3. A_t itself stores no explicit zero, so the CSR hash recorded in each certificate refers to the same matrix.
The certificates themselves always refer to the original stored matrix A_t (its SHA-256 is in every record) and are
re-verified against it by the frozen replay; the envelope only changes how PARDISO stores the factorized matrix.
Usage: python check_envelope.py -> check_envelope.json"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b'))
import common10b as K                                                # noqa: E402
import run_cell10b as RC                                             # noqa: E402
R6 = RC.R6
from cars import _reference_m0 as v                                  # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402
sys.path.insert(0, str(HERE))
import struct_session as SS                                          # noqa: E402

CELLS = [('E3-TS', 'S2'), ('E3-TS', 'S3'), ('D3-TS', 'S2')]


def main():
    t0 = time.time(); K.check_freeze(); man = K.check_inputs_frozen()['manifest']
    res = {}; ok_all = True
    for fam_name, size in CELLS:
        for seed in K.HELDOUT_SEEDS:
            fam = Family3D(fam_name, size, seed=seed, split='heldout')
            bad_in = RC.verify_inputs(fam, man[f'{fam_name}|{size}|{seed}']['rows'])
            ip, ix = SS.envelope(fam.mesh)
            probe = R6.PardisoSession.__new__(R6.PardisoSession)          # frozen _embed on the envelope pattern
            probe.full_indptr, probe.full_indices, probe.n = ip, ix, fam.n
            steps = []
            for t in range(fam.T + 1):
                A, b, meta = fam.get(t); A = R6.ensure_canonical(A)
                pos = R6.PardisoSession._embed(probe, A)
                contained = pos is not None
                exact = False; stored_zeros = int(np.sum(A.data == 0.0))
                if contained:
                    full = np.zeros(ix.size); full[pos] = A.data
                    M = sp.csr_matrix((full, ix.copy(), ip.copy()), shape=A.shape); M.eliminate_zeros(); M.sort_indices()
                    exact = (np.array_equal(M.indptr, A.indptr) and np.array_equal(M.indices, A.indices)
                             and np.array_equal(M.data.view(np.uint64), A.data.view(np.uint64)))
                steps.append(dict(t=t, nnz=int(A.nnz), contained=contained, exact=bool(exact), stored_zeros=stored_zeros,
                                  explicit_zeros_added=int(ix.size - A.nnz)))
                ok_all &= contained and exact and stored_zeros == 0
            res[f'{fam_name}|{size}|{seed}'] = dict(envelope_nnz=int(ix.size), input_mismatches=bad_in,
                                                    nnz_range=[min(s['nnz'] for s in steps), max(s['nnz'] for s in steps)],
                                                    all_contained=all(s['contained'] for s in steps),
                                                    all_exact=all(s['exact'] for s in steps),
                                                    max_explicit_zeros=max(s['explicit_zeros_added'] for s in steps), steps=steps)
            ok_all &= not bad_in
            print(fam_name, size, seed, 'envelope', ix.size, 'nnz', res[f'{fam_name}|{size}|{seed}']['nnz_range'],
                  'ok', res[f'{fam_name}|{size}|{seed}']['all_exact'], flush=True)
    out = dict(ALL_EQUIVALENT=bool(ok_all), seconds=time.time() - t0, cells=res)
    (HERE / 'check_envelope.json').write_text(json.dumps(out, indent=1))
    print('ALL_EQUIVALENT', ok_all)


if __name__ == '__main__':
    main()
