"""Generate and freeze SIMP3D windows (one complete pass per process; run twice and compare bit for bit).
For each instance (volume fraction v_f): run 61 OC iterations; window 'early' = iterations 1-21, window 'late' =
iterations 41-61. Writes coefficient vectors c_t = E(x_t)/N as exact hex, the density/compliance history, and a manifest
with SHA-256 of every matrix (canonical float assembly), rhs and coefficient vector.
Usage: python gen_simp.py <out_dir> <plan: pilot|eval>"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simp_family as SF                                             # noqa: E402

ITERS = 61
WINDOWS = {'early': 1, 'late': 41}
PLANS = {
    # development pilot (decides certifiability settings; never part of the evaluation)
    'pilot': dict(size='S2', emin=[1e-9, 1e-3], instances={'900': 0.50}),
    # evaluation instances (frozen in SIMP_PROTOCOL.md before generation)
    'eval': dict(size='S3', emin=[None], instances={'301': 0.20, '302': 0.25, '303': 0.30, '304': 0.35, '305': 0.40}),
}


def main():
    out = Path(sys.argv[1]); plan = PLANS[sys.argv[2]]
    emins = plan['emin']
    if emins == [None]:
        emins = [float(json.loads((HERE / 'simp_settings.json').read_text())['emin'])]
    out.mkdir(parents=True, exist_ok=True)
    N = SF.SIZES[plan['size']]; man = {'size': plan['size'], 'N': N, 'iters': ITERS, 'windows': WINDOWS, 'instances': {}}
    t0 = time.perf_counter()
    for emin in emins:
        for inst, vf in plan['instances'].items():
            key = inst if len(emins) == 1 else f'{inst}_emin{emin:.0e}'
            xs, hist = SF.generate(N, vf, ITERS, emin)
            entry = dict(vf=vf, emin=emin, history=hist, windows={})
            for w, k0 in WINDOWS.items():
                cs = [((emin + xs[k - 1] ** 3 * (1.0 - emin)) / N) for k in range(k0, k0 + 21)]
                fn = f'coeff_hex_{key}_{w}.json'
                (out / fn).write_text(json.dumps([[float(c).hex() for c in row] for row in cs]))
                entry['windows'][w] = dict(k0=k0, coeff_hex_file=fn,
                                           coeff_min=[float(c.min()) for c in cs], coeff_max=[float(c.max()) for c in cs],
                                           step_ratio_min=[float((cs[i + 1] / cs[i]).min()) for i in range(20)],
                                           step_ratio_max=[float((cs[i + 1] / cs[i]).max()) for i in range(20)])
            man['instances'][key] = entry
            print(key, vf, emin, 'compliance', round(hist[0]['compliance'], 6), '->', round(hist[-1]['compliance'], 6), flush=True)
    (out / 'simp_manifest.json').write_text(json.dumps(man, indent=0))
    # matrix / rhs / coefficient hashes through the Family3D-compatible view
    hashes = {}
    for key, entry in man['instances'].items():
        for w in WINDOWS:
            fam = SF.SimpFamily(out, plan['size'], key, w)
            rows = []
            for t in range(fam.T + 1):
                A, b, meta = fam.get(t); A = SF.R6.ensure_canonical(A)
                rows.append(dict(t=t, matrix_sha256=SF.v.matrix_sha256(A), rhs_sha256=SF.v.vec_sha256(b),
                                 coeff_sha256=meta['coeff_sha256'], n=int(A.shape[0]), nnz=int(A.nnz)))
            hashes[f'{key}|{w}'] = rows
    (out / 'simp_hashes.json').write_text(json.dumps(hashes, indent=0))
    (out / 'generation.json').write_text(json.dumps(dict(pid=os.getpid(), seconds=time.perf_counter() - t0,
                                                         at=time.strftime('%Y-%m-%d %H:%M:%S'))))
    print('DONE', round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()
