"""Phase 10A tamper tests: every modified record must be REJECTED by verify10a.
Uses the S0 smoke records (E3-TS, dev seed 0) of the new arms. Includes a SELF-CONSISTENT optimistic EFT certificate
(R from a deliberately wrong bound, eps and target recomputed consistently): replay must reject it.
Usage: python tamper10a.py <cell_dir>  -> <cell_dir>/tamper10a.json"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import copy, json, shutil
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CARS / 'PHASE6_DEV' / 'lean6'))
import verify10a as VV                                               # noqa: E402
import eft_residual as EFT                                           # noqa: E402
from replay6 import MemoFamily                                       # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402
from cars import _reference_m0 as v                                  # noqa: E402
from cars.verify import file_sha256                                  # noqa: E402


def h(x):
    return float(x).hex()


def main():
    cell = Path(sys.argv[1])
    fam = MemoFamily(Family3D('E3-TS', 'S0', seed=0, split='dev'))
    res = []
    for arm in ('B', 'B9', 'D', 'D2', 'B1'):
        f = next((cell / arm).glob('*.jsonl'))
        recs = [json.loads(l) for l in f.read_text().splitlines()]
        r0 = next(r for r in recs if r['status'] == 'CERTIFIED' and r['t'] >= 3)
        base_dir = cell / arm
        st, er = VV.verify_record10a(r0, fam, base_dir, exact_n=0)
        res.append(dict(arm=arm, test='untampered', status=st, expect='VERIFIED', ok=(st == 'VERIFIED')))
        cases = []
        r = copy.deepcopy(r0); r['lambda_lo'] = h(float.fromhex(r['lambda_lo']) * 1.01); cases.append(('lambda_lo_up', r))
        r = copy.deepcopy(r0); r['R_hi'] = h(float.fromhex(r['R_hi']) * 0.5); cases.append(('R_down', r))
        r = copy.deepcopy(r0); r['eps'] = h(float.fromhex(r['eps']) * 0.5); cases.append(('eps_down', r))
        r = copy.deepcopy(r0); r['target_met'] = not r['target_met']; cases.append(('target_flip', r))
        r = copy.deepcopy(r0); r['matrix_sha256'] = '0' * 64; cases.append(('matrix_sha', r))
        r = copy.deepcopy(r0); r['lambda_chain'][0]['sigma'] = h(float.fromhex(r['lambda_chain'][0]['sigma']) * 1.5)
        cases.append(('anchor_sigma_up', r))
        r = copy.deepcopy(r0); r['residual_kind'] = 'frozen-fallback'; r['R_hi'] = h(float.fromhex(r['R_hi']) * 0.5)
        r['eps'] = h(v.up(float.fromhex(r['R_hi']) / float.fromhex(r['lambda_lo']))); cases.append(('fallback_relabel_R_down', r))
        if r0['lambda_chain'][0].get('type') == 'm2a':
            r = copy.deepcopy(r0); r['lambda_chain'][0]['theta'] = h(float.fromhex(r['lambda_chain'][0]['theta']) * 1.001); cases.append(('theta_up', r))
            r = copy.deepcopy(r0); r['lambda_chain'][0]['zeta_t'] = h(0.0); cases.append(('zeta_t_zero', r))
        # self-consistent optimistic certificate: R from a wrong (mutant) bound, eps/target recomputed
        A, b, _ = fam.get(int(r0['t'])); x = np.load(base_dir / r0['solution_file'])
        for m in ('float_residual', 'no_product_errors', 'no_gamma2'):
            Rm, _ = EFT.eft_residual_bound(A, b, x, mutant=m)
            r = copy.deepcopy(r0); r['R_hi'] = h(Rm); eps = float(v.up(Rm / float.fromhex(r['lambda_lo'])))
            r['eps'] = h(eps); r['target_met'] = bool(eps <= float.fromhex(r['target']))
            cases.append((f'self_consistent_mutant_{m}' + ('' if Rm != float.fromhex(r0['R_hi']) else '_IDENTICAL_R'), r))
        # tampered solution vectors with consistent hashes: (i) 1 ulp in one entry, (ii) a large perturbation
        tdir = base_dir / '_tamper'; tdir.mkdir(parents=True, exist_ok=True)
        for tag, x2 in (('1ulp', x.copy()), ('large', x.copy())):
            if tag == '1ulp':
                x2[0] = np.nextafter(x2[0], np.inf)
            else:
                x2[0] = x2[0] + 1e-3 * float(np.max(np.abs(x)))
            r = copy.deepcopy(r0)
            fp = tdir / f'x_tampered_{tag}.npy'; np.save(fp, x2, allow_pickle=False)
            r['solution_file'] = str(Path('_tamper') / f'x_tampered_{tag}.npy'); r['solution_sha256'] = file_sha256(fp)
            r['x_sha256'] = v.vec_sha256(x2)
            cases.append((f'solution_changed_{tag}_consistent_hashes', r))
        for name, rr in cases:
            try:
                st, er = VV.verify_record10a(rr, fam, base_dir, exact_n=0)
            except Exception as e:
                st, er = 'EXCEPTION', [repr(e)[:120]]
            expect = 'VERIFIED' if name.endswith('_IDENTICAL_R') else 'REJECTED'
            ok = (st == expect)
            if name.startswith('solution_changed_1ulp') and rr.get('residual_kind') != 'eft' and st == 'VERIFIED':
                # a frozen-kind bound with slack may remain TRUE for a 1-ulp change: accept only if exactly valid
                from cars.verify import exact_residual2
                from fractions import Fraction
                x2 = np.load(base_dir / rr['solution_file']); A_, b_, _ = fam.get(int(rr['t']))
                ok = Fraction(float.fromhex(rr['R_hi'])) ** 2 >= exact_residual2(A_, b_, x2)
                expect = 'VERIFIED-only-if-exactly-valid'
            res.append(dict(arm=arm, test=name, status=st, errors=er[:3], expect=expect, ok=bool(ok),
                            residual_kind=rr.get('residual_kind')))
    for arm in ('B', 'B9', 'D', 'D2', 'B1'):
        shutil.rmtree(cell / arm / '_tamper', ignore_errors=True)
    out = dict(tests=res, total=len(res), passed=sum(r['ok'] for r in res))
    (cell / 'tamper10a.json').write_text(json.dumps(out, indent=1))
    for r in res:
        print(('PASS ' if r['ok'] else 'FAIL ') + r['arm'], r['test'], r['status'], r.get('errors', ''))
    print('TAMPER', out['passed'], '/', out['total'])


if __name__ == '__main__':
    main()
