"""CARS Phase 10A runner for the NEW comparison arms (development data only). The frozen reference arms are run by the
unchanged Phase 6 runner (PHASE6_DEV/lean6/runner6.py: run_sequence_5): A = RPS|P1F, C = RPS|M2A.

New arms (all share ONE residual strategy, one stopping rule and one target-restoration rule):
  RPS-EFT      (B1, ablation)  per-system Rump + PARDISO (symbolic reuse); per-step lambda estimate; sigma = 0.9 l~
  RPS-OPT      (B)             per-system Rump + PARDISO (symbolic reuse); CARRIED shift: the previous sigma is tried
                               first; a new estimate only when Rump's Cholesky fails (fail closed) or when the target is
                               missed; sigma = 0.5 l~ on every (re)estimate
  RPS-OPT-S09  (B9, symmetric)  RPS-OPT with sigma = 0.9 l~ (completes the per-system grid {carry} x {0.5, 0.9}; added
                               after the 6-arm S3 diagnostic showed sigma = 0.5 forcing the EFT bound at E3 S3)
  M2A-EFT      (D)             frozen M2-A logic (lean bit-identity binding, min-theta transfer, theta_min refresh, PCG-
                               iteration trigger), anchors sigma = 0.9 l~ (frozen policy)
  M2A-EFT-S05  (D2, symmetric) as D with anchors sigma = 0.5 l~ (the same sigma policy as RPS-OPT)
Shared strategy ('frozen-first cascade', fixed after the V1 cost measurement, before any benchmark):
  * stopping: PCG (scipy cg, frozen settings) with the FROZEN rule rtol = min(1e-12, max(1e-16, 0.5 tau lambda_lo/||b||))
  * certificate: eps = up(R / lambda_lo) with R the frozen bound; if it misses the target, R = the rigorous EFT bound
    (PHASE10A_DEV/eft/eft_residual.py; frozen bound as automatic fallback when the EFT preconditions fail); if the
    target is still missed, continue PCG from the iterate with rtol / 10 (at most 4 rounds)
  * restoration: if the target is still missed, the method re-anchors once at this step (per-system: new estimate and
    factorization; M2-A: anchor refresh) and repeats the solve; every extra cost is charged
Accounting: the frozen components (research/17 s5); 'io' excluded from both totals; counterfactual = measured -
anchor_V - (estimate for per-system methods), exactly as in runner6."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
for p in (CARS / 'PHASE6_DEV' / 'lean6', HERE.parent / 'eft'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import runner6 as R6                                         # noqa: E402  frozen Phase 6 runner (helpers only)
from cars import _reference_m0 as v                          # noqa: E402
from cars.verify import file_sha256                          # noqa: E402
from cars_m2.runner3b import _estimate                       # noqa: E402  frozen estimator
from m2_ref import assembly_discrepancy_rowsum, m2a_transfer # noqa: E402
from fem3d_ref import assemble_float                         # noqa: E402
from fastbind import FastAssembler                           # noqa: E402
import eft_residual as EFT                                   # noqa: E402

NEW = {'RPS-EFT': dict(kind='per', frac=0.9, carry=False),
       'RPS-OPT': dict(kind='per', frac=0.5, carry=True),
       'RPS-OPT-S09': dict(kind='per', frac=0.9, carry=True),
       'M2A-EFT': dict(kind='m2a', frac=0.9),
       'M2A-EFT-S05': dict(kind='m2a', frac=0.5)}
MAX_ROUNDS = 4
FORCE_EFT = os.environ.get('CARS10A_FORCE_EFT') == '1'     # VALIDATION ONLY (stage V1E): always try the EFT bound


def make_anchor_frac(A, lam_tilde, frac, tm):
    """The frozen retry rule with a configurable first shift: sigma = frac * l~, on failure sigma /= 5, at most 3
    retries; every attempt is charged."""
    sigma = frac * lam_tilde; reasons = []
    for _ in range(4):
        try:
            return R6.rump_anchor_once(A, sigma, 'RPS', tm), reasons
        except v.CertFail as e:
            reasons.append(e.reason)
        sigma /= 5.0
    raise v.CertFail('ANCHOR_FACTORIZATION_FAILED', ','.join(reasons))


def pcg_rtol(A, b, M, rtol, x0):
    n = len(b); it = [0]
    x, info = sla.cg(A, b, x0=(np.zeros(n) if x0 is None else x0), M=M, rtol=float(rtol), atol=0.0,
                     maxiter=min(10 * n, 8000), callback=lambda _: it.__setitem__(0, it[0] + 1))
    return np.asarray(x, np.float64), dict(iters=it[0], converged=bool(info == 0), scipy_info=int(info))


def solve_certify(A, b, M, lam, target, x0, holder, tm):
    """Shared residual strategy ('frozen-first cascade'): PCG with the FROZEN stopping rule (identical to the reference
    arms); certificate with the frozen bound; only if it misses the target, the rigorous EFT bound; if the target is
    still missed, continue PCG from the iterate with rtol / 10 (at most MAX_ROUNDS rounds). Both bounds are rigorous."""
    nb = max(float(np.linalg.norm(b)), 1e-300)
    rtol = float(min(1e-12, max(1e-16, 0.5 * float(target) * lam / nb)))
    x = x0; iters = 0; rounds = 0; conv = True; rec = None; eft_calls = 0
    while True:
        with tm('solve'):
            x, si = pcg_rtol(A, b, M, rtol, x)
        iters += si['iters']; conv = conv and si['converged']; rounds += 1
        with tm('residual'):
            rec = v.certificate(A, b, x, lam, True, target)
            rec['residual_kind'] = 'frozen'
            if rec['status'] == 'CERTIFIED' and (not rec['target_met'] or FORCE_EFT):
                if holder['layout'] is None or not holder['layout'].matches(A):   # built lazily, once per pattern
                    holder['layout'] = EFT.EftLayout(A)
                rec2 = EFT.certificate_eft(A, b, x, lam, target, holder['layout']); eft_calls += 1
                if rec2['status'] == 'CERTIFIED' and float.fromhex(rec2['eps']) < float.fromhex(rec['eps']):
                    rec = rec2
        if rec['status'] != 'CERTIFIED' or rec['target_met'] or rounds >= MAX_ROUNDS or rtol <= 1e-16:
            break
        rtol = max(1e-16, rtol / 10.0)
    return x, rec, dict(iters=iters, rounds=rounds, converged=conv, solver='pcg-RPS-cascade', final_rtol=rtol,
                        eft_calls=eft_calls)


def run_sequence_10a(family, method, output_dir, target=1e-8, repeat=0, run_order=0, code_hash='',
                     write_artifacts=True, family_tag=None):
    if method not in NEW:
        raise ValueError(method)
    cfg = NEW[method]
    for _s in list(R6.SESSIONS.values()):
        _s._reset()
    R6.SESSIONS.clear()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    art = out / 'artifacts'; art.mkdir(exist_ok=True)
    stem = f'{family.id}_{family.size_label}_{family.split}_{family.seed}_RPS_{method}_T1_lean_r{repeat}'
    dest = out / f'{stem}.jsonl'
    if dest.exists():
        dest.unlink()
    common = family.config['common']
    theta_min = float(common['theta_min']); itmax = float(common['it_max_factor'])
    st = dict(anc=None, op=None, c_s=None, z_s=None, it_ref=None, refresh_next=False, prev=None, evec=None,
              op_last=None, FA=None, link=None, sigma_carry=None, layout=None)
    rows = []
    for t in range(family.T + 1):
        A, b, meta = family.get(t)
        n = A.shape[0]; tm = R6.Timer(); begin = time.perf_counter()
        info = {'refresh': False, 'decision_reason': None, 'restorations': 0, 'estimates': 0}
        sinfo = {'iters': 0, 'converged': False}; x = np.zeros(n); lam = None; chain = []; new_anchor = None; fcount = 0

        def new_anchor_now(reason):
            nonlocal fcount
            with tm('estimate'):
                try:
                    lam_t, ev = _estimate(A, st['op_last'], st['evec'])
                except RuntimeError:            # previous preconditioner invalidated by a failed factorization
                    info['estimate_fallback'] = 'eigsh'          # (PARDISO session reset): start from scratch
                    st['op_last'] = None
                    lam_t, ev = _estimate(A, None, None)
            info['estimates'] += 1
            if not (np.isfinite(lam_t) and lam_t > 0):
                raise v.CertFail('LAMBDA_NONPOSITIVE')
            anc, retries = make_anchor_frac(A, lam_t, cfg['frac'], tm); fcount += 1 + len(retries)
            info['anchor_retries'] = info.get('anchor_retries', []) + retries
            st['evec'] = ev
            info['decision_reason'] = info['decision_reason'] or reason
            return anc

        try:
            if not v.arith_check():
                raise v.CertFail('FTZ_DETECTED')
            A = R6.ensure_canonical(A)
            anc = None
            if cfg['kind'] == 'per':
                if cfg['carry'] and st['sigma_carry'] is not None:
                    try:
                        anc = R6.rump_anchor_once(A, st['sigma_carry'], 'RPS', tm); fcount += 1
                        info['decision_reason'] = 'carried-shift'
                    except v.CertFail as e:
                        info['carry_failure'] = e.reason; fcount += 1
                        anc = new_anchor_now('carried-shift-failed')
                else:
                    anc = new_anchor_now('per-system')
            else:   # M2-A logic (frozen rules)
                reanchor = st['anc'] is None or st['refresh_next']
                if st['refresh_next']:
                    info['decision_reason'] = 'pcg-iter-trigger'
                if st['FA'] is None:
                    with tm('struct_setup'):
                        st['FA'] = FastAssembler(family.mesh, family.Kref, n)
                if not reanchor:
                    if family.mesh is None or meta['c'] is None:
                        raise v.CertFail('TRANSFER_DATA_INVALID', 'no element data')
                    with tm('binding'):
                        ok = st['FA'].matches(A, meta['c'])
                        info['binding'] = 'lean' if ok else 'lean-miss'
                        if not ok:
                            own = v.canonical(assemble_float(family.mesh, family.Z, meta['c'], family.Kref))
                            if v.matrix_sha256(own) != v.matrix_sha256(A):
                                raise v.CertFail('ASSEMBLY_MISMATCH')
                            info['binding'] = 'frozen'
                    with tm('transfer'):
                        zt = assembly_discrepancy_rowsum(family.mesh, family.Z, meta['c'])
                        lam, xi = m2a_transfer(family.mesh, family.Z, st['c_s'], meta['c'], st['anc']['lam_lo'], st['z_s'], zt)
                        info.update(theta=float(xi['theta']), wmax=float(xi['wmax']), zeta_t=float(zt), bound_type='m2a')
                        if xi['theta'] < theta_min:
                            reanchor = True; info['decision_reason'] = 'theta-below-threshold'
                        elif not lam > 0:
                            reanchor = True; info['decision_reason'] = 'transfer-nonpositive'
                if reanchor:
                    anc = new_anchor_now('initial-anchor' if st['anc'] is None else 'refresh')
            if anc is not None:
                op, kinfo = R6.precond_from_anchor(anc, tm); info.update(kinfo)
                st.update(anc=anc, op=op, it_ref=None, refresh_next=False); st['op_last'] = op
                if cfg['kind'] == 'per' and cfg['carry']:
                    st['sigma_carry'] = anc['sigma']
                if cfg['kind'] == 'm2a':
                    with tm('binding'):
                        if not st['FA'].matches(A, meta['c']):
                            own_s = v.canonical(assemble_float(family.mesh, family.Z, meta['c'], family.Kref))
                            if v.matrix_sha256(own_s) != v.matrix_sha256(A):
                                raise v.CertFail('ASSEMBLY_MISMATCH', 'anchor matrix is not the element model')
                    with tm('transfer'):
                        st['z_s'] = assembly_discrepancy_rowsum(family.mesh, family.Z, meta['c'])
                    st['c_s'] = meta['c']
                lam = anc['lam_lo']; info['refresh'] = True; info['bound_type'] = 'anchor'; new_anchor = anc
            if not (lam is not None and np.isfinite(lam) and lam > 0):
                raise v.CertFail('LAMBDA_NONPOSITIVE')
            x, rec, sinfo = solve_certify(A, b, st['op'], lam, target, st['prev'], st, tm)
            if rec['status'] == 'CERTIFIED' and not rec['target_met']:
                # target restoration (symmetric rule): re-anchor once at this step and solve again
                info['restorations'] += 1
                anc = new_anchor_now('target-restoration')
                op, kinfo = R6.precond_from_anchor(anc, tm); info.update(kinfo)
                st.update(anc=anc, op=op, it_ref=None, refresh_next=False); st['op_last'] = op
                if cfg['kind'] == 'per' and cfg['carry']:
                    st['sigma_carry'] = anc['sigma']
                if cfg['kind'] == 'm2a':
                    with tm('transfer'):
                        st['z_s'] = assembly_discrepancy_rowsum(family.mesh, family.Z, meta['c'])
                    st['c_s'] = meta['c']
                lam = anc['lam_lo']; info['refresh'] = True; info['bound_type'] = 'anchor'; new_anchor = anc
                x, rec, s2 = solve_certify(A, b, st['op'], lam, target, x, st, tm)
                sinfo = dict(iters=sinfo['iters'] + s2['iters'], rounds=sinfo['rounds'] + s2['rounds'],
                             converged=sinfo['converged'] and s2['converged'], solver=s2['solver'], final_rtol=s2['final_rtol'],
                             eft_calls=sinfo['eft_calls'] + s2['eft_calls'])
            if cfg['kind'] == 'm2a':
                if st['it_ref'] is None:
                    st['it_ref'] = max(1, sinfo['iters'])
                elif sinfo['iters'] > itmax * st['it_ref']:
                    st['refresh_next'] = True
            if rec['status'] != 'CERTIFIED':
                raise v.CertFail(rec['reason'] or 'CERT_FAILED')
            st['prev'] = x.copy()
        except (v.CertFail, ValueError, RuntimeError, OverflowError, ZeroDivisionError, MemoryError) as e:
            reason = e.reason if isinstance(e, v.CertFail) else type(e).__name__
            rec = {'schema': 'cars-cert/0.4-eft', 'status': 'UNVERIFIED', 'reason': reason, 'target_met': None,
                   'target': float(target).hex(), 'matrix_sha256': v.matrix_sha256(A), 'rhs_sha256': v.vec_sha256(b),
                   'x_sha256': v.vec_sha256(x)}
            info['failure_detail'] = str(e)[:300]; st['prev'] = None
            if cfg['kind'] == 'per' and cfg.get('carry'):
                st['sigma_carry'] = None
        tm.t['other'] += max(0.0, time.perf_counter() - begin - sum(tm.t.values()))
        with tm('io'):
            sol_sha = None; xfile = None
            if write_artifacts:
                xfile = art / f'{stem}_x_t{t}.npy'; np.save(xfile, x, allow_pickle=False); sol_sha = file_sha256(xfile)
            if new_anchor is not None:
                st['link'] = {'kind': 'rump', 'anchor_t': t, 'anchor_matrix_sha256': v.matrix_sha256(A), 'engine': 'RPS',
                              'sigma': float(new_anchor['sigma']).hex(), 'c': float(new_anchor['c']).hex(),
                              's_hi': float(new_anchor['s_hi']).hex(),
                              **{k: x_ for k, x_ in new_anchor['extra'].items() if x_ is not None}}
        if rec['status'] == 'CERTIFIED':
            base = dict(st['link']); typ = info.get('bound_type')
            if typ == 'anchor':
                base['type'] = 'rump-isspd'; chain = [base]
            elif typ == 'm2a':
                link = dict(base); link['type'] = 'm2a'; link['anchor_type'] = 'rump-isspd'
                link.update(theta=float(info['theta']).hex(), zeta_s=float(st['z_s']).hex(), zeta_t=float(info['zeta_t']).hex(),
                            coeff_s_sha256=v.vec_sha256(st['c_s']), coeff_t_sha256=v.vec_sha256(meta['c']),
                            element_data_sha256=family.factor_hash, split_theta0=False)
                chain = [link]
        meas = sum(tm.t[k] for k in R6.COMPONENTS if k != 'io')
        cf = meas - tm.t['anchor_V'] - (tm.t['estimate'] if cfg['kind'] == 'per' else 0.0)
        rec.update(family=family.id, method=method, engine='RPS', seed=family.seed, split=family.split,
                   size=family.size_label, t=t, n=n, nnz=meta['nnz'], label=family.label,
                   info_model=('gray-box' if cfg['kind'] == 'm2a' else 'black-box'), track='T1', binding='lean',
                   repeat=repeat, run_order=run_order, code_hash=code_hash, lambda_chain=chain, solver=sinfo,
                   details=info, solution_file=(str(xfile.relative_to(out)) if xfile else None), solution_sha256=sol_sha,
                   coeff_sha256=meta.get('coeff_sha256'), timing_s={**tm.t, 'measured': meas, 'counterfactual': cf},
                   factor_count=fcount, phase='PHASE10A-DEV', sigma_frac=cfg['frac'])
        with dest.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + '\n')
        rows.append(rec)
    return dest, rows
