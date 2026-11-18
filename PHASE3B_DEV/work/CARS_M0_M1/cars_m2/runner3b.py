"""Phase 3B runner (new file). The pilot's cars_m2/runner.py is preserved unchanged for provenance.

Software fixes relative to the pilot runner (PROTOCOL_AMENDMENTS.md, SW-1 .. SW-8):
 SW-1  artifact file I/O (factor .npz, solution .npy, hashing) is timed in component 'io' and EXCLUDED from
       both totals (research/17 section 5). The pilot counted it inside 'setup'.
 SW-2  shared shift estimator (research/17 section 4.2): eigsh(SA) only when no factor exists yet; otherwise
       5 steps of preconditioned inverse iteration with the most recent factor from the previous eigenvector.
 SW-3  M2-A gray-box binding: the solved matrix must be bit-identical to the own assembly of (Z, K_ref, c_t),
       else ASSEMBLY_MISMATCH (fail closed). Its cost is charged to M2-A ('transfer').
 SW-4  kernels K1 / K3 (cars_m2.kernels); preconditioner set-up time in 'precond_setup' (counted).
 SW-5  track T2: one shared, method-independent preconditioner sequence (factor of A_{5j}, j = 0..4);
       its cost is reported separately and excluded from method totals; the PCG-iteration refresh trigger is
       disabled under T2 (it measures the shared preconditioner, not the method).
 SW-6  C-cw stores its weight vector (the pilot's C-cw records could not be replayed).
 SW-7  methods M2-A' (split theta0 = second-smallest ratio) and C-fix-5 / C-weyl under the same harness.
 SW-8  per-record provenance: kernel, track, repeat index, run order, code hash.
Totals: measured = all components except 'io'; counterfactual (stringent, hypothetical) = measured - anchor_V
        - (estimate for C-per-*). anchor_V includes L extraction and structural checks (baseline-favouring).
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from cars import _reference_m0 as v
from cars.verify import file_sha256
from cars.runner import _positive_weight
from cars_m2.anchor import anchor_inertia_lu
from cars_m2.families import Family3D, REF
from cars_m2 import kernels
if str(REF) not in sys.path:
    sys.path.insert(0, str(REF))
sys.modules['cars_ref'] = v
from m2_ref import assembly_discrepancy_rowsum, m2a_transfer  # noqa: E402
from fem3d_ref import assemble_float  # noqa: E402

METHODS = ('C-per-1F', 'C-per-2F', 'M2-A', "M2-A'", 'C-fix-5', 'C-weyl', 'C-cw')
COMPONENTS = ('estimate', 'anchor_F', 'anchor_V', 'precond_setup', 'transfer', 'solve', 'residual', 'other', 'io')


class Timer:
    def __init__(self):
        self.t = {k: 0.0 for k in COMPONENTS}

    def __call__(self, key):
        tm = self

        class _C:
            def __enter__(s):
                s.s = time.perf_counter(); return s

            def __exit__(s, *a):
                tm.t[key] += time.perf_counter() - s.s
        return _C()


def _estimate(A, op=None, x0=None, steps=5):
    """Heuristic only (never a bound). Returns (lambda_tilde, unit vector)."""
    n = A.shape[0]
    if op is None:
        lam = float(sla.eigsh(A, k=1, which='SA', v0=np.linspace(1.0, 1.1, n), tol=1e-6, maxiter=10000,
                              return_eigenvectors=False)[0])
        x = None
        return lam, x
    x = np.ones(n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    x /= np.linalg.norm(x)
    for _ in range(steps):
        x = op.matvec(x); x /= np.linalg.norm(x)
    return float(x @ (A @ x)), x


def _anchor(A, lam_tilde, tm):
    """M0 F6 anchor with the M0 retry rule (sigma /= 5, <= 3 retries). All attempts are charged."""
    sigma = 0.9 * lam_tilde; reasons = []
    for _ in range(4):
        try:
            anc = anchor_inertia_lu(A, sigma, 0)
            tm.t['anchor_F'] += anc['instrumentation']['F_s']; tm.t['anchor_V'] += anc['instrumentation']['V_s']
            if anc['rho'] > 0 and anc['nu'] == 0:
                return anc
            reasons.append('RHO_NONPOSITIVE')
        except v.CertFail as e:
            reasons.append(e.reason)
        sigma /= 5.0
    raise v.CertFail('ANCHOR_FACTORIZATION_FAILED', ','.join(reasons))


def _pcg(A, b, M, lam, target, x0):
    n = len(b)
    tau = max(0.5 * float(target) * lam, 0.0)
    rtol = float(min(1e-12, max(1e-16, tau / max(np.linalg.norm(b), 1e-300))))
    it = [0]
    x, info = sla.cg(A, b, x0=(np.zeros(n) if x0 is None else x0), M=M, rtol=rtol, atol=0.0,
                     maxiter=min(10 * n, 8000), callback=lambda _: it.__setitem__(0, it[0] + 1))
    x = np.asarray(x, np.float64)
    return x, dict(iters=it[0], converged=bool(info == 0), scipy_info=int(info))


class SharedPreconditioner:
    """Track T2: method-independent preconditioner sequence; built once per (family, size, seed)."""

    def __init__(self, family, kernel):
        self.ops = {}; self.cost = 0.0; self.kernel = kernel
        x = None; op = None
        for j in range(0, family.T + 1, 5):
            A, _, _ = family.get(j)
            t0 = time.perf_counter()
            lam, x = _estimate(A, op, x)
            tm = Timer(); anc = _anchor(A, lam, tm)
            op, _ = kernels.make(kernel, anc['factor'])
            self.cost += time.perf_counter() - t0
            self.ops[j] = op

    def get(self, t):
        return self.ops[max(j for j in self.ops if j <= t)]


def run_sequence_3b(family: Family3D, method: str, output_dir, kernel='K3', track='T1', target=1e-8,
                    shared=None, repeat=0, run_order=0, code_hash='', write_artifacts=True):
    if method not in METHODS:
        raise ValueError(method)
    if track == 'T2' and shared is None:
        raise ValueError('T2 needs a SharedPreconditioner')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); art = out / 'artifacts'; art.mkdir(exist_ok=True)
    stem = f'{family.id}_{family.size_label}_{family.split}_{family.seed}_{method}_{kernel}_{track}_r{repeat}'.replace("'", 'p')
    dest = out / f'{stem}.jsonl'
    if dest.exists():
        dest.unlink()
    common = family.config['common']
    theta_min = float(common['theta_min']); itmax = float(common['it_max_factor'])
    st = dict(anc=None, op=None, A_s=None, c_s=None, z_s=None, anchor_t=None, link=None, it_ref=None,
              refresh_next=False, prev=None, evec=None, op_last=None)
    rows = []
    for t in range(family.T + 1):
        A, b, meta = family.get(t)                      # problem generation: untimed for every method
        n = A.shape[0]; tm = Timer(); begin = time.perf_counter()
        info = {'refresh': False, 'decision_reason': None}; sinfo = {'iters': 0, 'converged': False}
        x = np.zeros(n); lam = None; chain = []; w_saved = None; new_anchor_obj = None
        try:
            if not v.arith_check():
                raise v.CertFail('FTZ_DETECTED')
            reanchor = False
            if method in ('C-per-1F', 'C-per-2F'):
                reanchor = True; info['decision_reason'] = 'per-system'
            elif method == 'C-cw':
                with tm('transfer'):
                    lam, w, worigin = _positive_weight(A, st['prev'])
                if not lam > 0:
                    raise v.CertFail('LAMBDA_NONPOSITIVE')
                w_saved = np.asarray(w, dtype=np.float64); info['bound_type'] = 'gershgorin'; info['w_origin'] = worigin
            else:
                reanchor = st['anc'] is None or (method == 'C-fix-5' and t % 5 == 0) or st['refresh_next']
                if st['refresh_next']:
                    info['decision_reason'] = 'pcg-iter-trigger'
                if not reanchor:
                    with tm('transfer'):
                        if method in ('M2-A', "M2-A'"):
                            if family.mesh is None or meta['c'] is None:
                                raise v.CertFail('TRANSFER_DATA_INVALID', 'no element data')
                            own = v.canonical(assemble_float(family.mesh, family.Z, meta['c'], family.Kref))
                            if v.matrix_sha256(own) != v.matrix_sha256(A):
                                raise v.CertFail('ASSEMBLY_MISMATCH')
                            zt = assembly_discrepancy_rowsum(family.mesh, family.Z, meta['c'])
                            th0 = None
                            if method == "M2-A'":
                                r_ = np.sort(meta['c'] / st['c_s']); th0 = float(r_[1]) if len(r_) > 1 else float(r_[0])
                            lam, xi = m2a_transfer(family.mesh, family.Z, st['c_s'], meta['c'], st['anc']['rho'],
                                                   st['z_s'], zt, theta0=th0)
                            info.update(theta=float(xi['theta']), wmax=float(xi['wmax'])); info['bound_type'] = 'm2a'
                            info['zeta_t'] = zt
                            if xi['theta'] < theta_min:
                                reanchor = True; info['decision_reason'] = 'theta-below-threshold'
                            elif not lam > 0:
                                reanchor = True; info['decision_reason'] = 'transfer-nonpositive'
                        else:   # C-fix-5, C-weyl
                            lam = v.weyl_transfer_lower(st['A_s'], A, st['anc']['rho']); info['bound_type'] = 'weyl'
                            if not lam > 0:
                                reanchor = True; info['decision_reason'] = 'weyl-invalid'
                if reanchor and not info['decision_reason']:
                    info['decision_reason'] = 'initial-anchor' if st['anc'] is None else 'fixed-schedule'
            if reanchor:
                with tm('estimate'):
                    lam_t, ev = _estimate(A, st['op_last'], st['evec'])
                if not (np.isfinite(lam_t) and lam_t > 0):
                    raise v.CertFail('LAMBDA_NONPOSITIVE')
                anc = _anchor(A, lam_t, tm)
                with tm('precond_setup'):
                    if method == 'C-per-2F' or track == 'T2':
                        op, kinfo = None, {'kernel': 'none' if method == 'C-per-2F' else f'{kernel}-shared'}
                    else:
                        op, kinfo = kernels.make(kernel, anc['factor'])
                info.update(kinfo)
                st.update(anc=anc, op=op, A_s=A, c_s=meta['c'], anchor_t=t, it_ref=None, refresh_next=False, evec=ev)
                if op is not None:
                    st['op_last'] = op
                if method in ('M2-A', "M2-A'"):
                    with tm('transfer'):
                        if family.mesh is None or meta['c'] is None:
                            raise v.CertFail('TRANSFER_DATA_INVALID', 'no element data')
                        own_s = v.canonical(assemble_float(family.mesh, family.Z, meta['c'], family.Kref))
                        if v.matrix_sha256(own_s) != v.matrix_sha256(A):
                            raise v.CertFail('ASSEMBLY_MISMATCH', 'anchor matrix is not the element model')
                        st['z_s'] = assembly_discrepancy_rowsum(family.mesh, family.Z, meta['c'])
                lam = anc['rho']; info['refresh'] = True; info['bound_type'] = 'anchor-inertia'
                new_anchor_obj = anc
            if not (lam is not None and np.isfinite(lam) and lam > 0):
                raise v.CertFail('LAMBDA_NONPOSITIVE')
            # ---- solve
            with tm('solve'):
                if method == 'C-per-2F':
                    lu2 = sla.splu(A.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                                   options={'SymmetricMode': True})
                    x = np.asarray(lu2.solve(b), np.float64); sinfo = dict(iters=0, converged=True, solver='direct')
                    # estimator operator for the next step (inverse iteration with the previous unshifted factor)
                    st['op_last'] = sla.LinearOperator((n, n), matvec=lu2.solve, dtype=np.float64)
                else:
                    if method == 'C-cw':
                        dg = A.diagonal(); M = sla.LinearOperator((n, n), matvec=lambda r: r / dg, dtype=np.float64)
                        sname = 'pcg-Jacobi'
                    elif track == 'T2':
                        M = shared.get(t); sname = f'pcg-{kernel}-shared'
                        st['op_last'] = M
                    else:
                        M = st['op']; sname = f'pcg-{kernel}'
                    x, sinfo = _pcg(A, b, M, lam, target, st['prev']); sinfo['solver'] = sname
            if method in ('M2-A', "M2-A'") and track == 'T1':
                if st['it_ref'] is None:
                    st['it_ref'] = max(1, sinfo['iters'])
                elif sinfo['iters'] > itmax * st['it_ref']:
                    st['refresh_next'] = True
            with tm('residual'):
                rec = v.certificate(A, b, x, lam, True, target)
            if rec['status'] != 'CERTIFIED':
                raise v.CertFail(rec['reason'] or 'CERT_FAILED')
            st['prev'] = x.copy()
        except (v.CertFail, ValueError, RuntimeError, OverflowError, ZeroDivisionError, MemoryError) as e:
            reason = e.reason if isinstance(e, v.CertFail) else type(e).__name__
            rec = {'schema': 'cars-cert/0.3', 'status': 'UNVERIFIED', 'reason': reason, 'target_met': None,
                   'target': float(target).hex(), 'matrix_sha256': v.matrix_sha256(A), 'rhs_sha256': v.vec_sha256(b),
                   'x_sha256': v.vec_sha256(x)}
            info['failure_detail'] = str(e)[:300]; st['prev'] = None
        tm.t['other'] += max(0.0, time.perf_counter() - begin - sum(tm.t.values()))
        # ---- artifacts (timed as io; excluded from totals)
        with tm('io'):
            xfile = art / f'{stem}_x_t{t}.npy'; np.save(xfile, x, allow_pickle=False)
            sol_sha = file_sha256(xfile)
            if new_anchor_obj is not None and write_artifacts:
                fp = art / f'{stem}_anchor_t{t}.npz'
                Lc = v.canonical(new_anchor_obj['factor']['L'])
                np.savez(fp, data=Lc.data, indices=Lc.indices, indptr=Lc.indptr, shape=np.array(Lc.shape),
                         d=new_anchor_obj['factor']['d'], perm=new_anchor_obj['factor']['perm'],
                         sigma=np.float64(new_anchor_obj['sigma']), nu=np.int64(new_anchor_obj['nu']))
                st['link'] = {'anchor_t': t, 'factor_file': str(fp.relative_to(out)), 'factor_sha256': file_sha256(fp),
                              'anchor_matrix_sha256': v.matrix_sha256(A)}
            if w_saved is not None:
                wf = art / f'{stem}_w_t{t}.npy'; np.save(wf, w_saved, allow_pickle=False)
                wlink = {'type': 'gershgorin', 'w_sha256': v.vec_sha256(w_saved), 'w_file': str(wf.relative_to(out)),
                         'w_file_sha256': file_sha256(wf)}
        if rec['status'] == 'CERTIFIED':
            if method == 'C-cw':
                chain = [wlink]
            else:
                link = dict(st['link']); typ = info.get('bound_type')
                link['type'] = 'anchor-inertia' if typ == 'anchor-inertia' else typ
                if typ in ('m2a',):
                    link.update(theta=float(info['theta']).hex(), zeta_s=float(st['z_s']).hex(),
                                zeta_t=float(info['zeta_t']).hex(), coeff_s_sha256=v.vec_sha256(st['c_s']),
                                coeff_t_sha256=v.vec_sha256(meta['c']), element_data_sha256=family.factor_hash,
                                split_theta0=(method == "M2-A'"))
                chain = [link]
        meas = sum(tm.t[k] for k in COMPONENTS if k != 'io')
        cf = meas - tm.t['anchor_V'] - (tm.t['estimate'] if method in ('C-per-1F', 'C-per-2F') else 0.0)
        rec.update(family=family.id, method=method, seed=family.seed, split=family.split, size=family.size_label,
                   t=t, n=n, nnz=meta['nnz'], label=family.label,
                   info_model=('gray-box' if method in ('M2-A', "M2-A'") else 'black-box'),
                   kernel=kernel, track=track, repeat=repeat, run_order=run_order, code_hash=code_hash,
                   lambda_chain=chain, solver=sinfo, details=info, solution_file=str(xfile.relative_to(out)),
                   solution_sha256=sol_sha, coeff_sha256=meta.get('coeff_sha256'),
                   timing_s={**tm.t, 'measured': meas, 'counterfactual': cf},
                   factor_count=(int(new_anchor_obj is not None) + int(method == 'C-per-2F')),
                   factor_nnz_L=(int(new_anchor_obj['factor']['L'].nnz) if new_anchor_obj is not None else 0))
        with dest.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + '\n')
        rows.append(rec)
    return dest, rows
