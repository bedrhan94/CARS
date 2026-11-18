"""Phase 3B replay verifier (new file; pilot verify_m2.py preserved unchanged).

Fixes relative to the pilot verifier (PROTOCOL_AMENDMENTS.md, VF-1 .. VF-5):
 VF-1  an 'anchor-inertia' link is accepted only if the anchor matrix IS the certified matrix
       (anchor_matrix_sha256 == record matrix_sha256 and anchor_t == t). The pilot verifier accepted a link to
       an older anchor, i.e. lambda_lo of A_s claimed for A_t (demonstrated in audit_m2.py, part E).
 VF-2  safe anchor cache: key = (factor file SHA-256, anchor matrix SHA-256, SHA-256 of the M0 source,
       arith_check result). The cached value is only the re-validated (nu, rho, delta, sigma) of THAT factor
       for THAT matrix; transfers and residuals are always recomputed. File hashes are cached per
       (path, size, mtime_ns) and recomputed if any of them changes.
 VF-3  three-way outcome per record: VERIFIED / REJECTED / UNAVAILABLE (missing artifact), never conflated.
 VF-4  M2-A links: the regenerated matrix must equal the own assembly of (Z, K_ref, c_t) (via the family
       definition and the matrix hash); split links (M2-A') recompute theta0 = second-smallest ratio.
 VF-5  C-cw weights are replayed from the stored weight file.
The verifier reuses the M0 arithmetic primitives: it is an independent INVOCATION, not an independent
arithmetic implementation (exact rational checks are added for n <= exact_n).
"""
from __future__ import annotations
import hashlib, json, os, sys
from fractions import Fraction
from pathlib import Path
import numpy as np
from cars import _reference_m0 as v
from cars.verify import recheck_anchor, exact_inertia, exact_residual2
from cars_m2.families import REF, Family3D
if str(REF) not in sys.path:
    sys.path.insert(0, str(REF))
sys.modules['cars_ref'] = v
from m2_ref import assembly_discrepancy_rowsum, m2a_transfer  # noqa: E402

_M0_SHA = hashlib.sha256(Path(v.__file__).read_bytes()).hexdigest()
_FILE_SHA = {}
_ANCHOR = {}
STATS = {'anchor_rechecks': 0, 'anchor_cache_hits': 0}


class Unavailable(Exception):
    pass


def fsha(path):
    p = Path(path)
    if not p.is_file():
        raise Unavailable(str(p))
    st = p.stat(); key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _FILE_SHA:
        _FILE_SHA[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    return _FILE_SHA[key]


def _inside(base, rel):
    p = (Path(base) / rel)
    if not p.resolve().is_relative_to(Path(base).resolve()):
        raise ValueError('PATH_ESCAPE')
    return p


def _anchor(link, family, out):
    A0, _, _ = family.get(int(link['anchor_t']))
    if v.matrix_sha256(A0) != link['anchor_matrix_sha256']:
        raise ValueError('ANCHOR_MATRIX_SHA')
    f = _inside(out, link['factor_file'])
    sha = fsha(f)
    if sha != link['factor_sha256']:
        raise ValueError('FACTOR_SHA')
    key = (sha, link['anchor_matrix_sha256'], _M0_SHA, bool(v.arith_check()))
    if key in _ANCHOR:
        STATS['anchor_cache_hits'] += 1
        return _ANCHOR[key], A0
    STATS['anchor_rechecks'] += 1
    rec = recheck_anchor(A0, f)
    if rec['nu'] != 0 or not rec['rho'] > 0:
        raise ValueError('ANCHOR_NOT_SPD')
    _ANCHOR[key] = rec
    return rec, A0


def verify_record(record, family: Family3D, base_dir, exact_n=0):
    """Returns (outcome, errors) with outcome in {'VERIFIED', 'REJECTED', 'UNAVAILABLE'}."""
    err = []; out = Path(base_dir)
    try:
        t = int(record['t'])
        A, b, meta = family.get(t)
        if record['matrix_sha256'] != v.matrix_sha256(A): err.append('MATRIX_SHA')
        if record['rhs_sha256'] != v.vec_sha256(b): err.append('RHS_SHA')
        fp = _inside(out, record['solution_file'])
        if fsha(fp) != record['solution_sha256']: err.append('SOLUTION_FILE_SHA')
        x = np.load(fp, allow_pickle=False)
        if v.vec_sha256(x) != record['x_sha256']: err.append('SOLUTION_SHA')
        if record['status'] == 'CERTIFIED':
            chain = record.get('lambda_chain', [])
            if len(chain) != 1: raise ValueError('CHAIN_COUNT')
            link = chain[0]; typ = link.get('type')
            if typ == 'gershgorin':
                wf = _inside(out, link['w_file'])
                if fsha(wf) != link['w_file_sha256']: err.append('W_FILE_SHA')
                w = np.load(wf, allow_pickle=False)
                if v.vec_sha256(w) != link['w_sha256']: err.append('W_SHA')
                lam = v.gershgorin_lower(A, w)
            else:
                anc, A0 = _anchor(link, family, out)
                if int(link['anchor_t']) > t: err.append('ANCHOR_FROM_FUTURE')
                if typ == 'anchor-inertia':
                    if link['anchor_matrix_sha256'] != record['matrix_sha256'] or int(link['anchor_t']) != t:
                        err.append('ANCHOR_INERTIA_NOT_FOR_THIS_MATRIX')
                    lam = anc['rho']
                elif typ == 'm2a':
                    cs = family.coeff(int(link['anchor_t'])); ct = family.coeff(t)
                    if link['coeff_s_sha256'] != v.vec_sha256(cs): err.append('COEFF_S_SHA')
                    if link['coeff_t_sha256'] != v.vec_sha256(ct): err.append('COEFF_T_SHA')
                    if link['element_data_sha256'] != family.factor_hash: err.append('ELEMENT_DATA_SHA')
                    zs = assembly_discrepancy_rowsum(family.mesh, family.Z, cs)
                    zt = assembly_discrepancy_rowsum(family.mesh, family.Z, ct)
                    th0 = None
                    if link.get('split_theta0'):
                        r_ = np.sort(ct / cs); th0 = float(r_[1]) if len(r_) > 1 else float(r_[0])
                    lam, xi = m2a_transfer(family.mesh, family.Z, cs, ct, anc['rho'], zs, zt, theta0=th0)
                    if float.fromhex(link['theta']) != xi['theta']: err.append('THETA')
                    if float.fromhex(link['zeta_s']) != zs: err.append('ZETA_S')
                    if float.fromhex(link['zeta_t']) != zt: err.append('ZETA_T')
                elif typ == 'weyl':
                    lam = v.weyl_transfer_lower(A0, A, anc['rho'])
                else:
                    raise ValueError('UNKNOWN_CHAIN_' + str(typ))
            lo = float.fromhex(record['lambda_lo'])
            if not (np.isfinite(lo) and lo > 0 and lo <= lam): err.append('UNSUPPORTED_LAMBDA_LO')
            rr = v.residual_bound(A, b, x); rhi = float.fromhex(record['R_hi'])
            if rr > rhi: err.append('RESIDUAL_RECOMPUTE')
            eps = float.fromhex(record['eps']); tar = float.fromhex(record['target'])
            if v.up(rhi / lo) != eps: err.append('EPS')
            if record['target_met'] != bool(eps <= tar): err.append('TARGET_MET')
            if exact_n and A.shape[0] <= exact_n:
                if Fraction(float(rhi)) ** 2 < exact_residual2(A, b, x): err.append('EXACT_RESIDUAL')
                inertia = exact_inertia(A.toarray(), lo)
                if inertia is None: err.append('EXACT_INERTIA_UNDETERMINED')
                elif inertia[0] != 0: err.append('EXACT_LAMBDA')
        else:
            if record.get('target_met') is not None: err.append('NONCERT_TARGET')
            if record.get('eps') is not None: err.append('NONCERT_EPS')
            if not record.get('reason'): err.append('NONCERT_REASON')
    except Unavailable as e:
        return 'UNAVAILABLE', ['MISSING_ARTIFACT:' + str(e)[-120:]]
    except Exception as e:
        err.append('VERIFY_EXCEPTION:' + repr(e)[:200])
    return ('VERIFIED' if not err else 'REJECTED'), err


def verify_jsonl(path, exact_n=0):
    path = Path(path)
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not recs:
        return {}
    r0 = recs[0]
    fam = Family3D(r0['family'], r0['size'], r0['seed'], r0['split'])
    res = {'VERIFIED': 0, 'REJECTED': 0, 'UNAVAILABLE': 0, 'certified': 0, 'errors': []}
    for r in recs:
        o, e = verify_record(r, fam, path.parent, exact_n=exact_n)
        res[o] += 1; res['certified'] += r['status'] == 'CERTIFIED'
        if o != 'VERIFIED':
            res['errors'].append((r['t'], o, e[:3]))
    return res
