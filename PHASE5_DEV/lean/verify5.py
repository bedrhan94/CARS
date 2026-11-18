"""Phase 5 record verifier (new).
 * FRZ and M0L records (anchor kind 'm0'): delegated UNCHANGED to the frozen Phase 3B verifier
   (cars_m2.verify_m2_3b.verify_record: VF-1..VF-5, M0 re-validation of the stored factor).
 * Rump records ('rump-isspd' anchors, and 'm2a' / 'weyl' links on such anchors):
   R-1 identity: matrix / rhs / solution hashes as in the frozen verifier;
   R-2 VF-1 analogue: a 'rump-isspd' link certifies only its own matrix (anchor_t == t, same matrix SHA-256);
   R-3 the a-priori constant c and s_hi = up(sigma + c) are recomputed bit-exactly from the regenerated anchor matrix;
   R-4 CROSS-ENGINE RE-PROOF: A~ is rebuilt and its floating-point Cholesky is re-run with the OTHER engine
       (RPA -> CHOLMOD, RCH -> PARDISO); by Rump (2006) Cor. 2.4 completion with either implementation proves
       lambda_min(A_s) > sigma independently of the run that produced the record;
   R-5 m2a / weyl transfers recomputed from the frozen formulas with lambda_s^lo = sigma_s;
   R-6 residual bound, eps = up(R_hi/lambda_lo) and target flag as in the frozen verifier;
   R-7 exact rational inertia of A - lambda_lo I and exact residual for n <= exact_n.
Outcomes: VERIFIED / REJECTED / UNAVAILABLE."""
from __future__ import annotations
import sys
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner5 as R                                                   # noqa: E402
from cars import _reference_m0 as v                                    # noqa: E402
from cars.verify import exact_inertia, exact_residual2                 # noqa: E402
from cars_m2 import verify_m2_3b as F3B                                # noqa: E402  frozen verifier
from m2_ref import assembly_discrepancy_rowsum, m2a_transfer           # noqa: E402

CROSS = {'RPA': 'RCH', 'RCH': 'RPA', 'RPS': 'RCH'}
_RUMP_CACHE = {}
STATS = {'rump_reproofs': 0, 'rump_cache_hits': 0}


def reprove_rump(link, family, engine_override=None):
    """Returns (lambda_s_lo, A_s) after an independent re-proof; raises ValueError on any failure."""
    A0, _, _ = family.get(int(link['anchor_t']))
    A0 = R.ensure_canonical(A0)
    if v.matrix_sha256(A0) != link['anchor_matrix_sha256']:
        raise ValueError('ANCHOR_MATRIX_SHA')
    eng = engine_override or CROSS[link['engine']]
    key = (link['anchor_matrix_sha256'], link['sigma'], link['c'], link['s_hi'], eng)
    if key in _RUMP_CACHE:
        STATS['rump_cache_hits'] += 1
        return _RUMP_CACHE[key], A0
    if not R.is_symmetric(A0):
        raise ValueError('NOT_SYMMETRIC')
    sigma = float.fromhex(link['sigma'])
    c = R.rump_constant(A0)
    if float(c).hex() != link['c']:
        raise ValueError('RUMP_C')
    s_hi = float(v.up(sigma + c))
    if float(s_hi).hex() != link['s_hi']:
        raise ValueError('RUMP_S_HI')
    pos = R.diag_positions(A0)
    At = A0.copy(); At.data[pos] = v.dn(At.data[pos] - s_hi)
    try:
        R.ENGINE_FACTOR[eng](At)
    except v.CertFail as e:
        raise ValueError('RUMP_REPROOF_FAILED_' + e.reason)
    STATS['rump_reproofs'] += 1
    if not (sigma > 0 and np.isfinite(sigma)):
        raise ValueError('RUMP_SIGMA')
    _RUMP_CACHE[key] = sigma
    return sigma, A0


def verify_record5(record, family, base_dir, exact_n=0, engine_override=None):
    chain = record.get('lambda_chain') or []
    if record.get('engine') in (None, 'M0L') or record['method'] in ('C-per-1F', 'C-per-2F', 'M2-A'):
        if chain and chain[0].get('kind', 'm0') == 'm0':
            return F3B.verify_record(record, family, base_dir, exact_n=exact_n)
        if not chain:
            return F3B.verify_record(record, family, base_dir, exact_n=exact_n)
    err = []; out = Path(base_dir)
    try:
        t = int(record['t'])
        A, b, meta = family.get(t)
        if record['matrix_sha256'] != v.matrix_sha256(A): err.append('MATRIX_SHA')
        if record['rhs_sha256'] != v.vec_sha256(b): err.append('RHS_SHA')
        if record.get('solution_file') is None:
            raise F3B.Unavailable('solution file not written')
        fp = F3B._inside(out, record['solution_file'])
        if F3B.fsha(fp) != record['solution_sha256']: err.append('SOLUTION_FILE_SHA')
        x = np.load(fp, allow_pickle=False)
        if v.vec_sha256(x) != record['x_sha256']: err.append('SOLUTION_SHA')
        if record['status'] == 'CERTIFIED':
            if len(chain) != 1: raise ValueError('CHAIN_COUNT')
            link = chain[0]; typ = link.get('type')
            if link.get('kind') != 'rump' and typ != 'rump-isspd' and link.get('anchor_type') != 'rump-isspd':
                raise ValueError('UNKNOWN_ANCHOR_KIND')
            lam_s, A0 = reprove_rump(link, family, engine_override)
            if int(link['anchor_t']) > t: err.append('ANCHOR_FROM_FUTURE')
            if typ == 'rump-isspd':
                if link['anchor_matrix_sha256'] != record['matrix_sha256'] or int(link['anchor_t']) != t:
                    err.append('ANCHOR_NOT_FOR_THIS_MATRIX')
                lam = lam_s
            elif typ == 'm2a':
                cs = family.coeff(int(link['anchor_t'])); ct = family.coeff(t)
                if link['coeff_s_sha256'] != v.vec_sha256(cs): err.append('COEFF_S_SHA')
                if link['coeff_t_sha256'] != v.vec_sha256(ct): err.append('COEFF_T_SHA')
                if link['element_data_sha256'] != family.factor_hash: err.append('ELEMENT_DATA_SHA')
                # gray-box premise: regenerated A_t and A_s are the own assembly of their coefficients
                if v.matrix_sha256(R.assemble_float(family.mesh, family.Z, ct, family.Kref)) != v.matrix_sha256(A):
                    err.append('ASSEMBLY_T')
                zs = assembly_discrepancy_rowsum(family.mesh, family.Z, cs)
                zt = assembly_discrepancy_rowsum(family.mesh, family.Z, ct)
                lam, xi = m2a_transfer(family.mesh, family.Z, cs, ct, lam_s, zs, zt)
                if float.fromhex(link['theta']) != xi['theta']: err.append('THETA')
                if float.fromhex(link['zeta_s']) != zs: err.append('ZETA_S')
                if float.fromhex(link['zeta_t']) != zt: err.append('ZETA_T')
            elif typ == 'weyl':
                lam = v.weyl_transfer_lower(A0, A, lam_s)
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
        elif record['status'] == 'UNVERIFIED':
            if record.get('target_met') is not None: err.append('NONCERT_TARGET')
            if record.get('eps') is not None: err.append('NONCERT_EPS')
            if not record.get('reason'): err.append('NONCERT_REASON')
        elif record['status'] == 'HEURISTIC':
            if record.get('eps') is not None or record.get('target_met') is not None: err.append('HEURISTIC_CLAIMS')
        else:
            err.append('UNKNOWN_STATUS')
    except F3B.Unavailable as e:
        return 'UNAVAILABLE', ['MISSING_ARTIFACT:' + str(e)[-120:]]
    except Exception as e:
        err.append('VERIFY_EXCEPTION:' + repr(e)[:200])
    return ('VERIFIED' if not err else 'REJECTED'), err


CERT_FIELDS = ('status', 'reason', 'lambda_lo', 'R_hi', 'eps', 'target_met', 'matrix_sha256', 'rhs_sha256', 'x_sha256')


def cert_view(r, with_chain=True):
    """Certificate content without timing, run order, repeat index and artifact file names/hashes."""
    d = {k: r.get(k) for k in CERT_FIELDS}
    if with_chain:
        ch = []
        for l in r.get('lambda_chain') or []:
            ch.append({k: x for k, x in l.items() if k not in ('factor_file', 'factor_sha256')})
        d['chain'] = ch
        d['decision'] = (r['details'].get('decision_reason'), bool(r['details'].get('refresh')))
    return d
