"""Phase 10A record verifier.
Frozen arms (A = RPS|P1F, C = RPS|M2A): the Phase 5 verifier (PHASE5_DEV/lean/verify5.py), imported unchanged.
New arms (RPS-EFT, RPS-OPT, M2A-EFT, M2A-EFT-S05): the same rules R-1..R-7 as verify5, except R-6:
  R-1 identity: matrix / rhs / solution hashes (solution file hash and content hash)
  R-2 a 'rump-isspd' link certifies only its own matrix (anchor_t == t, same SHA-256)
  R-3 Rump c and s_hi recomputed bit-exactly from the regenerated anchor matrix
  R-4 cross-engine re-proof: the Cholesky of A~ is re-run with CHOLMOD (Rump 2006 Cor. 2.4)
  R-5 m2a transfers recomputed from the frozen formulas (theta, zeta_s, zeta_t, gray-box assembly premise)
  R-6' residual: the rigorous EFT bound is recomputed from (A, b, x) and must equal the recorded R bitwise (or, for a
       recorded 'frozen-fallback', the frozen bound must not exceed it); eps = up(R / lambda_lo) and target_met checked
  R-7 exact rational residual and exact positive definiteness of A - lambda_lo I (FLINT fmpq elimination), for
      n <= exact_n, applied to EVERY arm (for A/C after the unchanged frozen verifier)
Outcomes: VERIFIED / REJECTED / UNAVAILABLE."""
from __future__ import annotations
import sys
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
for p in (CARS / 'PHASE5_DEV' / 'lean', CARS / 'PHASE5_DEV' / 'arith', HERE.parent / 'eft'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import verify5 as V5                                              # noqa: E402  unchanged
import eft_residual as EFT                                        # noqa: E402
import independent5 as I5                                         # noqa: E402  fmpq exact elimination (Phase 5)
from cars import _reference_m0 as v                               # noqa: E402
from cars.verify import exact_inertia, exact_residual2            # noqa: E402
from cars_m2 import verify_m2_3b as F3B                           # noqa: E402
from m2_ref import assembly_discrepancy_rowsum, m2a_transfer      # noqa: E402

NEW = ('RPS-EFT', 'RPS-OPT', 'RPS-OPT-S09', 'M2A-EFT', 'M2A-EFT-S05')


def exact_checks(record, A, b, x):
    """Exact rational checks (FLINT fmpq elimination; Phase 5 independent5): A - lambda_lo I positive definite and
    ||b - A x||^2 <= R^2. Returns a list of error codes."""
    err = []
    lo = float.fromhex(record['lambda_lo']); R = float.fromhex(record['R_hi'])
    if Fraction(float(R)) ** 2 < I5.exact_residual_sq(A, b, x): err.append('EXACT_RESIDUAL')
    pd, _ = I5.exact_pd(A, lo)
    if not pd: err.append('EXACT_LAMBDA')
    return err


def verify_record10a(record, family, base_dir, exact_n=0):
    if record['method'] not in NEW:
        st, err = V5.verify_record5(record, family, base_dir, exact_n=0)      # frozen verifier, unchanged
        if st == 'VERIFIED' and exact_n and record['status'] == 'CERTIFIED':
            A, b, _ = family.get(int(record['t'])); A = V5.R.ensure_canonical(A)
            if A.shape[0] <= exact_n:
                x = np.load(F3B._inside(Path(base_dir), record['solution_file']), allow_pickle=False)
                e2 = exact_checks(record, A, b, x)
                if e2:
                    return 'REJECTED', e2
        return st, err
    err = []; out = Path(base_dir); chain = record.get('lambda_chain') or []
    try:
        t = int(record['t'])
        A, b, meta = family.get(t)
        A = V5.R.ensure_canonical(A)
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
            if link.get('kind') != 'rump':
                raise ValueError('UNKNOWN_ANCHOR_KIND')
            lam_s, A0 = V5.reprove_rump(link, family)
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
                if v.matrix_sha256(V5.R.assemble_float(family.mesh, family.Z, ct, family.Kref)) != v.matrix_sha256(A):
                    err.append('ASSEMBLY_T')
                if v.matrix_sha256(V5.R.assemble_float(family.mesh, family.Z, cs, family.Kref)) != link['anchor_matrix_sha256']:
                    err.append('ASSEMBLY_S')
                zs = assembly_discrepancy_rowsum(family.mesh, family.Z, cs)
                zt = assembly_discrepancy_rowsum(family.mesh, family.Z, ct)
                lam, xi = m2a_transfer(family.mesh, family.Z, cs, ct, lam_s, zs, zt)
                if float.fromhex(link['theta']) != xi['theta']: err.append('THETA')
                if float.fromhex(link['zeta_s']) != zs: err.append('ZETA_S')
                if float.fromhex(link['zeta_t']) != zt: err.append('ZETA_T')
            else:
                raise ValueError('UNKNOWN_CHAIN_' + str(typ))
            lo = float.fromhex(record['lambda_lo'])
            if not (np.isfinite(lo) and lo > 0 and lo <= lam): err.append('UNSUPPORTED_LAMBDA_LO')
            R = float.fromhex(record['R_hi'])
            if record.get('residual_kind') == 'eft':
                R2, info = EFT.eft_residual_bound(A, b, x)
                if info['kind'] != 'eft' or R2 != R: err.append('EFT_RECOMPUTE')
            else:
                if v.residual_bound(A, b, x) > R: err.append('RESIDUAL_RECOMPUTE')
            eps = float.fromhex(record['eps']); tar = float.fromhex(record['target'])
            if v.up(R / lo) != eps: err.append('EPS')
            if record['target_met'] != bool(eps <= tar): err.append('TARGET_MET')
            if exact_n and A.shape[0] <= exact_n:
                err += exact_checks(record, A, b, x)
        elif record['status'] == 'UNVERIFIED':
            if record.get('target_met') is not None: err.append('NONCERT_TARGET')
            if record.get('eps') is not None: err.append('NONCERT_EPS')
            if not record.get('reason'): err.append('NONCERT_REASON')
        else:
            err.append('UNKNOWN_STATUS')
    except F3B.Unavailable as e:
        return 'UNAVAILABLE', ['MISSING_ARTIFACT:' + str(e)[-120:]]
    except Exception as e:
        err.append('VERIFY_EXCEPTION:' + repr(e)[:200])
    return ('VERIFIED' if not err else 'REJECTED'), err
