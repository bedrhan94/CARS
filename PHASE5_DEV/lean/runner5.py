"""CARS Phase 5 development runner (NEW file; the frozen Phase 3B/4 code is imported read-only, never modified).

Engines (certificate of lambda_min(A) for an anchor):
  M0L  lean M0: the frozen M0 a-posteriori certificate (SuperLU LDL^T candidate of A - sigma I + validated residual
       + Sylvester/Weyl), mathematically and bitwise identical to the frozen anchor; lean overheads only:
       direct CSR diagonal shift (bit-identical), array symmetry check, no diagnostics in the timed path, K3 with
       array-equality identity checks on SuperLU's own CSC L (same operator).
  RCH  Rump-type a-priori certificate (Rump 2006, BIT 46:433-452, Corollary 2.4 with bound I) using CHOLMOD
       simplicial LL^T (cholespy 2.2.0, final_ll = 1, NESDIS). lambda_min(A) > sigma if the floating-point Cholesky
       of A~ (a~_ij = a_ij, a~_ii <= a_ii - sigma - c) runs to completion; c >= gamma_{n+1}/(1-gamma_{n+1}) tr(A)
       + n*3(2n + max a_ii)*eta. No a-posteriori residual (V ~ 0).
  RPA  the same criterion with Intel oneMKL PARDISO, mtype = 2 (supernodal LL^T), 1 thread, no scaling/matching,
       no iterative refinement; fail closed on error, on perturbed pivots (iparm[13] != 0).
Methods: P1F (per-system, one factorization: certificate factor = preconditioner), P2F (per-system: certificate
  factor + unshifted direct solve), M2A (gray-box transfer, frozen theorem and refresh rules), WEYL (black-box anchored
  Weyl reuse), U-DIRECT / U-REUSE (UNCERTIFIED context only).
Accounting as frozen (research/17 s5): measured = all components except 'io'; counterfactual = measured - anchor_V
  - (estimate for per-system methods). New components: 'shift' (shifted-matrix construction and symmetry check,
  frozen: inside 'other'), 'struct_setup' (one-time gray-box map), 'binding' (gray-box bit-identity check; frozen:
  inside 'transfer'). All are counted in both totals.
"""
from __future__ import annotations
import ctypes, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

CARS = Path(__file__).resolve().parents[2]
_W = CARS / 'PHASE3B_DEV' / 'work' / 'CARS_M0_M1'
if str(_W) not in sys.path:
    sys.path.insert(0, str(_W))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from cars import _reference_m0 as v                     # noqa: E402  frozen
from cars.verify import file_sha256                     # noqa: E402  frozen
from cars_m2.families import Family3D                   # noqa: E402  frozen
from cars_m2.runner3b import _estimate, _pcg            # noqa: E402  frozen shared components
from m2_ref import assembly_discrepancy_rowsum, m2a_transfer   # noqa: E402  frozen
from fem3d_ref import assemble_float                    # noqa: E402  frozen
from fastbind import FastAssembler                      # noqa: E402

COMPONENTS = ('estimate', 'shift', 'anchor_F', 'anchor_V', 'precond_setup', 'struct_setup', 'binding', 'transfer',
              'solve', 'residual', 'other', 'io')
PER_SYSTEM = ('P1F', 'P2F')
CERTIFIED_METHODS = ('P1F', 'P2F', 'M2A', 'WEYL')
UNCERTIFIED = ('U-DIRECT', 'U-REUSE')


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


# ----------------------------------------------------------------------------- shared lean helpers
def diag_positions(A):
    """Positions of the diagonal entries in canonical CSR A, or None if one is structurally missing."""
    n = A.shape[0]
    rows = np.repeat(np.arange(n, dtype=A.indices.dtype), np.diff(A.indptr))
    pos = np.flatnonzero(A.indices == rows)
    return pos if pos.size == n else None


def is_symmetric(A):
    """Exact symmetry of canonical CSR A (no explicit zeros assumed; otherwise fails closed)."""
    T = A.T.tocsr(); T.sort_indices()
    return bool(np.array_equal(A.indptr, T.indptr) and np.array_equal(A.indices, T.indices)
                and np.array_equal(A.data, T.data))


def ensure_canonical(A):
    if sp.isspmatrix_csr(A) and A.dtype == np.float64 and A.has_canonical_format:
        return A
    return v.canonical(A)


def rump_constant(A):
    """c >= ||Delta(A~)||_2 for every A~ with the off-diagonal of A and 0 <= a~_ii <= a_ii (Rump 2006, Thm 2.3,
    bound I: s(i,j) <= n-1 => alpha_ij <= gamma_{n+1}; ||Delta||_2 <= gamma_{n+1}(1-gamma_{n+1})^{-1} tr(A) + n M eta,
    M = 3(2n + max a_ii)). Every operation is RN followed by nextafter-up (Lemma V1). Requires a_ii > 0."""
    n = A.shape[0]; d = A.diagonal()
    if not (np.all(np.isfinite(d)) and np.all(d > 0)):
        raise v.CertFail('RUMP_DIAGONAL_NONPOSITIVE')
    g = v.gamma_hi(float(n + 1))
    f = v.up(g / v.dn(1.0 - g))
    tr = v.sum_nonneg_hi(d)
    M = v.up(3.0 * v.up(2.0 * n + float(d.max())))
    c = v.up(v.up(f * tr) + v.up(v.up(float(n) * M) * v.ETA))
    return float(v.finite_or_fail(c, 'rump_c'))


# ----------------------------------------------------------------------------- PARDISO (direct ctypes)
class _MKL:
    lib = None

    @classmethod
    def get(cls):
        if cls.lib is None:
            import pypardiso
            ps = pypardiso.PyPardisoSolver()
            cls.lib = ps.libmkl
            f = cls.lib.pardiso
            f.argtypes = [ctypes.POINTER(ctypes.c_int64)] + [ctypes.POINTER(ctypes.c_int32)] * 5 + \
                         [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                          ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                          ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
            f.restype = None
            try:
                cls.lib.MKL_Set_Num_Threads(ctypes.c_int(1))
            except AttributeError:
                pass
        return cls.lib


class PardisoFactor:
    """Upper triangle of a symmetric CSR matrix, mtype 2. Raises CertFail('NOT_PD'/'PARDISO_ERROR')."""

    def __init__(self, A):
        lib = _MKL.get(); self._f = lib.pardiso
        U = sp.triu(A, format='csr'); U.sort_indices()
        self.n = A.shape[0]
        self.ia = np.ascontiguousarray(U.indptr, dtype=np.int32)
        self.ja = np.ascontiguousarray(U.indices, dtype=np.int32)
        self.a = np.ascontiguousarray(U.data, dtype=np.float64)
        self.pt = np.zeros(64, dtype=np.int64)
        ip = np.zeros(64, dtype=np.int32)
        ip[0] = 1; ip[1] = 2; ip[3] = 0; ip[4] = 0; ip[5] = 0; ip[7] = 0; ip[9] = 13; ip[10] = 0; ip[12] = 0
        ip[17] = -1; ip[20] = 0; ip[23] = 0; ip[24] = 0; ip[26] = 0; ip[27] = 0; ip[34] = 1; ip[36] = 0
        self.iparm = ip; self.alive = False
        err = self._call(12, np.zeros(self.n), np.zeros(self.n))
        if err != 0:
            self.release()
            raise v.CertFail('NOT_PD' if err == -4 else 'PARDISO_ERROR', f'error={err}')
        self.alive = True
        self.perturbed = int(self.iparm[13]); self.nnz_factor = int(self.iparm[17])
        if self.perturbed != 0:
            self.release()
            raise v.CertFail('PARDISO_PERTURBED_PIVOTS', str(self.perturbed))

    def _call(self, phase, b, x):
        err = ctypes.c_int32(0); i32 = ctypes.c_int32
        self._f(self.pt.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.byref(i32(1)), ctypes.byref(i32(1)),
                ctypes.byref(i32(2)), ctypes.byref(i32(phase)), ctypes.byref(i32(self.n)),
                self.a.ctypes.data, self.ia.ctypes.data_as(ctypes.POINTER(i32)),
                self.ja.ctypes.data_as(ctypes.POINTER(i32)), ctypes.POINTER(i32)(), ctypes.byref(i32(1)),
                self.iparm.ctypes.data_as(ctypes.POINTER(i32)), ctypes.byref(i32(0)), b.ctypes.data,
                x.ctypes.data, ctypes.byref(err))
        return err.value

    def solve(self, r):
        b = np.ascontiguousarray(r, dtype=np.float64).ravel(); x = np.zeros(self.n)
        err = self._call(33, b, x)
        if err != 0:
            raise RuntimeError(f'PARDISO solve error {err}')
        return x

    def release(self):
        if getattr(self, 'pt', None) is not None and np.any(self.pt):
            dummy = np.zeros(1)
            self._call(-1, dummy, dummy)
        self.alive = False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class PardisoSession:
    """Engine RPS (amendment A1): PARDISO mtype 2 with the analysis (phase 11: METIS ordering + symbolic) done once per
    sequence and role, numeric refactorization (phase 22) per matrix. The full CSR pattern is compared exactly with the
    analysed one at every call (re-analysis on mismatch). Any error releases the handle (next call re-analyses).
    Views of older factorizations are invalidated (solve on a stale view raises)."""

    def __init__(self):
        self._f = _MKL.get().pardiso; self.pt = None; self.gen = 0; self.analyses = 0

    def _reset(self):
        if getattr(self, 'pt', None) is not None and np.any(self.pt):
            d = np.zeros(1); self._call(-1, d, d)
        self.pt = None; self.gen += 1

    def _call(self, phase, b, x):
        return PardisoFactor._call(self, phase, b, x)

    def _embed(self, A):
        """Positions of A's stored entries inside the analysed union pattern P (CSR, sorted), or None if A has an
        entry outside P. O(nnz): rows with identical counts are mapped directly; others by per-row search."""
        P_ip, P_ix = self.full_indptr, self.full_indices
        if A.shape[0] != self.n:
            return None
        ca = np.diff(A.indptr); cp = np.diff(P_ip)
        if np.any(ca > cp):
            return None
        off = np.arange(A.indices.size) - np.repeat(A.indptr[:-1], ca)
        pos = np.repeat(P_ip[:-1], ca) + off
        for r in np.flatnonzero(ca != cp):
            a0, a1 = A.indptr[r], A.indptr[r + 1]; p0, p1 = P_ip[r], P_ip[r + 1]
            pos[a0:a1] = p0 + np.searchsorted(P_ix[p0:p1], A.indices[a0:a1])
        if pos.size and (pos.max() >= P_ix.size or not np.array_equal(P_ix[pos], A.indices)):
            return None
        return pos

    def _analyse(self, A):
        self._reset()
        if getattr(self, 'full_indptr', None) is not None and A.shape[0] == self.n:      # union with the old pattern
            P = sp.csr_matrix((np.ones(self.full_indices.size), self.full_indices, self.full_indptr), shape=A.shape)
            Q = sp.csr_matrix((np.ones(A.indices.size), A.indices, A.indptr), shape=A.shape)
            U0 = (P + Q).tocsr(); U0.sort_indices()
        else:
            U0 = sp.csr_matrix((np.ones(A.indices.size), A.indices.copy(), A.indptr.copy()), shape=A.shape)
        self.n = A.shape[0]
        self.full_indptr = U0.indptr.astype(np.int64); self.full_indices = U0.indices.astype(np.int64)
        rows = np.repeat(np.arange(self.n), np.diff(self.full_indptr)); self.upos = np.flatnonzero(self.full_indices >= rows)
        ip_u = np.concatenate([[0], np.cumsum(np.bincount(rows[self.upos], minlength=self.n))])
        self.ia = np.ascontiguousarray(ip_u, dtype=np.int32)
        self.ja = np.ascontiguousarray(self.full_indices[self.upos], dtype=np.int32)
        self.pt = np.zeros(64, dtype=np.int64)
        ip = np.zeros(64, dtype=np.int32)
        ip[0] = 1; ip[1] = 2; ip[9] = 13; ip[17] = -1; ip[34] = 1
        self.iparm = ip
        self.a = np.zeros(self.ja.size)
        err = self._call(11, np.zeros(self.n), np.zeros(self.n)); self.analyses += 1
        if err != 0:
            self._reset(); raise v.CertFail('PARDISO_ERROR', f'analysis error={err}')

    def factor(self, A):
        pos = self._embed(A) if self.pt is not None else None
        if pos is None:
            self._analyse(A)
            pos = self._embed(A)
            if pos is None:
                self._reset(); raise v.CertFail('PARDISO_ERROR', 'embedding failed')
        full = np.zeros(self.full_indices.size); full[pos] = A.data   # same real matrix, explicit zeros elsewhere
        self.a = np.ascontiguousarray(full[self.upos])
        self.gen += 1
        err = self._call(22, np.zeros(self.n), np.zeros(self.n))
        if err != 0:
            self._reset(); raise v.CertFail('NOT_PD' if err == -4 else 'PARDISO_ERROR', f'error={err}')
        if int(self.iparm[13]) != 0:
            k = int(self.iparm[13]); self._reset(); raise v.CertFail('PARDISO_PERTURBED_PIVOTS', str(k))
        return _PardisoView(self, self.gen)

    def __del__(self):
        try:
            self._reset()
        except Exception:
            pass


class _PardisoView:
    def __init__(self, sess, gen):
        self.sess = sess; self.gen = gen; self.n = sess.n; self.perturbed = 0; self.nnz_factor = int(sess.iparm[17])

    def solve(self, r):
        if self.sess.gen != self.gen:
            raise RuntimeError('STALE_PARDISO_FACTOR')
        b = np.ascontiguousarray(r, dtype=np.float64).ravel(); x = np.zeros(self.n)
        err = self.sess._call(33, b, x)
        if err != 0:
            raise RuntimeError(f'PARDISO solve error {err}')
        return x


SESSIONS = {}


def _engine_factor(engine, A, role):
    if engine == 'RPS':
        if role not in SESSIONS:
            SESSIONS[role] = PardisoSession()
        return SESSIONS[role].factor(A)
    return ENGINE_FACTOR[engine](A)


class CholmodFactor:
    def __init__(self, A):
        import cholespy
        n = A.shape[0]; self.n = n
        try:
            self.s = cholespy.CholeskySolverD(n, A.indptr.astype(np.int32), A.indices.astype(np.int32),
                                              np.ascontiguousarray(A.data), cholespy.MatrixType.CSR)
        except ValueError as e:
            raise v.CertFail('NOT_PD', str(e))

    def solve(self, r):
        x = np.zeros(self.n); self.s.solve(np.ascontiguousarray(r, dtype=np.float64).ravel(), x); return x

    def release(self):
        self.s = None


ENGINE_FACTOR = {'RCH': CholmodFactor, 'RPA': PardisoFactor}


# ----------------------------------------------------------------------------- anchors
def lean_k3(Lcsc, d, perm):
    """Frozen K3 operator (identical object), built from SuperLU's own CSC L; same identity checks by array equality."""
    if not np.all(d > 0):
        raise v.CertFail('PRECONDITIONER_NOT_SPD')
    n = Lcsc.shape[0]
    luL = sla.splu(Lcsc, permc_spec='NATURAL', diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    ar = np.arange(n)
    if not (np.array_equal(luL.perm_r, ar) and np.array_equal(luL.perm_c, ar)):
        raise v.CertFail('K3_STRUCTURE', 'non-identity permutation')
    Lp = luL.L; Lp.sort_indices(); Up = luL.U
    if not (np.array_equal(Lp.indptr, Lcsc.indptr) and np.array_equal(Lp.indices, Lcsc.indices)
            and np.array_equal(Lp.data, Lcsc.data)):
        raise v.CertFail('K3_STRUCTURE', 'L factor differs')
    if not (Up.nnz == n and np.array_equal(Up.indptr, np.arange(n + 1)) and np.array_equal(Up.indices, ar)
            and bool(np.all(Up.data == 1.0))):
        raise v.CertFail('K3_STRUCTURE', 'U factor is not the identity')
    P = sp.csr_matrix((np.ones(n), (perm, ar)), shape=(n, n)); PT = P.T.tocsr()
    dd = np.asarray(d, dtype=np.float64)

    def apply(r):
        y = luL.solve(P @ np.asarray(r, dtype=np.float64).ravel())
        z = luL.solve(y / dd, trans='T')
        return PT @ z
    return sla.LinearOperator((n, n), matvec=apply, dtype=np.float64)


def m0_lean_anchor(A, sigma, tm):
    """Frozen M0 anchor mathematics (cars_m2/anchor.py lines 15-52), lean overheads. Returns the frozen fields."""
    n = A.shape[0]; sigma = float(sigma)
    with tm('shift'):
        if not is_symmetric(A):
            raise v.CertFail('NOT_SYMMETRIC')
        dhi, dlo = v.two_sum(A.diagonal(), np.full(n, -sigma))
        pos = diag_positions(A)
        if pos is None:
            Mt = A.tolil(); Mt.setdiag(dhi); Mt = v.canonical(Mt)
        else:
            Mt = A.copy(); Mt.data[pos] = dhi
    with tm('anchor_F'):
        try:
            lu = sla.splu(Mt.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                          options=dict(SymmetricMode=True))
        except RuntimeError as e:
            raise v.CertFail('ANCHOR_FACTORIZATION_FAILED', str(e))
    with tm('anchor_V'):                                  # verbatim frozen validation
        if not np.array_equal(lu.perm_r, lu.perm_c): raise v.CertFail('PERM_NOT_SYMMETRIC')
        Lcsc = lu.L; Lcsc.sort_indices()
        L = v.canonical(Lcsc)
        if sp.triu(L, 1).nnz or not np.all(L.diagonal() == 1): raise v.CertFail('FACTOR_STRUCTURE')
        d = lu.U.diagonal().astype(np.float64)
        v.finite_or_fail(L.data, 'L'); v.finite_or_fail(d, 'D')
        if np.any(d == 0): raise v.CertFail('ZERO_PIVOT')
        nu = int(np.sum(d < 0))
        if nu > 0: raise v.CertFail('ANCHOR_INERTIA_MISMATCH', f'nu={nu} > 0')
        ar = np.arange(n); P = sp.csr_matrix((np.ones(n), (lu.perm_r, ar)), shape=(n, n))
        B = v.canonical(P @ Mt @ P.T); lo_p = P @ dlo
        C = v.canonical(L @ sp.diags(d) @ L.T)
        absL = abs(L)
        absC = v.canonical(absL @ sp.diags(np.abs(d)) @ absL.T)
        pat = (absL > 0).astype(np.float64)
        patC = v.canonical(pat @ pat.T); patC.data[:] = 1.0
        c = float(np.max(np.diff(L.indptr))); g = v.gamma_hi(c + 1.0); den = v.dn(1.0 - g)
        tiny = v.up(2 * c * v.ETA)
        T = v.canonical(absC + patC * tiny); T.data = v.up(T.data)
        T.data = v.up(T.data / den)
        EC = T.copy(); EC.data = v.up(EC.data * g)
        EC = v.canonical(EC + patC * tiny); EC.data = v.up(EC.data)
        Rt = v.canonical(B - C)
        Rbar = abs(Rt); Rbar.data = v.up(Rbar.data / v.dn(1.0 - v.U))
        Rbar = v.canonical(Rbar + EC); Rbar.data = v.up(Rbar.data)
        Rbar = v.canonical(Rbar + sp.diags(np.abs(lo_p))); Rbar.data = v.up(Rbar.data)
        delta = v.norm2_sym_hi(Rbar); rho = float(v.dn(sigma - delta)); v.finite_or_fail(rho, 'rho')
    return dict(kind='m0', nu=nu, rho=rho, lam_lo=rho, delta=delta, sigma=sigma, n=n,
                factor=dict(L=L, d=d, perm=lu.perm_r.copy()), Lcsc=Lcsc, lu=lu)


def rump_anchor_once(A, sigma, engine, tm):
    n = A.shape[0]; sigma = float(sigma)
    with tm('shift'):
        if not is_symmetric(A):
            raise v.CertFail('NOT_SYMMETRIC')
        pos = diag_positions(A)
        if pos is None:
            raise v.CertFail('RUMP_DIAGONAL_MISSING')
    with tm('anchor_V'):
        c = rump_constant(A)
        s_hi = float(v.up(sigma + c))                     # >= sigma + c
    with tm('shift'):
        At = A.copy(); At.data[pos] = v.dn(At.data[pos] - s_hi)   # a~_ii <= a_ii - s_hi exactly (Lemma V1)
    with tm('anchor_F'):
        fac = _engine_factor(engine, At, 'anchor')         # raises CertFail('NOT_PD') if Cholesky does not complete
    extra = {'pardiso_perturbed': getattr(fac, 'perturbed', None), 'nnz_factor': getattr(fac, 'nnz_factor', None)}
    return dict(kind='rump', engine=engine, lam_lo=sigma, rho=sigma, sigma=sigma, c=c, s_hi=s_hi, n=n, fac=fac,
                extra=extra)


def make_anchor(A, lam_tilde, engine, tm):
    """M0 retry rule: sigma = 0.9 lam~, on failure sigma /= 5, at most 3 retries; every attempt is charged."""
    sigma = 0.9 * lam_tilde; reasons = []
    for _ in range(4):
        try:
            if engine == 'M0L':
                anc = m0_lean_anchor(A, sigma, tm)
                if anc['rho'] > 0 and anc['nu'] == 0:
                    return anc, reasons
                reasons.append('RHO_NONPOSITIVE')
            else:
                return rump_anchor_once(A, sigma, engine, tm), reasons
        except v.CertFail as e:
            reasons.append(e.reason)
        sigma /= 5.0
    raise v.CertFail('ANCHOR_FACTORIZATION_FAILED', ','.join(reasons))


def precond_from_anchor(anc, tm):
    with tm('precond_setup'):
        n = anc['n']
        if anc['kind'] == 'm0':
            op = lean_k3(anc['Lcsc'], anc['factor']['d'], anc['factor']['perm']); info = {'kernel': 'K3-lean'}
        else:
            op = sla.LinearOperator((n, n), matvec=anc['fac'].solve, dtype=np.float64)
            info = {'kernel': anc['engine'] + '-LLT'}
    return op, info


def direct_solver(A, engine):
    if engine == 'M0L':
        lu2 = sla.splu(A.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'SymmetricMode': True})
        return lu2.solve, lu2
    fac = _engine_factor(engine, A, 'direct')
    return fac.solve, fac


class SharedPreconditioner5:
    """Track T2 for Phase 5: one method-independent preconditioner sequence (engine anchor of A_{5j}); its cost is
    reported separately and excluded from method totals (frozen T2 semantics)."""

    def __init__(self, family, engine):
        self.ops = {}; self.cost = 0.0; x = None; op = None; self.keep = []
        for j in range(0, family.T + 1, 5):
            A, _, _ = family.get(j)
            t0 = time.perf_counter()
            lam, x = _estimate(A, op, x)
            anc, _ = make_anchor(A, lam, engine, Timer())
            op, _ = precond_from_anchor(anc, Timer())
            self.cost += time.perf_counter() - t0
            self.ops[j] = op; self.keep.append(anc)

    def get(self, t):
        return self.ops[max(j for j in self.ops if j <= t)]


def _factor_content_sha(L, d, perm):
    h = hashlib.sha256()
    for a in (L.indptr.astype('<i8'), L.indices.astype('<i8'), L.data.astype('<f8'), np.asarray(d, '<f8'),
              np.asarray(perm, '<i8')):
        h.update(a.tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- the sequence runner
def run_sequence_5(family: Family3D, method: str, engine: str, output_dir, track='T1', binding='lean',
                   target=1e-8, shared=None, repeat=0, run_order=0, code_hash='', write_artifacts=True):
    if method not in CERTIFIED_METHODS + UNCERTIFIED:
        raise ValueError(method)
    if engine not in ('M0L', 'RCH', 'RPA', 'RPS'):
        raise ValueError(engine)
    for _s in list(SESSIONS.values()):
        _s._reset()
    SESSIONS.clear()
    if track == 'T2' and shared is None:
        raise ValueError('T2 needs a SharedPreconditioner5')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    art = out / 'artifacts'; art.mkdir(exist_ok=True)
    stem = f'{family.id}_{family.size_label}_{family.split}_{family.seed}_{engine}_{method}_{track}_{binding}_r{repeat}'
    dest = out / f'{stem}.jsonl'
    if dest.exists():
        dest.unlink()
    common = family.config['common']
    theta_min = float(common['theta_min']); itmax = float(common['it_max_factor'])
    st = dict(anc=None, op=None, A_s=None, c_s=None, z_s=None, anchor_t=None, link=None, it_ref=None,
              refresh_next=False, prev=None, evec=None, op_last=None, FA=None, direct=None)
    rows = []
    for t in range(family.T + 1):
        A, b, meta = family.get(t)                         # problem generation: untimed for every method
        n = A.shape[0]; tm = Timer(); begin = time.perf_counter()
        info = {'refresh': False, 'decision_reason': None}; sinfo = {'iters': 0, 'converged': False}
        x = np.zeros(n); lam = None; chain = []; new_anchor = None; heuristic = None; fcount = 0
        try:
            if not v.arith_check():
                raise v.CertFail('FTZ_DETECTED')
            A = ensure_canonical(A)
            reanchor = False
            if method in PER_SYSTEM:
                reanchor = True; info['decision_reason'] = 'per-system'
            elif method == 'U-DIRECT':
                reanchor = False
            elif method == 'U-REUSE':
                reanchor = (t % 5 == 0); info['decision_reason'] = 'fixed-schedule-uncertified'
            else:
                reanchor = st['anc'] is None or st['refresh_next']
                if st['refresh_next']:
                    info['decision_reason'] = 'pcg-iter-trigger'
                if not reanchor:
                    if method == 'M2A':
                        if family.mesh is None or meta['c'] is None:
                            raise v.CertFail('TRANSFER_DATA_INVALID', 'no element data')
                        with tm('binding'):
                            ok = False
                            if binding == 'lean':
                                ok = st['FA'].matches(A, meta['c'])
                                info['binding'] = 'lean' if ok else 'lean-miss'
                            if not ok:
                                own = v.canonical(assemble_float(family.mesh, family.Z, meta['c'], family.Kref))
                                if v.matrix_sha256(own) != v.matrix_sha256(A):
                                    raise v.CertFail('ASSEMBLY_MISMATCH')
                                info['binding'] = info.get('binding', 'frozen')
                        with tm('transfer'):
                            zt = assembly_discrepancy_rowsum(family.mesh, family.Z, meta['c'])
                            lam, xi = m2a_transfer(family.mesh, family.Z, st['c_s'], meta['c'], st['anc']['lam_lo'],
                                                   st['z_s'], zt)
                            info.update(theta=float(xi['theta']), wmax=float(xi['wmax']), zeta_t=zt)
                            info['bound_type'] = 'm2a'
                            if xi['theta'] < theta_min:
                                reanchor = True; info['decision_reason'] = 'theta-below-threshold'
                            elif not lam > 0:
                                reanchor = True; info['decision_reason'] = 'transfer-nonpositive'
                    else:   # WEYL (black-box)
                        with tm('transfer'):
                            lam = v.weyl_transfer_lower(st['A_s'], A, st['anc']['lam_lo']); info['bound_type'] = 'weyl'
                            if not lam > 0:
                                reanchor = True; info['decision_reason'] = 'weyl-invalid'
                if reanchor and not info['decision_reason']:
                    info['decision_reason'] = 'initial-anchor' if st['anc'] is None else 'refresh'
            if method == 'M2A' and st['FA'] is None and binding == 'lean':
                with tm('struct_setup'):
                    st['FA'] = FastAssembler(family.mesh, family.Kref, n)
            if reanchor:
                with tm('estimate'):
                    lam_t, ev = _estimate(A, st['op_last'], st['evec'])
                if not (np.isfinite(lam_t) and lam_t > 0):
                    raise v.CertFail('LAMBDA_NONPOSITIVE')
                old = st['anc']
                anc, retries = make_anchor(A, lam_t, engine, tm); fcount += 1 + len(retries)
                info['anchor_retries'] = retries
                if method == 'P2F' or track == 'T2':
                    op = None; info['kernel'] = 'none' if method == 'P2F' else f'{engine}-shared'
                else:
                    op, kinfo = precond_from_anchor(anc, tm); info.update(kinfo)
                st.update(anc=anc, op=op, A_s=A, c_s=meta['c'], anchor_t=t, it_ref=None, refresh_next=False, evec=ev)
                if op is not None:
                    st['op_last'] = op
                if old is not None and old.get('fac') is not None and old is not anc and method in PER_SYSTEM:
                    pass                                    # released by refcount when st['op_last'] moves on
                if method == 'M2A':
                    with tm('binding'):
                        ok = binding == 'lean' and st['FA'].matches(A, meta['c'])
                        if not ok:
                            own_s = v.canonical(assemble_float(family.mesh, family.Z, meta['c'], family.Kref))
                            if v.matrix_sha256(own_s) != v.matrix_sha256(A):
                                raise v.CertFail('ASSEMBLY_MISMATCH', 'anchor matrix is not the element model')
                    with tm('transfer'):
                        st['z_s'] = assembly_discrepancy_rowsum(family.mesh, family.Z, meta['c'])
                lam = anc['lam_lo']; info['refresh'] = True; info['bound_type'] = 'anchor'
                new_anchor = anc
                if method in UNCERTIFIED:
                    heuristic = lam_t
            if method in CERTIFIED_METHODS and not (lam is not None and np.isfinite(lam) and lam > 0):
                raise v.CertFail('LAMBDA_NONPOSITIVE')
            # ---- solve
            with tm('solve'):
                if method in ('P2F', 'U-DIRECT'):
                    solve, fobj = direct_solver(A, engine); fcount += 1
                    x = np.asarray(solve(b), np.float64); sinfo = dict(iters=0, converged=True, solver='direct')
                    st['op_last'] = sla.LinearOperator((n, n), matvec=solve, dtype=np.float64)
                    st['direct'] = fobj
                else:
                    if track == 'T2':
                        M = shared.get(t); sname = f'pcg-{engine}-shared'; st['op_last'] = M
                    else:
                        M = st['op']; sname = f'pcg-{engine}'
                    lam_stop = lam if method in CERTIFIED_METHODS else heuristic_lam(st, heuristic)
                    x, sinfo = _pcg(A, b, M, lam_stop, target, st['prev']); sinfo['solver'] = sname
            if method == 'M2A' and track == 'T1':
                if st['it_ref'] is None:
                    st['it_ref'] = max(1, sinfo['iters'])
                elif sinfo['iters'] > itmax * st['it_ref']:
                    st['refresh_next'] = True
            if method in CERTIFIED_METHODS:
                with tm('residual'):
                    rec = v.certificate(A, b, x, lam, True, target)
                if rec['status'] != 'CERTIFIED':
                    raise v.CertFail(rec['reason'] or 'CERT_FAILED')
            else:
                rec = {'schema': 'cars-cert/0.3', 'status': 'HEURISTIC', 'reason': 'UNCERTIFIED_METHOD',
                       'target_met': None, 'target': float(target).hex(), 'matrix_sha256': v.matrix_sha256(A),
                       'rhs_sha256': v.vec_sha256(b), 'x_sha256': v.vec_sha256(x),
                       'residual_norm_float': float(np.linalg.norm(b - A @ x))}
            st['prev'] = x.copy()
        except (v.CertFail, ValueError, RuntimeError, OverflowError, ZeroDivisionError, MemoryError) as e:
            reason = e.reason if isinstance(e, v.CertFail) else type(e).__name__
            rec = {'schema': 'cars-cert/0.3', 'status': 'UNVERIFIED', 'reason': reason, 'target_met': None,
                   'target': float(target).hex(), 'matrix_sha256': v.matrix_sha256(A), 'rhs_sha256': v.vec_sha256(b),
                   'x_sha256': v.vec_sha256(x)}
            info['failure_detail'] = str(e)[:300]; st['prev'] = None
        tm.t['other'] += max(0.0, time.perf_counter() - begin - sum(tm.t.values()))
        # ---- artifacts (timed as io; excluded from both totals)
        with tm('io'):
            sol_sha = None; xfile = None
            if write_artifacts:
                xfile = art / f'{stem}_x_t{t}.npy'; np.save(xfile, x, allow_pickle=False); sol_sha = file_sha256(xfile)
            if new_anchor is not None:
                if new_anchor['kind'] == 'm0':
                    f = new_anchor['factor']; csha = _factor_content_sha(f['L'], f['d'], f['perm'])
                    link = {'anchor_t': t, 'anchor_matrix_sha256': v.matrix_sha256(A), 'factor_content_sha256': csha}
                    if write_artifacts:
                        fp = art / f'{stem}_anchor_t{t}.npz'; Lc = f['L']
                        np.savez(fp, data=Lc.data, indices=Lc.indices, indptr=Lc.indptr, shape=np.array(Lc.shape),
                                 d=f['d'], perm=f['perm'], sigma=np.float64(new_anchor['sigma']),
                                 nu=np.int64(new_anchor['nu']))
                        link.update(factor_file=str(fp.relative_to(out)), factor_sha256=file_sha256(fp))
                    link['kind'] = 'm0'
                else:
                    link = {'kind': 'rump', 'anchor_t': t, 'anchor_matrix_sha256': v.matrix_sha256(A),
                            'engine': new_anchor['engine'], 'sigma': float(new_anchor['sigma']).hex(),
                            'c': float(new_anchor['c']).hex(), 's_hi': float(new_anchor['s_hi']).hex(),
                            **{k: x_ for k, x_ in new_anchor['extra'].items() if x_ is not None}}
                st['link'] = link
        if rec['status'] == 'CERTIFIED':
            base = dict(st['link']); typ = info.get('bound_type')
            if typ == 'anchor':
                base['type'] = 'anchor-inertia' if base['kind'] == 'm0' else 'rump-isspd'
                chain = [base]
            elif typ == 'm2a':
                link = dict(base); link['type'] = 'm2a'; link['anchor_type'] = 'anchor-inertia' if base['kind'] == 'm0' else 'rump-isspd'
                link.update(theta=float(info['theta']).hex(), zeta_s=float(st['z_s']).hex(),
                            zeta_t=float(info['zeta_t']).hex(), coeff_s_sha256=v.vec_sha256(st['c_s']),
                            coeff_t_sha256=v.vec_sha256(meta['c']), element_data_sha256=family.factor_hash,
                            split_theta0=False)
                chain = [link]
            elif typ == 'weyl':
                link = dict(base); link['type'] = 'weyl'; link['anchor_type'] = 'anchor-inertia' if base['kind'] == 'm0' else 'rump-isspd'
                chain = [link]
        meas = sum(tm.t[k] for k in COMPONENTS if k != 'io')
        cf = meas - tm.t['anchor_V'] - (tm.t['estimate'] if method in PER_SYSTEM else 0.0)
        if 'zeta_t' in info:
            info['zeta_t'] = float(info['zeta_t'])
        rec.update(family=family.id, method=method, engine=engine, seed=family.seed, split=family.split,
                   size=family.size_label, t=t, n=n, nnz=meta['nnz'], label=family.label,
                   info_model=('gray-box' if method == 'M2A' else 'black-box'), track=track, binding=binding,
                   repeat=repeat, run_order=run_order, code_hash=code_hash, lambda_chain=chain, solver=sinfo,
                   details=info, solution_file=(str(xfile.relative_to(out)) if xfile else None),
                   solution_sha256=sol_sha, coeff_sha256=meta.get('coeff_sha256'),
                   timing_s={**tm.t, 'measured': meas, 'counterfactual': cf}, factor_count=fcount,
                   phase='PHASE5-DEV')
        with dest.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + '\n')
        rows.append(rec)
    return dest, rows


def heuristic_lam(st, heuristic):
    """Uncertified methods stop with the estimate lam~ (research/17 s4.2), carried from the last estimate."""
    if heuristic is not None:
        st['lam_heur'] = heuristic
    return st.get('lam_heur')
