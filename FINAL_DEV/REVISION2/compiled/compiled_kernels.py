"""Revision 2, exploratory (post hoc): compiled (numba) implementations of the two operations that the reviewer
identified as interpreted-Python overheads of the frozen methods:

  CompiledAssembler   the lean binding of the reuse arms (frozen PHASE5_DEV/lean/fastbind.FastAssembler): checks that
                      the stored CSR matrix equals the element-model floating-point assembly of the coefficients
                      bit for bit. Same memoized summation order, same products, same sequential sums, same
                      zero-dropping rule; the comparison runs in one compiled pass without temporary arrays.
  eft_residual_bound  the accurate residual enclosure (frozen PHASE10A_DEV/eft/eft_residual.eft_residual_bound) of the
                      shared residual strategy, used by per-system AND reuse arms. Same TwoProduct splitting, same
                      cascaded TwoSum order, same directed-rounding steps; the final sum of squares uses the frozen
                      NumPy routine, so the returned bound is BIT-IDENTICAL to the frozen one (checked by
                      test_kernels.py on held-out matrices and stored solutions) and the frozen replay verifies it.
  LightLayout         replaces the padded (K x n) EftLayout, which the compiled bound does not need.

Nothing in the frozen code is modified; install(...) swaps these objects into the frozen modules for one arm.
Compilation happens during the untimed warm-up (and is cached on disk)."""
import numpy as np
import numba as nb

U = 2.0 ** -53
ETA = 2.0 ** -1074
SPLITTER = 134217729.0
TP_MIN = 2.0 ** -969
MAXABS = 2.0 ** 995
INF = np.inf

v = None          # frozen cars._reference_m0 (set by install)
FA_FROZEN = None  # frozen FastAssembler class
EFT = None        # frozen eft_residual module
_FROZEN_EFT = None
_FROZEN_LAYOUT = None


# ----------------------------------------------------------------------------------------------- binding kernels
@nb.njit(cache=True)
def _assemble_data(coeff, Kflat, elem, loc, starts, lens, fsrc):
    nseg = starts.size
    acc = np.empty(nseg)
    for s in range(nseg):
        f = starts[s]
        a = coeff[elem[f]] * Kflat[loc[f]]
        for k in range(1, lens[s]):
            a += coeff[elem[f + k]] * Kflat[loc[f + k]]      # sequential left-to-right, as csr_sum_duplicates
        acc[s] = a
    out = np.empty(fsrc.size)
    for p in range(fsrc.size):
        out[p] = acc[fsrc[p]]
    return out


@nb.njit(cache=True)
def _matches(d, f_indptr, f_indices, a_indptr, a_indices, a_data, n):
    """True iff (a_indptr, a_indices, a_data) equal the assembly d on pattern f with exact zeros dropped."""
    q = 0
    nnz_a = a_indices.size
    for i in range(n):
        if a_indptr[i] != q:
            return False
        for p in range(f_indptr[i], f_indptr[i + 1]):
            val = d[p]
            if val != 0.0:
                if q >= nnz_a or a_indices[q] != f_indices[p] or a_data[q] != val:
                    return False
                q += 1
    return a_indptr[n] == q and q == nnz_a


def make_assembler_class(FA):
    class CompiledAssembler(FA):
        """Frozen FastAssembler plus a compiled data()/matches(); identical results (see test_kernels.py)."""

        def __init__(self, mesh, Kref, n):
            super().__init__(mesh, Kref, n)
            m2 = int(self.Kflat.size)
            src = np.flatnonzero(self.keep)[self.order]        # flat (element, local) index of each sorted slot
            self.elem = (src // m2).astype(np.int64); self.loc = (src % m2).astype(np.int64)
            self.starts64 = self.starts.astype(np.int64); self.lens64 = self.lens.astype(np.int64)
            self.fsrc64 = self.fsrc.astype(np.int64)

        def data(self, coeff):
            return _assemble_data(np.ascontiguousarray(coeff, dtype=np.float64), self.Kflat, self.elem, self.loc, self.starts64,
                                  self.lens64, self.fsrc64)

        def matches(self, A, coeff):
            if A.shape != (self.n, self.n):
                return False
            d = self.data(coeff)
            return bool(_matches(d, self.indptr, self.indices, A.indptr, A.indices, A.data, self.n))
    return CompiledAssembler


# ------------------------------------------------------------------------------------------------- EFT kernel
@nb.njit(cache=True)
def _eft_rows(indptr, indices, data, x, b, g2, gK, dn1gK, dn1u, N):
    """Per-row v_i of the frozen enclosure. status: 0 ok, 1 TWOPRODUCT_UNDERFLOW, 2 NONFINITE_PRODUCT,
    3 NONFINITE_SUM (some res_i not finite). Operations and their order follow eft_residual_bound exactly."""
    n = b.size
    vi = np.empty(n)
    under = False; nonfin = False; resbad = False
    for i in range(n):
        Se = 0.0; Eabs = 0.0; Pabs = 0.0
        s = b[i]; sig = 0.0
        for q in range(indptr[i], indptr[i + 1]):
            a = data[q]; xx = x[indices[q]]
            p = a * xx
            c = SPLITTER * a; ah = c - (c - a); al = a - ah
            c2 = SPLITTER * xx; bh = c2 - (c2 - xx); bl = xx - bh
            e = al * bl - (((p - ah * bh) - al * bh) - ah * bl)
            if (a != 0.0 and xx != 0.0 and p == 0.0) or (p != 0.0 and abs(p) <= TP_MIN):
                under = True
            if not (np.isfinite(p) and np.isfinite(e)):
                nonfin = True
            Se += e; Eabs += abs(e); Pabs += abs(p)
            pk = -p
            sn = s + pk; bb = sn - s
            t1 = s - (sn - bb); t2 = pk - bb; t1 = t1 + t2
            sig = sig + t1
            s = sn
        # two_sum(s, -Se)
        mSe = -Se
        s2 = s + mSe; bb = s2 - s
        t = (s - (s2 - bb)) + (mSe - bb)
        sig = sig + t
        res = s2 + sig
        if not np.isfinite(res):
            resbad = True
            vi[i] = np.nan
            continue
        Pabs_hi = np.nextafter(Pabs / dn1gK, INF)
        T = np.nextafter(np.nextafter(abs(b[i]) + Pabs_hi, INF) + abs(Se), INF)
        Eabs_hi = np.nextafter(Eabs / dn1gK, INF)
        term = np.nextafter(g2 * T, INF)
        num = np.nextafter(abs(res) + term, INF)
        w = np.nextafter(num / dn1u, INF)
        w = np.nextafter(w + np.nextafter(gK * Eabs_hi, INF), INF)
        w = np.nextafter(w + N * ETA, INF)
        vi[i] = w
    status = 1 if under else (2 if nonfin else (3 if resbad else 0))
    return status, vi


class LightLayout:
    """Pattern-only layout object for the runner's lazily built 'layout' (the compiled bound needs no padding)."""

    def __init__(self, A):
        self.n = A.shape[0]; self.indptr = A.indptr.copy(); self.indices = A.indices.copy()
        cnt = np.diff(A.indptr); self.K = int(cnt.max()) if cnt.size else 0

    def matches(self, A):
        return (A.shape[0] == self.n and np.array_equal(A.indptr, self.indptr)
                and np.array_equal(A.indices, self.indices))


def eft_residual_bound(A, b, x, layout=None, mutant=None):
    """Compiled version of the frozen eft_residual_bound (mutant must be None); returns (R, info) identically."""
    if mutant is not None:
        return _FROZEN_EFT(A, b, x, layout, mutant)
    A = v.canonical(A)                                               # as the frozen function (cheap if canonical)
    b = np.asarray(b, dtype=np.float64); x = np.asarray(x, dtype=np.float64)
    reason = None
    if not (np.all(np.isfinite(A.data)) and np.all(np.isfinite(b)) and np.all(np.isfinite(x))):
        reason = 'NONFINITE_INPUT'
    elif (A.data.size and np.max(np.abs(A.data)) >= MAXABS) or (x.size and np.max(np.abs(x)) >= MAXABS):
        reason = 'SPLIT_OVERFLOW_RISK'
    if reason is None:
        cnt = np.diff(A.indptr); K = int(cnt.max()) if cnt.size else 0
        N = K + 2
        gN = v.gamma_hi(float(max(N - 1, 1))); g2 = float(v.up(gN * gN))
        gK = float(v.gamma_hi(float(max(K - 1, 1))))
        dn1gK = float(v.dn(1.0 - gK)); dn1u = float(v.dn(1.0 - U))
        status, vi = _eft_rows(A.indptr, A.indices, A.data, np.ascontiguousarray(x), np.ascontiguousarray(b),
                               g2, gK, dn1gK, dn1u, float(N))
        if status == 1:
            reason = 'TWOPRODUCT_UNDERFLOW'
        elif status == 2:
            reason = 'NONFINITE_PRODUCT'
        elif status == 3:
            return float(v.residual_bound(A, b, x)), {'kind': 'frozen-fallback', 'reason': 'NONFINITE_SUM'}
    if reason is not None:
        return float(v.residual_bound(A, b, x)), {'kind': 'frozen-fallback', 'reason': reason}
    R = float(v.up(np.sqrt(v.sum_nonneg_hi(v.up(vi * vi)))))       # frozen final reduction (pairwise np.sum)
    if not np.isfinite(R):
        return float(v.residual_bound(A, b, x)), {'kind': 'frozen-fallback', 'reason': 'NONFINITE_BOUND'}
    return R, {'kind': 'eft', 'N': N}


# ------------------------------------------------------------------------------------------------- install
def install(runner10a_module, runner6_module, eft_module, ref_module):
    """Bind to the frozen modules (once)."""
    global v, FA_FROZEN, EFT, _FROZEN_EFT, _FROZEN_LAYOUT, CompiledAssembler
    v = ref_module; EFT = eft_module
    FA_FROZEN = runner10a_module.FastAssembler
    _FROZEN_EFT = eft_module.eft_residual_bound; _FROZEN_LAYOUT = eft_module.EftLayout
    CompiledAssembler = make_assembler_class(FA_FROZEN)
    globals()['CompiledAssembler'] = CompiledAssembler
    return CompiledAssembler


def use(kind, runner10a_module, runner6_module):
    """kind 'frozen' or 'compiled': swap binding and EFT implementations in the frozen modules for the next arm."""
    if kind == 'compiled':
        runner10a_module.FastAssembler = CompiledAssembler; runner6_module.FastAssembler = CompiledAssembler
        EFT.eft_residual_bound = eft_residual_bound; EFT.EftLayout = LightLayout
    else:
        runner10a_module.FastAssembler = FA_FROZEN; runner6_module.FastAssembler = FA_FROZEN
        EFT.eft_residual_bound = _FROZEN_EFT; EFT.EftLayout = _FROZEN_LAYOUT
