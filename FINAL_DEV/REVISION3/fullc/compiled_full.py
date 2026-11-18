"""Revision 3, exploratory (post hoc): compiled versions of the interpreted parts of building the shifted matrix for
Rump's criterion (frozen PHASE6_DEV/lean6/runner6.py: is_symmetric, diag_positions, and the copy with the
directed-rounding update of the diagonal in rump_anchor_once). Every Rump anchor uses them: 21 times per sequence for
the per-system arms, 4-6 times for the reuse arms.

  is_symmetric_c    exact symmetry of a canonical CSR matrix in one linear pass (for every row j a pointer walks the
                    entries right of the diagonal while the rows below are scanned in order); every stored (i, j, a)
                    must have a stored (j, i, a') with a' == a (IEEE comparison as in numpy.array_equal: NaN fails,
                    -0.0 == 0.0), and every entry must be matched. Same boolean as the frozen code.
  diag_positions_c  positions of the diagonal entries (int64), or None if one is missing; same result as the frozen code.
  shifted_copy_c    the shifted matrix of the criterion: a copy of the values with a~_ii = dn(a_ii - s_hi) at the diagonal
                    positions (dn = nextafter towards -inf after rounding to nearest, as in the frozen code); the index
                    arrays are shared with A instead of copied (they are never modified). Values are bit-identical.
  rump_anchor_once  rebuilt from the source of the frozen function with only the two lines of the shift block replaced
                    (assert: exactly one occurrence); everything else, including the Rump constant, is the frozen code.

Levels for use(): 'frozen', 'partial' (Revision 2: compiled binding and accurate residual enclosure), 'full' (partial +
the kernels above). Nothing in the frozen files is modified."""
import inspect
import sys
from pathlib import Path
import numpy as np
import numba as nb
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'REVISION2' / 'compiled'))
import compiled_kernels as CK                                        # noqa: E402  Revision 2 kernels

_FROZEN = {}
NEG_INF = -np.inf
FROZEN_SHIFT = """    with tm('shift'):
        At = A.copy(); At.data[pos] = v.dn(At.data[pos] - s_hi)   # a~_ii <= a_ii - s_hi exactly (Lemma V1)
"""
COMPILED_SHIFT = """    with tm('shift'):
        At = shifted_copy_c(A, pos, s_hi)                          # REVISION 3: compiled, bit-identical values
"""


@nb.njit(cache=True)
def _is_symmetric(indptr, indices, data, n):
    ptr = np.empty(n, np.int64)
    for j in range(n):                       # first entry right of the diagonal in row j
        q = indptr[j]
        while q < indptr[j + 1] and indices[q] <= j:
            q += 1
        ptr[j] = q
    for i in range(n):
        for q in range(indptr[i], indptr[i + 1]):
            j = indices[q]
            if j >= i:
                if j == i and not (data[q] == data[q]):   # the frozen comparison of A with A^T fails on NaN
                    return False
                break
            p = ptr[j]                       # the next unmatched entry right of the diagonal in row j
            if p >= indptr[j + 1] or indices[p] != i or not (data[p] == data[q]):
                return False
            ptr[j] = p + 1
    for j in range(n):                       # every entry right of the diagonal must have been matched
        if ptr[j] != indptr[j + 1]:
            return False
    return True


@nb.njit(cache=True)
def _diag_positions(indptr, indices, n):
    pos = np.empty(n, np.int64)
    for i in range(n):
        found = -1
        for q in range(indptr[i], indptr[i + 1]):
            c = indices[q]
            if c == i:
                found = q
                break
            if c > i:
                break
        if found < 0:
            return pos, False
        pos[i] = found
    return pos, True


@nb.njit(cache=True)
def _shifted_data(data, pos, s_hi):
    out = data.copy()
    for k in range(pos.size):
        p = pos[k]
        out[p] = np.nextafter(data[p] - s_hi, NEG_INF)
    return out


def is_symmetric_c(A):
    if not (A.has_sorted_indices and A.has_canonical_format) or A.shape[0] != A.shape[1]:
        return _FROZEN['is_symmetric'](A)
    return bool(_is_symmetric(A.indptr, A.indices, A.data, A.shape[0]))


def diag_positions_c(A):
    if not (A.has_sorted_indices and A.has_canonical_format):
        return _FROZEN['diag_positions'](A)
    pos, ok = _diag_positions(A.indptr, A.indices, A.shape[0])
    return pos if ok else None


def shifted_copy_c(A, pos, s_hi):
    At = sp.csr_matrix((_shifted_data(A.data, np.asarray(pos, np.int64), float(s_hi)), A.indices, A.indptr), shape=A.shape)
    At.has_sorted_indices = True
    return At


def build_anchor(R6):
    src = inspect.getsource(_FROZEN['rump_anchor_once'])
    assert src.count(FROZEN_SHIFT) == 1, 'frozen shift block not found exactly once'
    new = src.replace(FROZEN_SHIFT, COMPILED_SHIFT)
    ns = dict(R6.__dict__)
    ns.update(is_symmetric=is_symmetric_c, diag_positions=diag_positions_c, shifted_copy_c=shifted_copy_c)
    exec(compile(new, '<rump_anchor_once_compiled>', 'exec'), ns)
    return ns['rump_anchor_once'], src, new


def install(N, R6, EFT, v):
    CK.install(N, R6, EFT, v)
    _FROZEN.update(is_symmetric=R6.is_symmetric, diag_positions=R6.diag_positions, rump_anchor_once=R6.rump_anchor_once)
    _FROZEN['anchor_c'], _FROZEN['src_old'], _FROZEN['src_new'] = build_anchor(R6)


def use(level, N, R6):
    """level: 'frozen', 'partial' (binding + EFT compiled) or 'full' (partial + shift kernels)."""
    CK.use('compiled' if level in ('partial', 'full') else 'frozen', N, R6)
    if level == 'full':
        R6.is_symmetric = is_symmetric_c; R6.diag_positions = diag_positions_c; R6.rump_anchor_once = _FROZEN['anchor_c']
    else:
        R6.is_symmetric = _FROZEN['is_symmetric']; R6.diag_positions = _FROZEN['diag_positions']
        R6.rump_anchor_once = _FROZEN['rump_anchor_once']
