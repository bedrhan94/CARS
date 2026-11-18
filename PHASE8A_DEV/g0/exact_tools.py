"""Phase 8A exact-arithmetic toolkit (independent of all Phase 5-7 helper code).

Every binary64 number is a dyadic rational m * 2^e. All quantities that the audit calls "exact" are computed here
with integers (Python int / FLINT fmpz) or rationals (FLINT fmpq). No floating-point operation enters an exact value.
Upper bounds of square roots are rational and proved by integer comparison."""
import math
import numpy as np
import flint

Q = flint.fmpq
U_Q = Q(1, 2 ** 53)                       # unit roundoff u
ETA_Q = Q(1, 2 ** 1074)                   # smallest positive subnormal


def fq(x):
    """Exact rational value of a binary64 number (raises on inf/nan)."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError('non-finite value in exact conversion')
    n, d = x.as_integer_ratio()
    return Q(n, d)


def gamma_q(k):
    return Q(k) * U_Q / (1 - Q(k) * U_Q)


def scaled_ints(arr):
    """arr (float array) == ints * 2^E exactly. Returns (flat list of Python ints, E)."""
    a = np.ascontiguousarray(arr, dtype=np.float64).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError('non-finite value in exact conversion')
    m, e = np.frexp(a)
    mant = (m * 2.0 ** 53).astype(np.int64)            # exact: |m| in [0.5, 1) -> 53-bit integer
    ex = (e - 53).astype(np.int64)
    nz = mant != 0
    if not nz.any():
        return [0] * a.size, 0
    E = int(ex[nz].min())
    sh = (ex - E)
    sh[~nz] = 0
    out = [int(mi) << int(si) for mi, si in zip(mant.tolist(), sh.tolist())]
    return out, E


def fmpz_matrix(M):
    """Dense float matrix M -> (fmpz_mat Mi, E) with M = Mi * 2^E exactly."""
    M = np.asarray(M, dtype=np.float64)
    r, c = M.shape
    ints, E = scaled_ints(M)
    return flint.fmpz_mat(r, c, ints), E


def align(Mi, E, Enew):
    """Re-express Mi * 2^E as Mi' * 2^Enew (Enew <= E)."""
    assert Enew <= E
    return Mi * (1 << (E - Enew)) if E != Enew else Mi


def sumsq_ints(ints):
    return sum(i * i for i in ints)


def ceil_sqrt_int(N):
    """Smallest integer >= sqrt(N) for integer N >= 0."""
    if N <= 0:
        return 0
    r = math.isqrt(N)
    return r if r * r == N else r + 1


def sqrt_up_q(x):
    """Rational upper bound of sqrt(x) for rational x >= 0, relative excess < 2^-60 (proved by integer test)."""
    x = Q(x)
    if x <= 0:
        return Q(0)
    p, q = int(x.p), int(x.q)
    # sqrt(p/q) = sqrt(p q)/q ; scale for 64 extra bits of precision
    s = 1 << 128
    r = ceil_sqrt_int(p * q * s * s)
    ub = Q(r, q * s)
    assert ub * ub >= x
    return ub


def frob2_dense_exact(M):
    """Exact ||M||_F^2 (rational) of a float matrix/vector."""
    ints, E = scaled_ints(np.asarray(M, dtype=np.float64))
    return Q(sumsq_ints(ints)) * (Q(2) ** (2 * E))


def frob_up_q(M):
    return sqrt_up_q(frob2_dense_exact(M))


def fmpz_mat_frob2(Mi):
    """Exact sum of squares of the entries of an fmpz_mat (Python int)."""
    s = 0
    for v in Mi.entries():
        iv = int(v)
        s += iv * iv
    return s


def qfloat_dn(x):
    """Largest binary64 <= rational x (exact comparison)."""
    from fractions import Fraction
    x = Q(x)
    f = float(Fraction(int(x.p), int(x.q)))             # correctly rounded start value, then exact adjustment
    while fq(f) > x:
        f = float(np.nextafter(f, -np.inf))
    while True:
        g = float(np.nextafter(f, np.inf))
        if fq(g) <= x:
            f = g
        else:
            break
    return f


def qfloat_up(x):
    return -qfloat_dn(-Q(x))
