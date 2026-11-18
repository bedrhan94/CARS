"""Revision 2, exploratory (post hoc): the carried-shift arms with a FIXED estimate fallback (arms Bf, B9f).

Defect in the frozen arms B (RPS-OPT) and B9 (RPS-OPT-S09), observed only on the SIMP3D sequences: when the carried
shift fails (Rump's shifted Cholesky does not complete), the failed factorization resets the PARDISO session, the
previous preconditioner becomes stale, and the frozen code falls back to a COLD estimate of lambda_min
(ARPACK, which='SA', no shift-invert, maxiter 10 000). On the high-contrast SIMP matrices (E_min = 1e-9) this cold
Lanczos run did not converge, the step was UNVERIFIED, and the next carried shift failed again.

Fix (one change, nothing else): whenever a new estimate is needed after the first anchor of the sequence and no valid
preconditioner exists (stale or absent), lambda_min is estimated by the SAME five-step inverse iteration as the frozen
estimator (cars_m2.runner3b._estimate), with a fresh PARDISO factorization of the UNSHIFTED matrix A as the operator
(shift-invert with shift 0), started from the last eigenvector estimate. The factorization uses the anchor session
(same pattern, so no new symbolic analysis), is charged to the 'estimate' component and counted as a factorization.
If that factorization fails, the frozen cold estimate is used as before (recorded). The first anchor of a sequence,
the anchor rule (sigma = frac * l~, /5 retries), the carried shift, the residual strategy, the restoration rule and
all accounting are unchanged.

Implementation: the source of the frozen run_sequence_10a is read from the frozen module, the estimate block is
replaced textually (assert: exactly one occurrence), and the result is compiled into a copy of the frozen module
namespace as run_sequence_fb. The frozen files are not modified."""
import inspect
import numpy as np
import scipy.sparse.linalg as sla

FROZEN_BLOCK = """            with tm('estimate'):
                try:
                    lam_t, ev = _estimate(A, st['op_last'], st['evec'])
                except RuntimeError:            # previous preconditioner invalidated by a failed factorization
                    info['estimate_fallback'] = 'eigsh'          # (PARDISO session reset): start from scratch
                    st['op_last'] = None
                    lam_t, ev = _estimate(A, None, None)
"""
FIXED_BLOCK = """            with tm('estimate'):
                if st['anc'] is None:                             # first anchor of the sequence: frozen estimate
                    lam_t, ev = _estimate(A, st['op_last'], st['evec'])
                else:
                    try:
                        if st['op_last'] is None:
                            raise RuntimeError('NO_VALID_PRECONDITIONER')
                        lam_t, ev = _estimate(A, st['op_last'], st['evec'])
                    except RuntimeError:        # stale/absent preconditioner: REVISION 2 fixed fallback
                        st['op_last'] = None
                        try:
                            fac0 = R6._engine_factor('RPS', A, 'anchor'); fcount += 1     # unshifted A (shift-invert, 0)
                            op0 = sla.LinearOperator(A.shape, matvec=fac0.solve, dtype=np.float64)
                            lam_t, ev = _estimate(A, op0, st['evec'])
                            info['estimate_fallback'] = 'shift-invert'
                        except v.CertFail as e0:            # failed attempt is charged (time) and counted
                            fcount += 1
                            info['estimate_fallback'] = 'shift-invert-failed:' + str(e0.reason) + ';eigsh'
                            lam_t, ev = _estimate(A, None, None)
"""


def build(N):
    """Return run_sequence_fb, compiled from the frozen run_sequence_10a source with the one block replaced."""
    src = inspect.getsource(N.run_sequence_10a)
    assert src.count(FROZEN_BLOCK) == 1, 'frozen estimate block not found exactly once'
    new = src.replace(FROZEN_BLOCK, FIXED_BLOCK).replace('def run_sequence_10a(', 'def run_sequence_fb(', 1)
    ns = dict(N.__dict__); ns.setdefault('sla', sla); ns.setdefault('np', np)
    exec(compile(new, '<run_sequence_fb>', 'exec'), ns)
    return ns['run_sequence_fb'], src, new
