"""Revision 1, exploratory (post hoc): PARDISO sessions for the structural-envelope experiment.

The frozen engine (PHASE6_DEV/lean6/runner6.PardisoSession) analyses the union of the sparsity patterns it has seen so
far and re-analyses whenever an incoming matrix has a stored entry outside that union (and after any factorization
error). Because the float assembly does not store entries that cancel to exactly zero, the stored pattern of a matrix
depends on its coefficients, and the union can keep growing along a sequence.

This module provides two subclasses; neither changes the frozen code or any arithmetic of a factorization:
  InstrSession   the frozen session unchanged, plus timing of PARDISO phases 11 (analysis) and 22 (numerical
                 factorization) and counts. Used for the re-measured ORIGINAL arms.
  StructSession  InstrSession whose union pattern is pre-seeded, before the first analysis, with the STRUCTURAL
                 ENVELOPE of the finite-element discretization: every (i, j) such that free degrees of freedom i and j
                 belong to a common element. The envelope is built from the mesh connectivity only (mesh.dofs), never
                 from coefficient values or benchmark outcomes. Every assembled matrix of the mesh is stored inside the
                 envelope, so the frozen embedding never fails and the frozen session analyses once per sequence (and
                 again only after a factorization error, as before). Entries of the envelope that a matrix does not
                 store are passed to PARDISO as explicit zeros, exactly as the frozen session already does for its
                 union pattern; the numerical matrix is unchanged (checked by check_envelope.py).
The envelope is constructed inside the first factorization call of each sequence, i.e. inside the frozen 'anchor_F'
timer, so its cost is charged to the structural arms; it is also recorded separately (t_envelope).
Activate with use(kind, mesh) before a sequence; read per-sequence statistics with take_stats()."""
import time
import numpy as np
import scipy.sparse as sp

R6 = None            # the frozen runner6 module, set by install()
_ORIG = None
KIND = 'orig'
MESH = None
REGISTRY = []


def envelope(mesh):
    """Structural envelope (CSR pattern, int64) of the free-dof couplings of the mesh, including the diagonal."""
    d = np.asarray(mesh.dofs)
    m = d.shape[1]
    rows = np.repeat(d, m, axis=1).ravel(); cols = np.tile(d, (1, m)).ravel()
    keep = (rows >= 0) & (cols >= 0)
    E = sp.csr_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(mesh.n, mesh.n))
    E.sum_duplicates(); E.sort_indices()
    return E.indptr.astype(np.int64), E.indices.astype(np.int64)


def install(r6_module):
    """Bind to the frozen runner6 module and define the session classes on top of its PardisoSession."""
    global R6, _ORIG, InstrSession, StructSession
    R6 = r6_module; _ORIG = R6.PardisoSession

    class InstrSession(_ORIG):
        def __init__(self):
            super().__init__()
            self.stats = dict(kind=KIND, sessions=1, analyses=0, t_analyse=0.0, numeric=0, t_numeric=0.0,
                              t_envelope=0.0, envelope_nnz=0, analysed_nnz=[])
            REGISTRY.append(self.stats)

        def _call(self, phase, b, x):
            t0 = time.perf_counter()
            err = _ORIG._call(self, phase, b, x)
            dt = time.perf_counter() - t0
            if phase == 11:
                self.stats['analyses'] += 1; self.stats['t_analyse'] += dt
                self.stats['analysed_nnz'].append(int(self.full_indices.size))
            elif phase == 22:
                self.stats['numeric'] += 1; self.stats['t_numeric'] += dt
            return err

    class StructSession(InstrSession):
        def __init__(self):
            super().__init__()
            t0 = time.perf_counter()
            ip, ix = envelope(MESH)
            self.n = MESH.n; self.full_indptr = ip; self.full_indices = ix        # seeds the frozen union pattern
            self.stats['t_envelope'] = time.perf_counter() - t0; self.stats['envelope_nnz'] = int(ix.size)

    globals()['InstrSession'] = InstrSession; globals()['StructSession'] = StructSession


def use(kind, mesh):
    """Select the session class used by the frozen _engine_factor for the next sequence ('orig' or 'struct')."""
    global KIND, MESH
    KIND = kind; MESH = mesh
    R6.PardisoSession = StructSession if kind == 'struct' else InstrSession


def take_stats():
    """Aggregate and clear the statistics of all sessions created since the last call."""
    out = dict(sessions=0, analyses=0, t_analyse=0.0, numeric=0, t_numeric=0.0, t_envelope=0.0, envelope_nnz=0,
               analysed_nnz=[])
    for s in REGISTRY:
        for k in ('sessions', 'analyses', 'numeric'):
            out[k] += s[k]
        for k in ('t_analyse', 't_numeric', 't_envelope'):
            out[k] += s[k]
        out['envelope_nnz'] = max(out['envelope_nnz'], s['envelope_nnz'])
        out['analysed_nnz'] += s['analysed_nnz']
    REGISTRY.clear()
    return out
