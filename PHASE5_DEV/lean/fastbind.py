"""Lean gray-box binding for M2-A (Phase 5). Reproduces fem3d_ref.assemble_float BIT-FOR-BIT by replaying SciPy's
own duplicate-summation order, which is a deterministic function of the index sequence only:
  coo_tocsr (counting sort by row, stable)  ->  csr_sort_indices (std::sort on (col, value) keyed on col only)
  ->  csr_sum_duplicates (sequential left-to-right sums)  ->  exact symmetrization (upper copied to lower).
The permutation is obtained once per mesh with tag values. The binding itself stays fail closed: the solved matrix
must be bit-identical to this assembly; any mismatch falls back to the frozen assembler (and then ASSEMBLY_MISMATCH).
The one-time map construction is charged to M2-A (component 'struct_setup')."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse import _sparsetools


class FastAssembler:
    def __init__(self, mesh, Kref, n):
        m = Kref.shape[0]
        rows = np.repeat(mesh.dofs, m, axis=1).ravel()
        cols = np.tile(mesh.dofs, (1, m)).ravel()
        keep = (rows >= 0) & (cols >= 0)
        self.keep = keep; self.Kflat = np.asarray(Kref, np.float64).ravel(); self.n = n
        r = rows[keep].astype(np.int32); c = cols[keep].astype(np.int32); nnz = r.size
        tags = np.arange(nnz, dtype=np.float64)                      # exact integers < 2^53
        indptr = np.empty(n + 1, np.int32); indices = np.empty(nnz, np.int32); data = np.empty(nnz, np.float64)
        _sparsetools.coo_tocsr(n, n, nnz, r, c, tags, indptr, indices, data)
        _sparsetools.csr_sort_indices(n, indptr, indices, data)
        order = data.astype(np.int64)                                # input position of each sorted slot
        rowof = np.repeat(np.arange(n), np.diff(indptr))
        newseg = np.ones(nnz, bool); newseg[1:] = (indices[1:] != indices[:-1]) | (rowof[1:] != rowof[:-1])
        starts = np.flatnonzero(newseg); lens = np.diff(np.append(starts, nnz))
        self.order, self.starts, self.lens = order, starts, lens
        self.maxlen = int(lens.max())
        srow, scol = rowof[starts], indices[starts]                 # canonical pattern of the summed matrix
        # final matrix: upper part (incl. diagonal) as summed; strict lower = transpose of strict upper (exact copy)
        up_mask = scol >= srow
        up_idx = np.flatnonzero(up_mask)
        lo_src = np.flatnonzero(scol > srow)
        frow = np.concatenate([srow[up_idx], scol[lo_src]]); fcol = np.concatenate([scol[up_idx], srow[lo_src]])
        fsrc = np.concatenate([up_idx, lo_src])
        key = frow.astype(np.int64) * n + fcol
        perm = np.argsort(key, kind='stable')
        self.fsrc = fsrc[perm]
        fr, fc = frow[perm], fcol[perm]
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(fr, minlength=n))]).astype(np.int32)
        self.indices = fc.astype(np.int32)

    def data(self, coeff):
        vals = (np.asarray(coeff, np.float64)[:, None] * self.Kflat[None, :]).ravel()[self.keep][self.order]
        acc = vals[self.starts].copy()
        for k in range(1, self.maxlen):                              # sequential left-to-right, as csr_sum_duplicates
            m = self.lens > k
            acc[m] += vals[self.starts[m] + k]
        return acc[self.fsrc]

    def assemble_arrays(self, coeff):
        """(indptr, indices, data) of the frozen assembly. Entries whose value is exactly zero are dropped, as
        SciPy's csr + csr in the frozen exact-symmetrization step drops them."""
        d = self.data(coeff)
        nz = d != 0
        if nz.all():
            return self.indptr, self.indices, d
        rowof = np.repeat(np.arange(self.n), np.diff(self.indptr))
        ip = np.concatenate([[0], np.cumsum(np.bincount(rowof[nz], minlength=self.n))]).astype(np.int32)
        return ip, self.indices[nz], d[nz]

    def matches(self, A, coeff):
        """Bit-identity of the CSR arrays of A with the element-model assembly of coeff."""
        if A.shape != (self.n, self.n):
            return False
        ip, ix, d = self.assemble_arrays(coeff)
        return bool(A.indptr.size == ip.size and A.indices.size == ix.size and np.array_equal(A.indptr, ip)
                    and np.array_equal(A.indices, ix) and np.array_equal(A.data, d))
