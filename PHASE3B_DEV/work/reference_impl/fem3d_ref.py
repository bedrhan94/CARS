"""
CARS Phase 3 -- reference generators for the 3D benchmark families (INFORMATIVE; normative text: research/17).

Element data are stored in exactly-PSD factor form: the element matrix of element e is DEFINED as
    K_e := c_e * Z Z^T        (exact real arithmetic; Z a frozen float array, c_e a float coefficient)
so that the exact system matrix A(c) = sum_e c_e Z_e Z_e^T is PSD by construction (M2-A / M2-C data model).
The float matrix seen by every solver is the assembled  Ahat = assemble(c)  (validated discrepancy in m2_ref).

Families (all on the unit cube, N elements per side, h = 1/N, trilinear Q1 hexahedra):
  'elas'  : 3D linear elasticity, E = c_e, nu = 0.3, clamped on the face x = 0 (non-M, non-H; see exp06 part D)
  'diff'  : 3D scalar diffusion on stretched hexahedra (hx : hy : hz = 1 : 1 : aspect), homogeneous Dirichlet
            on the whole boundary (aspect >= 2 gives positive off-diagonals)
This file contains no randomness; seeds enter only through coefficient schedules (research/17).
"""
import numpy as np
import scipy.sparse as sp

GP = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
# local node order (x fastest): (0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)
LOC = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], dtype=np.int64)


def _dN_ref(xi, eta, zeta):
    """8x3 derivatives of trilinear shape functions on [-1,1]^3."""
    s = 2.0 * LOC - 1.0                              # node signs in {-1,+1}
    g = np.empty((8, 3))
    g[:, 0] = s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zeta) / 8.0
    g[:, 1] = s[:, 1] * (1 + s[:, 0] * xi) * (1 + s[:, 2] * zeta) / 8.0
    g[:, 2] = s[:, 2] * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta) / 8.0
    return g


def elastic_D(nu=0.3):
    lam = nu / ((1 + nu) * (1 - 2 * nu)); mu = 1.0 / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    return D


def elastic_factor_raw(h, nu=0.3):
    """Natural factor of the 2x2x2-Gauss Q1 elasticity element on a cube of side h (E = 1): 24 x 48."""
    C = np.linalg.cholesky(elastic_D(nu))
    detJ = (h / 2.0) ** 3
    cols = []
    for zeta in GP:
        for eta in GP:
            for xi in GP:
                dN = _dN_ref(xi, eta, zeta) * (2.0 / h)     # physical gradients, 8x3
                B = np.zeros((6, 24))
                for a in range(8):
                    gx, gy, gz = dN[a]
                    B[0, 3 * a] = gx; B[1, 3 * a + 1] = gy; B[2, 3 * a + 2] = gz
                    B[3, 3 * a] = gy; B[3, 3 * a + 1] = gx
                    B[4, 3 * a + 1] = gz; B[4, 3 * a + 2] = gy
                    B[5, 3 * a] = gz; B[5, 3 * a + 2] = gx
                cols.append(np.sqrt(detJ) * (B.T @ C))
    return np.hstack(cols)


def elastic_factor_18(h, nu=0.3):
    """Compressed factor Z (24 x 18) with Z Z^T ~ K_ref (rank 18). The MODEL is defined by the returned Z,
    which must be frozen as data (hex floats) for cross-platform reproducibility (research/17 section 2)."""
    Zr = elastic_factor_raw(h, nu)
    w, V = np.linalg.eigh(Zr @ Zr.T)
    keep = w > 1e-10 * w.max()
    assert keep.sum() == 18, keep.sum()
    return V[:, keep] * np.sqrt(w[keep])


def diffusion_factor(hx, hy, hz):
    """Natural factor (8 x 24) of the 2x2x2-Gauss Q1 diffusion element on a box hx*hy*hz (kappa = 1)."""
    detJ = hx * hy * hz / 8.0
    cols = []
    for zeta in GP:
        for eta in GP:
            for xi in GP:
                dN = _dN_ref(xi, eta, zeta) * np.array([2.0 / hx, 2.0 / hy, 2.0 / hz])
                cols.append(np.sqrt(detJ) * dN)              # 8 x 3
    return np.hstack(cols)


class HexMesh:
    """Structured N x N x N (or Nx x Ny x Nz) hexahedral mesh of the unit cube with element dof maps."""

    def __init__(self, Nx, Ny, Nz, ndof_per_node, fixed):
        self.Nx, self.Ny, self.Nz, self.k = Nx, Ny, Nz, ndof_per_node
        nx, ny, nz = Nx + 1, Ny + 1, Nz + 1
        I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        node = lambda i, j, k: i + nx * (j + ny * k)
        fixed_node = np.zeros(nx * ny * nz, bool)
        fixed_node[node(I, J, K)[fixed(I, J, K)]] = True
        free = np.repeat(~fixed_node, ndof_per_node)
        self.free_map = -np.ones(free.size, np.int64)
        self.free_map[free] = np.arange(free.sum())
        self.n = int(free.sum())
        ei, ej, ek = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
        ei, ej, ek = ei.ravel(order="F"), ej.ravel(order="F"), ek.ravel(order="F")   # x fastest
        self.centers = np.stack([(ei + 0.5) / Nx, (ej + 0.5) / Ny, (ek + 0.5) / Nz], axis=1)
        nodes = np.stack([node(ei + a, ej + b, ek + c) for a, b, c in LOC], axis=1)   # ne x 8
        gd = (nodes[:, :, None] * ndof_per_node + np.arange(ndof_per_node)).reshape(len(ei), -1)
        self.dofs = self.free_map[gd]          # ne x (8k); -1 = eliminated (Dirichlet) dof
        self.ne = len(ei)


def elasticity_mesh(N):
    return HexMesh(N, N, N, 3, lambda I, J, K: I == 0)


def diffusion_mesh(M, aspect=3):
    """Nx = Ny = aspect*M, Nz = M on the unit cube: hz / hx = aspect. Use diffusion_factor(1/Nx, 1/Ny, 1/Nz)."""
    Nx = Ny = aspect * M
    Nz = M
    return HexMesh(Nx, Ny, Nz, 1, lambda I, J, K: (I == 0) | (J == 0) | (K == 0) |
                   (I == Nx) | (J == Ny) | (K == Nz))


def kref_float(Z):
    """fl(Z Z^T) (any r-term dot order), made exactly symmetric by mirroring the upper triangle.
    For bitwise cross-platform reproducibility use the FROZEN 'Kref' stored next to Z in fixtures/phase3."""
    K = Z @ Z.T
    return np.triu(K) + np.triu(K, 1).T


def assemble_float(mesh, Z, c, Kref=None):
    """Float assembly Ahat = sum_e fl(c_e * Kref) restricted to free dofs, symmetrized exactly, where
    Kref = fl(Z Z^T) (computed, or the frozen fixture value). Operation order is the one analysed in
    m2_ref.assembly_discrepancy (own-assembly mode)."""
    Kref = kref_float(Z) if Kref is None else np.asarray(Kref, dtype=np.float64)
    m = Kref.shape[0]
    rows = np.repeat(mesh.dofs, m, axis=1).ravel()
    cols = np.tile(mesh.dofs, (1, m)).ravel()
    vals = (np.asarray(c, float)[:, None] * Kref.ravel()[None, :]).ravel()
    keep = (rows >= 0) & (cols >= 0)
    A = sp.coo_matrix((vals[keep], (rows[keep], cols[keep])), shape=(mesh.n, mesh.n)).tocsr()
    A.sum_duplicates(); A.sort_indices()
    U = sp.triu(A).tocsr()
    A = (U + sp.triu(A, 1).T).tocsr()                         # exact symmetrization
    A.sum_duplicates(); A.sort_indices()
    return A


def lap7_3d(N, kappa=None):
    """7-point FV Laplacian on N^3 interior nodes (M-matrix control family). kappa: face weights fn or None."""
    e = np.ones(N)
    T = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1])
    I = sp.eye(N)
    A = sp.kron(sp.kron(T, I), I) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(I, I), T)
    return sp.csr_matrix(A)
