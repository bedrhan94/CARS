"""Deterministic public-formula-generated benchmark families A1/B1/D1.
No empirical source matrices are claimed here: generated, not downloaded datasets.
"""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from ._reference_m0 import canonical, matrix_sha256


def _kappa(x, y, t):
    return float(np.exp(0.6*np.sin(2*np.pi*(x+0.02*t))*np.sin(np.pi*y)+0.01*t*x))


def _coo_from_edges(n, entries):
    rows, cols, vals = [], [], []
    for i, j, v in entries:
        rows.append(i); cols.append(j); vals.append(float(v))
    return canonical(sp.coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsr())


def a1_fv(N, t):
    h=1./(N+1); n=N*N; edges=[]
    def add_edge(a,b,w):
        if a is not None: edges.append((a,a,w))
        if b is not None: edges.append((b,b,w))
        if a is not None and b is not None:
            edges.append((a,b,-w));edges.append((b,a,-w))
    for i in range(N):
        for j in range(N+1):
            a=i*N+j-1 if j>0 else None
            b=i*N+j if j<N else None
            add_edge(a,b,_kappa((j+0.5)*h,(i+1)*h,t)/h**2)
    for j in range(N):
        for i in range(N+1):
            a=(i-1)*N+j if i>0 else None
            b=i*N+j if i<N else None
            add_edge(a,b,_kappa((j+1)*h,(i+0.5)*h,t)/h**2)
    A=_coo_from_edges(n,edges)
    return A, np.ones(n,dtype=np.float64)


def _q1_local(hx,hy,kappa):
    gp=[0.5-0.5/np.sqrt(3),0.5+0.5/np.sqrt(3)]
    K=np.zeros((4,4), dtype=np.float64)
    for xi in gp:
        for eta in gp:
            gx=np.array([-(1-eta),(1-eta),eta,-eta],dtype=float)/hx
            gy=np.array([-(1-xi),-xi,xi,(1-xi)],dtype=float)/hy
            K += (0.25*kappa*hx*hy)*(np.outer(gx,gx)+np.outer(gy,gy))
    return K


def b1_q1(nx,t):
    ny=3*nx; hx=1./nx; hy=1./ny; n=(nx-1)*(ny-1)
    triples=[]
    def inside(i,j):
        return (j-1)*(nx-1)+(i-1) if 1<=i<nx and 1<=j<ny else None
    for ey in range(ny):
        for ex in range(nx):
            k=_kappa((ex+0.5)*hx,(ey+0.5)*hy,t)
            K=_q1_local(hx,hy,k)
            nodes=[inside(ex,ey),inside(ex+1,ey),inside(ex+1,ey+1),inside(ex,ey+1)]
            for i,a in enumerate(nodes):
                if a is not None:
                    for j,b in enumerate(nodes):
                        if b is not None: triples.append((a,b,K[i,j]))
    A=_coo_from_edges(n,triples)
    return A,np.ones(n,dtype=np.float64)


def d1_pd_loss(N,t,T=20):
    e=np.ones(N)
    Q=sp.diags((-e[:-1],2*e,-e[:-1]),(-1,0,1),format='csr')
    I=sp.eye(N,format='csr')
    lam1=8*np.sin(np.pi/(2*(N+1)))**2
    c=float(lam1*(0.5-3*t/T))
    A=canonical(sp.kron(Q,I,format='csr')+sp.kron(I,Q,format='csr')+c*sp.eye(N*N,format='csr'))
    return A,np.ones(N*N,dtype=np.float64)


@dataclass(frozen=True)
class MatrixSequence:
    family: str
    size: int
    T: int=10
    seed: int=20261112
    def __post_init__(self):
        if self.family not in {'A1','B1','D1'}: raise ValueError('Unsupported family')
        if self.size<3: raise ValueError('size must be >= 3')
    def get(self,t):
        if not 0<=t<=self.T: raise IndexError(t)
        if self.family=='A1': A,b=a1_fv(self.size,t)
        elif self.family=='B1': A,b=b1_q1(self.size,t)
        else: A,b=d1_pd_loss(self.size,t,self.T)
        coo=sp.triu(A,k=1).tocoo()
        pos=int(np.count_nonzero(coo.data>0))
        m={'family':self.family,'size':self.size,'seed':self.seed,'t':t,'T':self.T,
           'n':A.shape[0],'nnz':A.nnz,'positive_upper_offdiagonals':pos,
           'matrix_sha256':matrix_sha256(A)}
        return A,b,m
    def hash(self,t): return self.get(t)[2]['matrix_sha256']
    def info(self): return {'family':self.family,'size':self.size,'T':self.T,'seed':self.seed}
