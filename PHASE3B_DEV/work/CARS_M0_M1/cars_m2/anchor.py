"""Instrumented copy of Phase 2A reference anchor: unchanged mathematics, separately timed F/V.

Provenance: reference_impl/cars_ref.py; numerical validity is conditional on the handoff A1-A5.
The existing cars/ package is never modified.
"""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from cars import _reference_m0 as v


def anchor_inertia_lu(A, sigma, k_max=0):
    A=v.canonical(A)
    if abs(A-A.T).nnz: raise v.CertFail('NOT_SYMMETRIC')
    n=A.shape[0]; sigma=float(sigma)
    dhi,dlo=v.two_sum(A.diagonal(),np.full(n,-sigma))
    Mt=A.tolil();Mt.setdiag(dhi);Mt=v.canonical(Mt)
    begin=time.perf_counter()
    try:
        lu=spla.splu(Mt.tocsc(),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.0,options=dict(SymmetricMode=True))
    except RuntimeError as e: raise v.CertFail('ANCHOR_FACTORIZATION_FAILED',str(e))
    tF=time.perf_counter()-begin
    begin=time.perf_counter()
    if not np.array_equal(lu.perm_r,lu.perm_c):raise v.CertFail('PERM_NOT_SYMMETRIC')
    L=v.canonical(lu.L)
    if sp.triu(L,1).nnz or not np.all(L.diagonal()==1):raise v.CertFail('FACTOR_STRUCTURE')
    d=lu.U.diagonal().astype(np.float64)
    v.finite_or_fail(L.data,'L');v.finite_or_fail(d,'D')
    if np.any(d==0):raise v.CertFail('ZERO_PIVOT')
    nu=int(np.sum(d<0))
    if nu>k_max:raise v.CertFail('ANCHOR_INERTIA_MISMATCH',f'nu={nu} > {k_max}')
    ar=np.arange(n);P=sp.csr_matrix((np.ones(n),(lu.perm_r,ar)),shape=(n,n))
    B=v.canonical(P@Mt@P.T);lo_p=P@dlo
    C=v.canonical(L@sp.diags(d)@L.T)
    absL=abs(L)
    absC=v.canonical(absL@sp.diags(np.abs(d))@absL.T)
    pat=(absL>0).astype(np.float64)
    patC=v.canonical(pat@pat.T);patC.data[:]=1.0
    c=float(np.max(np.diff(L.indptr)));g=v.gamma_hi(c+1.0);den=v.dn(1.0-g)
    tiny=v.up(2*c*v.ETA)
    T=v.canonical(absC+patC*tiny);T.data=v.up(T.data)
    T.data=v.up(T.data/den)
    EC=T.copy();EC.data=v.up(EC.data*g)
    EC=v.canonical(EC+patC*tiny);EC.data=v.up(EC.data)
    Rt=v.canonical(B-C)
    Rbar=abs(Rt);Rbar.data=v.up(Rbar.data/v.dn(1.0-v.U))
    Rbar=v.canonical(Rbar+EC);Rbar.data=v.up(Rbar.data)
    Rbar=v.canonical(Rbar+sp.diags(np.abs(lo_p)));Rbar.data=v.up(Rbar.data)
    delta=v.norm2_sym_hi(Rbar);rho=float(v.dn(sigma-delta));v.finite_or_fail(rho,'rho')
    tV=time.perf_counter()-begin
    return dict(nu=nu,rho=rho,delta=delta,sigma=sigma,n=n,
                maxL=float(abs(L).max()),pivot_ratio=float(np.abs(d).max()/np.abs(d).min()),
                factor=dict(L=L,d=d,perm=lu.perm_r.copy()),lu=lu,
                instrumentation=dict(F_s=tF,V_s=tV,nnz_L=L.nnz,sum_col_count_sq=int(np.square(np.diff(L.tocsc().indptr).astype(np.int64)).sum())))


def lambda_estimate(A):
    try:
        n=A.shape[0]
        if n<=3:return float(np.linalg.eigvalsh(A.toarray())[0])
        return float(spla.eigsh(A,k=1,which='SA',v0=np.linspace(1.0,1.1,n),tol=1e-6,maxiter=10000,return_eigenvectors=False)[0])
    except Exception as e:raise v.CertFail('LAMBDA_ESTIMATE_FAILED',str(e))


def make_anchor(A):
    # The guess is only a heuristic for choosing the shift, never used as a bound.
    start=time.perf_counter();guess=lambda_estimate(A);t_est=time.perf_counter()-start
    if not np.isfinite(guess) or guess<=0:raise v.CertFail('LAMBDA_NONPOSITIVE')
    sigma=0.9*guess; reasons=[]
    for _ in range(4):
        try:
            anchor=anchor_inertia_lu(A,sigma,0)
            if anchor['rho']>0 and anchor['nu']==0:
                anchor['instrumentation']['selection_s']=t_est
                return anchor
            reasons.append('RHO_NONPOSITIVE')
        except v.CertFail as e:reasons.append(e.reason)
        sigma/=5.0
    raise v.CertFail('ANCHOR_FACTORIZATION_FAILED',','.join(reasons))
