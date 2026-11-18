"""Evidence replayer: re-evaluates stored-factor proofs without calling SuperLU.

This uses the audited validated-arithmetic primitives from the supplied Phase-2A
reference. Therefore it is an independently invoked chain checker, NOT an
independent mathematical implementation of those primitives. Small systems
are also verified using exact rational inertia and residual calculations.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
from fractions import Fraction
import numpy as np
import scipy.sparse as sp
from . import _reference_m0 as v


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def store_anchor(path, anchor):
    L=v.canonical(anchor['factor']['L']); Path(path).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path, data=L.data, indices=L.indices, indptr=L.indptr,
                        shape=np.array(L.shape), d=anchor['factor']['d'], perm=anchor['factor']['perm'],
                        sigma=np.float64(anchor['sigma']), nu=np.int64(anchor['nu']))
    return file_sha256(path)


def _factor_from_file(path):
    with np.load(path,allow_pickle=False) as z:
        nrows,ncols=map(int,z['shape'])
        L=sp.csr_matrix((z['data'],z['indices'],z['indptr']),shape=(nrows,ncols))
        d=z['d'].copy(); perm=z['perm'].copy(); sigma=float(z['sigma']); nu=int(z['nu'])
    return v.canonical(L),d,perm,sigma,nu


def recheck_anchor(A, factor_file):
    """Recompute a verified eigenvalue lower bound from SAVED candidate factors.
    No factorization, eigenvalue estimates, or pre-existing rho are trusted.
    """
    A=v.canonical(A)
    if abs(A-A.T).nnz: raise v.CertFail('NOT_SYMMETRIC')
    L,d,perm,sigma,nu=_factor_from_file(factor_file)
    n=A.shape[0]
    if L.shape!=(n,n) or d.size!=n or perm.size!=n: raise v.CertFail('FACTOR_STRUCTURE')
    if not np.array_equal(np.sort(perm),np.arange(n)): raise v.CertFail('FACTOR_STRUCTURE','not a permutation')
    if sp.triu(L,1).nnz or not np.all(L.diagonal()==1): raise v.CertFail('FACTOR_STRUCTURE')
    if not (np.isfinite(d).all() and np.isfinite(L.data).all() and np.isfinite(sigma)):
        raise v.CertFail('NONFINITE')
    if np.any(d==0) or nu!=int((d<0).sum()): raise v.CertFail('ZERO_PIVOT')
    sh,sl=v.two_sum(A.diagonal(),np.full(n,-sigma,dtype=np.float64))
    Mt=A.tolil();Mt.setdiag(sh);Mt=v.canonical(Mt)
    P=sp.csr_matrix((np.ones(n),(perm,np.arange(n))),shape=(n,n))
    B=v.canonical(P@Mt@P.T)
    low=P@sl
    C=v.canonical(L@sp.diags(d)@L.T)
    absL=abs(L)
    absC=v.canonical(absL@sp.diags(np.abs(d))@absL.T)
    pat=(absL>0).astype(np.float64)
    patC=v.canonical(pat@pat.T);patC.data[:]=1.0
    c=float(np.max(np.diff(L.indptr))); g=v.gamma_hi(c+1.0)
    den=v.dn(1.0-g);tiny=v.up(2.0*c*v.ETA)
    T=v.canonical(absC+patC*tiny);T.data=v.up(T.data)
    T.data=v.up(T.data/den)
    EC=T.copy();EC.data=v.up(EC.data*g)
    EC=v.canonical(EC+patC*tiny);EC.data=v.up(EC.data)
    Rt=v.canonical(B-C)
    Rbar=abs(Rt);Rbar.data=v.up(Rbar.data/v.dn(1.0-v.U))
    Rbar=v.canonical(Rbar+EC);Rbar.data=v.up(Rbar.data)
    Rbar=v.canonical(Rbar+sp.diags(np.abs(low)));Rbar.data=v.up(Rbar.data)
    delta=v.norm2_sym_hi(Rbar)
    rho=float(v.dn(sigma-delta))
    if not np.isfinite(rho):raise v.CertFail('NONFINITE')
    return {'rho':rho,'nu':nu,'delta':delta,'sigma':sigma}


def exact_inertia(A_dense,rho):
    """Exact rational LDL when no zero pivot; return None on a zero pivot.
    For exact small SPD verification the strict-positive proof can use
    exact symmetric elimination without pivoting; ambiguous cases are skipped.
    """
    n=A_dense.shape[0]
    M=[[Fraction(float(A_dense[i,j]))-(Fraction(float(rho)) if i==j else 0) for j in range(n)] for i in range(n)]
    pos=neg=0
    for k in range(n):
        piv=M[k][k]
        if piv==0:return None
        neg+=piv<0;pos+=piv>0
        for i in range(k+1,n):
            if M[i][k]:
                f=M[i][k]/piv
                for j in range(k+1,n):M[i][j]-=f*M[k][j]
    return (neg,0,pos)


def exact_residual2(A,b,x):
    A=v.canonical(A)
    acc=Fraction(0)
    for i in range(A.shape[0]):
        z=Fraction(float(b[i]))
        for j in range(A.indptr[i],A.indptr[i+1]):
            z-=Fraction(float(A.data[j]))*Fraction(float(x[A.indices[j]]))
        acc+=z*z
    return acc


def _is_exact_canonical(A):
    return abs(v.canonical(A)-v.canonical(A).T).nnz==0


def verify_record(record, seq, base_dir, exact_n=12):
    """Validate chain and data hashes from saved data; no refactorization.

    A successful result depends on the published V1-V5 assumptions; this is
    not an external formal verification of library-level IEEE behavior.
    """
    errors=[]
    if record.get('status') not in {'CERTIFIED','UNVERIFIED','HEURISTIC'}:errors.append('INVALID_STATUS')
    try:
        A,b,meta=seq.get(record['t'])
        sp_rel=Path(record['solution_file'])
        sf=Path(base_dir)/sp_rel
        if sp_rel.is_absolute() or '..' in sp_rel.parts or not sf.is_file(): raise ValueError('Missing/invalid solution path')
        if file_sha256(sf)!=record['solution_file_sha256']:raise ValueError('Solution artifact hash mismatch')
        with np.load(sf,allow_pickle=False) as z:
            x=z['x']; w=z['w']
        if record['matrix_sha256']!=v.matrix_sha256(A):errors.append('MATRIX_HASH_MISMATCH')
        if record['rhs_sha256']!=v.vec_sha256(b):errors.append('RHS_HASH_MISMATCH')
        if record['x_sha256']!=v.vec_sha256(x):errors.append('SOLUTION_HASH_MISMATCH')
        if not _is_exact_canonical(A):errors.append('NOT_SYMMETRIC')
        if record['status']=='CERTIFIED':
            if not v.arith_check():errors.append('FTZ_DETECTED')
            chain=record.get('lambda_chain',[])
            if not chain: errors.append('MISSING_CHAIN')
            bounds=[]
            for link in chain:
                typ=link['type']
                if typ=='gershgorin':
                    if not (np.all(np.isfinite(w)) and np.all(w>0)):
                        errors.append('INVALID_W');continue
                    bounds.append(v.gershgorin_lower(A,w))
                    if link.get('w_sha256')!=v.vec_sha256(w):errors.append('W_HASH_MISMATCH')
                elif typ in {'anchor-inertia','anchor-weyl'}:
                    at=int(link['anchor_t'])
                    A0,b0,meta0=seq.get(at)
                    fp_rel=Path(link['factor_file'])
                    fp=Path(base_dir)/fp_rel
                    if fp_rel.is_absolute() or '..' in fp_rel.parts or not fp.is_file():errors.append('MISSING_FACTOR');continue
                    if file_sha256(fp)!=link['factor_sha256']:errors.append('FACTOR_HASH_MISMATCH');continue
                    a0=recheck_anchor(A0,fp)
                    if a0['nu']!=0 or a0['rho']<=0:errors.append('ANCHOR_NOT_SPD');continue
                    if typ=='anchor-inertia':
                        if at!=record['t']:errors.append('ANCHOR_TIMESTEP_MISMATCH')
                        bounds.append(a0['rho'])
                    else:
                        # recompute anchored transfer using same verified arithmetic core
                        bounds.append(v.weyl_transfer_lower(A0,A,a0['rho']))
                    if link['anchor_matrix_sha256']!=v.matrix_sha256(A0): errors.append('ANCHOR_HASH_MISMATCH')
                else: errors.append('UNKNOWN_CHAIN_TYPE')
            lo=float.fromhex(record['lambda_lo']);rhi=float.fromhex(record['R_hi'])
            eps=float.fromhex(record['eps']);target=float.fromhex(record['target'])
            if not (np.isfinite(lo) and lo>0):errors.append('NONPOSITIVE_LO')
            if not bounds or max(bounds)<lo:errors.append('UNSUPPORTED_LAMBDA_LO')
            rr=v.residual_bound(A,b,x)
            if rr>rhi:errors.append('RESIDUAL_RECOMPUTE_TOO_LARGE')
            if v.up(rhi/lo)!=eps:errors.append('EPS_MISMATCH')
            if record['target_met'] != bool(eps<=target):errors.append('TARGET_MET_MISMATCH')
            if record.get('reason') is not None:errors.append('CERTIFIED_HAS_REASON')
            if A.shape[0]<=exact_n:
                r2=exact_residual2(A,b,x)
                if Fraction(float(rhi))**2<r2:errors.append('EXACT_RESIDUAL_VIOLATION')
                inertia=exact_inertia(A.toarray(),lo)
                if inertia is None:errors.append('EXACT_INERTIA_INDETERMINATE')
                elif inertia[0]!=0:errors.append('EXACT_LAMBDA_VIOLATION')
        else:
            if record.get('target_met') is not None:errors.append('NONCERT_TARGET_MET')
            if record.get('reason') is None:errors.append('NONCERT_MISSING_REASON')
            if 'eps' in record:errors.append('NONCERT_HAS_EPS')
    except Exception as e:
        errors.append('VERIFIER_EXCEPTION:'+str(e))
    return not errors,errors
