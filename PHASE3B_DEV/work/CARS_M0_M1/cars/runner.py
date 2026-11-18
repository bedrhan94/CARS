"""M1 baselines for CARS. No new algorithm and no speedup claims.

The M0 numeric primitives are included verbatim, with attribution, from the
user-supplied Phase 2A reference implementation. This module implements M1.
"""
from __future__ import annotations
import io,json,os,platform,contextlib,time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from .sequences import MatrixSequence
from . import _reference_m0 as v
from .verify import store_anchor,file_sha256,verify_record

CERTIFIED_METHODS={'C-per','C-cw','C-anchor-weyl'}
METHODS=['U-cg','U-pcg-j','C-per','C-cw','C-anchor-weyl']


def _estimate_lambda(A):
    try:
        if A.shape[0] <= 3:return float(np.linalg.eigvalsh(A.toarray())[0])
        return float(sla.eigsh(A,k=1,which='SA',v0=np.linspace(1.,1.1,A.shape[0]),tol=1e-7,maxiter=5000,return_eigenvectors=False)[0])
    except (sla.ArpackNoConvergence,RuntimeError,ValueError):
        return float('nan')


def _get_anchor(A):
    # Only eigsh determines shift. Its result NEVER becomes part of certificate.
    guess=_estimate_lambda(A)
    if not np.isfinite(guess) or guess<=0:raise v.CertFail('LAMBDA_NONPOSITIVE','eigenvalue estimate is not positive')
    sig=0.9*guess
    reasons=[]
    for _ in range(4):
        try:
            cert=v.anchor_inertia(A,sig,0)
            if cert['nu']==0 and cert['rho']>0:return cert
            reasons.append('RHO_NONPOSITIVE')
        except v.CertFail as e:reasons.append(e.reason)
        sig/=5
    raise v.CertFail('ANCHOR_FACTORIZATION_FAILED',','.join(reasons))


def _positive_weight(A,prev=None):
    """A1 bootstrap: attempt previous strictly positive vector, else current approximate solve.
    w is a *trial* weight, never a certificate; observed cost is included in decision time.
    """
    candidates=[]
    if prev is not None and len(prev)==A.shape[0] and np.all(np.isfinite(prev)) and np.all(prev>0):
        candidates.append(('previous-solution',prev))
    candidates.append(('all-ones',np.ones(A.shape[0])))
    for name,w in candidates:
        try:
            lo=v.gershgorin_lower(A,w)
            if lo>0:return lo,w,name
        except Exception:pass
    # Solve ONLY to choose w, not to assert the numerical bound.
    try:
        w=sla.spsolve(A,np.ones(A.shape[0]))
        if np.isfinite(w).all() and np.all(w>0):
            lo=v.gershgorin_lower(A,w)
            if lo>0:return lo,np.asarray(w,dtype=np.float64),'positive-bootstrap-spsolve'
    except Exception:pass
    return v.gershgorin_lower(A,np.ones(A.shape[0])),np.ones(A.shape[0]),'all-ones-nonpositive'


def _solve(A,b,method,lam=1.,target=1e-8,xprev=None,anchor=None):
    n=len(b);guess=np.zeros(n) if xprev is None else xprev.copy()
    if method=='direct':
        lu=sla.splu(A.tocsc(),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.0,
                    options={'SymmetricMode':True})
        x=lu.solve(b)
        return np.asarray(x), {'name':'second-lu','iters':0,'converged':True}
    inv=None;effective=method
    if method=='pcg-j':
        d=A.diagonal()
        if np.all(np.isfinite(d)) and np.all(d>0):
            inv=sla.LinearOperator((n,n),matvec=lambda x:x/d,dtype=np.float64)
        else:effective='cg'
    elif method=='pcg-anchor':
        if anchor is not None:
            try:inv=v.ldlt_preconditioner(anchor['factor'])
            except v.CertFail:effective='pcg-j'
        else:effective='pcg-j'
        if effective=='pcg-j':
            d=A.diagonal()
            if np.all(d>0) and np.isfinite(d).all():
                inv=sla.LinearOperator((n,n),matvec=lambda x:x/d,dtype=np.float64)
            else:effective='cg'
    iters=[0]
    def cb(_):iters[0]+=1
    tol=0.5*target*lam
    rtol=float(min(1e-12,max(1e-16,tol/max(np.linalg.norm(b),1e-300))))
    x,info=sla.cg(A,b,x0=guess,rtol=rtol,atol=0.0,maxiter=10*n,M=inv,callback=cb)
    x=np.asarray(x,dtype=np.float64)
    true_r=np.linalg.norm(b-A@x)
    # If SciPy rtol stops early due scaling, flag nonconvergence in solver metadata.
    converged=(info==0 and true_r<=tol)
    return x,{'name':effective,'iters':iters[0],'converged':bool(converged),
              'true_residual_heuristic':float(true_r).hex(), 'scipy_info':int(info)}


def _env_snapshot():
    s=io.StringIO()
    with contextlib.redirect_stdout(s):np.show_config()
    return {'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,
            'scipy':scipy.__version__,'arith_check':bool(v.arith_check()),
            'thread_env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            'numpy_config':s.getvalue()}


def _safe_token(s):return ''.join(c if (c.isalnum() or c in '_-') else '_' for c in s)


def _uncert_record(A,b,x,target,reason,heuristic=False):
    return {'schema':'cars-cert/0.2','matrix_sha256':v.matrix_sha256(A),
            'rhs_sha256':v.vec_sha256(b),'x_sha256':v.vec_sha256(x),
            'target':float(target).hex(),'status':'HEURISTIC' if heuristic else 'UNVERIFIED',
            'reason':reason,'target_met':None}


def run_sequence(seq:MatrixSequence,method:str,output_dir,target=1e-8,verify=True):
    if method not in METHODS:raise ValueError(method)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    env_file=out/'environment.json'
    if not env_file.exists():
        env_file.write_text(json.dumps(_env_snapshot(),indent=2),encoding='utf-8')
    art=out/'artifacts';art.mkdir(exist_ok=True)
    results=[];previous=None;anchor=None;anchor_A=None;anchor_t=None;anchor_file=None;anchor_hash=None
    run_id=f'{seq.family}_{seq.size}_{seq.T}_{seq.seed}_{method}'
    for t in range(seq.T+1):
        A,b,meta=seq.get(t)
        if not v.arith_check():raise RuntimeError('FTZ_DETECTED; stopping run')
        chosen_w=np.ones(A.shape[0]);lam=None;lo_valid=False;reason=None;chain=[]
        elapsed={'decision':0.0,'anchor':0.0,'solve':0.0,'certify':0.0}
        try:
            start=time.perf_counter()
            if method=='C-per':
                start_a=time.perf_counter(); ac=_get_anchor(A);elapsed['anchor']=time.perf_counter()-start_a
                lam=ac['rho'];lo_valid=True
                path=art/f'{_safe_token(run_id)}_anchor_t{t}.npz'
                sig=store_anchor(path,ac)
                chain=[{'type':'anchor-inertia','anchor_t':t,'anchor_matrix_sha256':meta['matrix_sha256'],
                        'factor_file':str(path.relative_to(out)),'factor_sha256':sig,'sigma':float(ac['sigma']).hex()}]
            elif method=='C-cw':
                lam,chosen_w,wkind=_positive_weight(A,previous)
                lo_valid=lam>0
                chain=[{'type':'gershgorin','w_sha256':v.vec_sha256(chosen_w),'w_origin':wkind}]
                if not lo_valid:raise v.CertFail('LAMBDA_NONPOSITIVE')
            elif method=='C-anchor-weyl':
                if t==0:
                    start_a=time.perf_counter(); anchor=_get_anchor(A);elapsed['anchor']+=time.perf_counter()-start_a
                    anchor_A=A;anchor_t=t
                    path=art/f'{_safe_token(run_id)}_anchor_t{t}.npz'
                    anchor_hash=store_anchor(path,anchor);anchor_file=str(path.relative_to(out))
                gw,chosen_w,wkind=_positive_weight(A,previous)
                candidates=[(gw,{'type':'gershgorin','w_sha256':v.vec_sha256(chosen_w),'w_origin':wkind})]
                if anchor is not None:
                    if t==anchor_t:wt=anchor['rho']
                    else:wt=v.weyl_transfer_lower(anchor_A,A,anchor['rho'])
                    candidates.append((wt,{'type':'anchor-inertia' if t==anchor_t else 'anchor-weyl',
                                           'anchor_t':anchor_t,'anchor_matrix_sha256':v.matrix_sha256(anchor_A),
                                           'factor_file':anchor_file,'factor_sha256':anchor_hash}))
                lam,link=max(candidates,key=lambda x:x[0]);chain=[link]
                if lam<=0:
                    start_a=time.perf_counter();anchor=_get_anchor(A);elapsed['anchor']+=time.perf_counter()-start_a
                    anchor_A=A;anchor_t=t
                    path=art/f'{_safe_token(run_id)}_anchor_t{t}.npz'
                    anchor_hash=store_anchor(path,anchor);anchor_file=str(path.relative_to(out))
                    lam=anchor['rho']
                    chain=[{'type':'anchor-inertia','anchor_t':t,'anchor_matrix_sha256':v.matrix_sha256(A),
                            'factor_file':anchor_file,'factor_sha256':anchor_hash}]
                lo_valid=True
            else:
                lam=_estimate_lambda(A)
                if not (np.isfinite(lam) and lam>0):raise v.CertFail('LAMBDA_NONPOSITIVE')
            elapsed['decision']=time.perf_counter()-start-elapsed['anchor']
            if not lo_valid and method in CERTIFIED_METHODS:
                raise v.CertFail('LAMBDA_NOT_VALIDATED')
            start_s=time.perf_counter()
            solver=('direct' if method=='C-per' else ('pcg-anchor' if method=='C-anchor-weyl' else
                       'pcg-j' if method in ('U-pcg-j','C-cw') else 'cg'))
            x,sinfo=_solve(A,b,solver,lam,target,previous,anchor if method=='C-anchor-weyl' else None)
            elapsed['solve']=time.perf_counter()-start_s
            start_c=time.perf_counter()
            if lo_valid:
                rec=v.certificate(A,b,x,lam,True,target)
                rec['lambda_chain']=chain
            else:
                rec=_uncert_record(A,b,x,target,'LAMBDA_ESTIMATED_ONLY',True)
            elapsed['certify']=time.perf_counter()-start_c
            previous=x.copy() if np.all(np.isfinite(x)) else None
            if rec['status']=='CERTIFIED' and not sinfo['converged']:
                sinfo['reason']='SOLVER_NOT_CONVERGED (candidate nevertheless certified)'
        except (v.CertFail,ValueError,RuntimeError,FloatingPointError,ZeroDivisionError) as e:
            code=e.reason if isinstance(e,v.CertFail) else 'INTERNAL_FAILURE'
            rec=_uncert_record(A,b,np.zeros_like(b),target,code)
            x=np.zeros_like(b);sinfo={'name':'none','iters':0,'converged':False,'reason':str(e)[:200]}
            chain=[];previous=None
        # save per-step vector artifacts; verification uses regenerated A,b and stored x,w
        sol=art/f'{_safe_token(run_id)}_solution_t{t}.npz'
        np.savez_compressed(sol,x=np.asarray(x,dtype=np.float64),w=np.asarray(chosen_w,dtype=np.float64))
        rec.update({'run_id':run_id,'family':seq.family,'size':seq.size,'seed':seq.seed,'T':seq.T,
                    't':t,'n':A.shape[0],'nnz':A.nnz,'method':method,
                    'solver':sinfo,'timing_s':elapsed,'positive_upper_offdiagonals':meta['positive_upper_offdiagonals'],
                    'solution_file':str(sol.relative_to(out)),'solution_file_sha256':file_sha256(sol)})
        results.append(rec)
    path=out/f'{_safe_token(run_id)}.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for r in results:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    if verify:
        errors=[]
        for r in results:
            ok,err=verify_record(r,seq,out,exact_n=40)
            if not ok:errors.append((r['t'],err))
        if errors:raise AssertionError(f'Failed post-run validation: {errors[:4]}')
    return path,results


def summarize(records):
    from collections import Counter
    counts=Counter((r['status'],r.get('target_met')) for r in records)
    reasons=Counter(r.get('reason') for r in records if r.get('reason'))
    certified=[r for r in records if r['status']=='CERTIFIED']
    return {'records':len(records), 'status_counts':{str(k):v for k,v in counts.items()},
            'reasons':dict(reasons),'median_eps':float(np.median([float.fromhex(r['eps']) for r in certified])) if certified else None,
            'median_lambda_lo':float(np.median([float.fromhex(r['lambda_lo']) for r in certified])) if certified else None}
