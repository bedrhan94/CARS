"""Frozen, reproducible 3D benchmark families; all generated, none measured.
Uses the user-supplied Phase 3 fem3d_ref.py reference for normative assembly ordering.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from cars import _reference_m0 as v

ROOT=Path(__file__).resolve().parents[2]
PHASE3=ROOT/'fixtures'/'phase3'
REF=ROOT/'reference_impl'
if str(REF) not in sys.path:sys.path.insert(0,str(REF))
from fem3d_ref import elasticity_mesh,diffusion_mesh,assemble_float,lap7_3d  # noqa


def frozen_factor(name):
    blob=json.loads((PHASE3/(name+'.json')).read_text())
    Z=np.array([float.fromhex(x) for x in blob['Z']],np.float64).reshape(blob['shape'])
    K=np.array([float.fromhex(x) for x in blob['Kref']],np.float64).reshape(blob['Kref_shape'])
    sha=hashlib.sha256(Z.astype('<f8').tobytes()).hexdigest()
    if sha!=blob['sha256_of_Z_bytes']:raise v.CertFail('ELEMENT_DATA_HASH_MISMATCH',name)
    shah=hashlib.sha256(K.astype('<f8').tobytes()).hexdigest()
    if shah!=blob['sha256_of_Kref_bytes']:raise v.CertFail('ELEMENT_DATA_HASH_MISMATCH','Kref '+name)
    return Z,K,sha


class Family3D:
    def __init__(self, family_id, size_label='S0',seed=0,split='dev',T=None):
        config=json.loads((PHASE3/'pilot_configs.json').read_text())
        spec=config['families'][family_id]
        if split=='heldout':spec={**spec,**config['heldout_variants'].get(family_id,{})}
        sizes=config['sizes'][spec['kind']]; size=sizes[size_label]
        if split=='heldout' and 'sizes' in spec:size=spec['sizes'][size_label]
        self.config=config;self.id=family_id;self.spec=spec;self.size_label=size_label;self.N=size;self.seed=seed
        self.split=split;self.T=(int(T) if T is not None else config['common']['T'])
        self.label=spec['label'];self.kind=spec['kind'];self.change=spec['change']
        if self.kind=='elas':self.mesh=elasticity_mesh(size)
        elif self.kind=='diff':self.mesh=diffusion_mesh(size,spec['aspect'])
        else:self.mesh=None
        self.n=self.mesh.n if self.mesh else size**3
        self.Z=self.Kref=self.factor_hash=None
        if self.mesh:
            self.Z,self.Kref,self.factor_hash=frozen_factor(spec['Z'])
        self.rng=np.random.default_rng(seed)
        self.phi=self.rng.random(3)
        self.damage_order=self._damage_order() if self.mesh and self.change in ('local','cut plane (c=0)') else []
        self._base=self._base_coeff() if self.mesh else None
        self._safe_lambda=None
        if self.change=='pd-loss shift':
            from cars_m2.anchor import lambda_estimate
            import scipy.sparse as sp
            A0=assemble_float(self.mesh,self.Z,self._base/self.N,self.Kref)
            self._safe_lambda=lambda_estimate(v.canonical(A0))

    def _base_coeff(self):
        p=self.mesh.centers; phi=self.phi
        base=np.exp(0.3*np.sin(2*np.pi*(p[:,0]+phi[0]))*np.cos(np.pi*(p[:,1]+phi[1])))
        return base

    def _damage_order(self):
        if self.change=='cut plane (c=0)':
            return [e for e,c in enumerate(self.mesh.centers) if int(np.floor(c[0]*self.N))==self.N//2]
        N=self.N
        i,j,k=N//2,0,N//2
        used=set();path=[]
        for _ in range(self.T*2+1):
            idx=i+N*(j+N*k)
            if idx not in used: used.add(idx);path.append(idx)
            moves=[]
            if j+1<N and (i+N*((j+1)+N*k)) not in used:moves.append((i,j+1,k))
            for ii,jj,kk in ((i+1,j,k),(i-1,j,k),(i,j,k+1),(i,j,k-1)):
                if 0<=ii<N and 0<=kk<N and (ii+N*(jj+N*kk)) not in used:moves.append((ii,jj,kk))
            if not moves:break
            first=(i,j+1,k)
            if first in moves and self.rng.random()<0.6:i,j,k=first
            else:
                alt=[mv for mv in moves if mv!=first]
                sel=alt if alt else moves
                i,j,k=sel[int(self.rng.integers(0,len(sel)))]
        return path

    def coeff(self,t):
        if self.mesh is None:return None
        p=self.mesh.centers; base=self._base.copy()
        if self.change=='gradual':
            xc=np.array([0.15+0.7*t/self.T,0.5+0.2*np.sin(2*np.pi*t/self.T+2*np.pi*self.phi[2]),0.5])
            base*=1-float(self.spec['a'])*np.exp(-np.sum((p-xc)**2,axis=1)/(2*float(self.spec['w'])**2))
        elif self.change=='local':
            for e in self.damage_order[:t*int(self.spec['per_step'])] if 'per_step' in self.spec else self.damage_order[:t]:
                base[e]*=1-float(self.spec['damage'])
        elif self.change=='cut plane (c=0)':
            for e in self.damage_order[:int(t)]:base[e]=0.0
        return (base/self.N if self.kind=='diff' else base/self.N).astype(np.float64)

    def get(self,t):
        if not 0<=t<=self.T:raise IndexError(t)
        if self.mesh:
            c=self.coeff(t)
            A=v.canonical(assemble_float(self.mesh,self.Z,c,self.Kref))
            if self.change=='pd-loss shift':
                # PROTOCOL CORRECTION: plus sign yields loss of positive definiteness at T/6.
                # The handoff writes '-' but describes crossing to indefinite.
                shift=self._safe_lambda*(0.5-3.0*t/self.T)
                A=v.canonical(A+sp.diags(np.full(A.shape[0],shift)))
            if self.kind=='elas':
                b=np.zeros(A.shape[0]);
                # Gravity on free z-dofs (clamped x=0); mesh numbering is node-major.
                b[2::3]=-(1.0/self.N)**3
            else:b=np.full(A.shape[0],(1.0/self.N)**3)
        else:
            A=v.canonical(lap7_3d(self.N));b=np.ones(self.n)
            c=None
        meta={'family':self.id,'seed':self.seed,'split':self.split,'size':self.size_label,
              'n':A.shape[0],'nnz':A.nnz,'t':t,'label':self.label,'matrix_sha256':v.matrix_sha256(A),
              'positive_offdiag':int(np.sum(sp.triu(A,1).data>0)),
              'coeff_sha256':(v.vec_sha256(c) if c is not None else None)}
        return A,b,{**meta,'c':c}

    def element_model(self):return self.mesh
