"""Revision 2, post hoc negative control (referee comment: the earlier control A0 + c_t I is rejected by the binding because
it is not an element model, so it does not test the reuse side).

Positive definiteness is lost through the ELEMENT COEFFICIENTS, so every matrix is a genuine floating-point element-model
assembly with admissible coefficients (c >= 0): the binding succeeds and the transfer data are formally valid.
  NC-E3  elasticity (E3-TS development geometry, S1, seed 0): the coefficients of the element layer that crosses the
         mid-plane x = 1/2 are scaled by s_t = max(0, 1 - t/8). From t = 8 on the layer is void, the part beyond it is
         disconnected from the clamp, and the stiffness matrix is singular (rigid-body modes).
  NC-D3  diffusion (D3-TS development geometry, S1, seed 0): the coefficients of the elements around one interior node
         are scaled in the same way. From t = 8 on the row of that node is exactly zero (not stored), and the matrix is
         singular.
All seven frozen arms run on each sequence (target 1e-8). Ground truth: lambda_min of every stored matrix by a dense
symmetric eigensolver (validation only). Checks: (1) no CERTIFIED record on a matrix with lambda_min <= tol, where
tol = 64 u ||A||_2; (2) every certified lambda_lo <= lambda_min + tol; (3) every record replayed with the frozen
Phase 10A verifier (verify10a) is VERIFIED. Development data only. Writes negctl_coeff.json; exit 1 on a violation."""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'; os.environ['CARS_P6_THREADS'] = '1'
sys.dont_write_bytecode = True
import json, shutil, time
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b'))
import common10b as K                                                # noqa: E402
import run_cell10b as C10                                            # noqa: E402
sys.path.insert(0, str(CARS / 'PHASE6_DEV' / 'lean6'))
import verify10a as VV                                               # noqa: E402
from replay6 import MemoFamily                                       # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402

U = 2.0 ** -53


class Fading(Family3D):
    """Development family whose coefficients of a fixed element set fade to exactly zero at t = 8."""

    def __init__(self, base_id, label):
        super().__init__(base_id, 'S1', seed=0, split='dev')
        self.id = label
        m = self.mesh
        if base_id == 'E3-TS':
            self.fade = np.flatnonzero(np.floor(m.centers[:, 0] * self.N).astype(int) == self.N // 2)
        else:
            cen = m.centers.mean(axis=0)
            e0 = int(np.argmin(np.sum((m.centers - cen) ** 2, axis=1)))
            d0 = int(max(d for d in m.dofs[e0] if d >= 0))
            self.node = d0
            self.fade = np.flatnonzero((m.dofs == d0).any(axis=1))

    def coeff(self, t):
        c = super().coeff(t)
        c[self.fade] *= max(0.0, 1.0 - t / 8.0)
        return c


def main():
    t0 = time.time(); work = HERE / 'negctl_runs'
    if work.exists():
        shutil.rmtree(work)
    ch = C10.code_hash(); out = dict(sequences={}, violations=[])
    for base, label in (('E3-TS', 'NC-E3'), ('D3-TS', 'NC-D3')):
        fam = Fading(base, label); mf = MemoFamily(fam)
        truth = {}
        for t in range(fam.T + 1):
            A, _, _ = fam.get(t)
            w = np.linalg.eigvalsh(A.toarray())
            truth[t] = dict(lmin=float(w[0]), norm=float(max(abs(w[0]), abs(w[-1]))))
        seq = dict(n=fam.n, faded_elements=int(fam.fade.size), lambda_min={str(t): v['lmin'] for t, v in truth.items()}, arms={})
        for label_arm, cfg in K.ARMS:
            path, rows = C10.run_arm(label_arm, cfg, fam, work / label, 0, 0, ch)
            st = dict(certified=[], unverified={}, transfers=[], replay=Counter(), lam_over_true_max=None)
            ratios = []
            for r in rows:
                t = int(r['t']); tv = truth[t]; tol = 64 * U * tv['norm']
                vst, errs = VV.verify_record10a(r, mf, Path(path).parent, exact_n=0)
                st['replay'][vst] += 1
                if r['status'] == 'CERTIFIED':
                    lam = float.fromhex(r['lambda_lo'])
                    st['certified'].append(t)
                    ratios.append(lam / tv['lmin'] if tv['lmin'] > 0 else float('inf'))
                    if r['details'].get('bound_type') == 'm2a':
                        st['transfers'].append(t)
                    if tv['lmin'] <= tol:
                        out['violations'].append(dict(seq=label, arm=label_arm, t=t, kind='certified_singular'))
                    if lam > tv['lmin'] + tol:
                        out['violations'].append(dict(seq=label, arm=label_arm, t=t, kind='bound_above_lambda_min'))
                else:
                    st['unverified'][str(t)] = r.get('reason')
                if vst != 'VERIFIED':
                    out['violations'].append(dict(seq=label, arm=label_arm, t=t, kind='replay_' + vst, errors=errs))
            st['lam_over_true_max'] = float(max(ratios)) if ratios else None
            st['replay'] = dict(st['replay'])
            st['unverified_reasons'] = dict(Counter(st['unverified'].values()))
            seq['arms'][label_arm] = st
            print(label, label_arm, 'certified', len(st['certified']), 'transfers', st['transfers'], 'unverified',
                  st['unverified_reasons'], 'max lam/true', st['lam_over_true_max'], st['replay'], flush=True)
        seq['singular_steps'] = [t for t, v in truth.items() if v['lmin'] <= 64 * U * v['norm']]
        out['sequences'][label] = seq
    out['ALL_OK'] = not out['violations']; out['seconds'] = time.time() - t0
    (HERE / 'negctl_coeff.json').write_text(json.dumps(out, indent=1))
    print('ALL_OK', out['ALL_OK'], out['violations'][:5])
    sys.exit(0 if out['ALL_OK'] else 1)


if __name__ == '__main__':
    main()
