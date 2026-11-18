"""Frozen-input loader: a Family3D-compatible family whose coefficient vectors and right-hand side are read from the
frozen exact-hex files of the Phase 10B input freeze (FINAL_DEV/phase10b/inputs/pass_a) instead of being recomputed.
Coefficients computed with np.exp/np.sin can differ in the last bit between platforms (earlier finding PA-12); the frozen
hex values remove that dependence, so only the float assembly (elementwise products and a fixed-order sparse sum) remains
platform-dependent, and it is checked against the frozen matrix SHA-256 at every step.
Usage: FrozenFamily(inputs_dir, family, size, seed) -> .get(t) returns (A, b, meta) exactly as Family3D.get."""
import json
from pathlib import Path
import numpy as np
from cars_m2.families import Family3D
from cars import _reference_m0 as v


class FrozenFamily(Family3D):
    def __init__(self, inputs_dir, family_id, size_label, seed):
        inputs_dir = Path(inputs_dir)
        man = json.loads((inputs_dir.parent / 'inputs_manifest10b.json').read_text())['manifest'][f'{family_id}|{size_label}|{seed}']
        super().__init__(family_id, size_label, seed=seed, split=man['split'])
        self._rows = man['rows']
        self._c = [np.array([float.fromhex(h) for h in row], dtype=np.float64) for row in
                   json.loads((inputs_dir / f"coeff_hex_{family_id}_{size_label}_{man['split']}_{seed}.json").read_text())]
        self._b = np.array([float.fromhex(h) for h in json.loads((inputs_dir / 'rhs_hex.json').read_text())[f'{family_id}|{size_label}']],
                           dtype=np.float64)
        self.mismatches = []

    def coeff(self, t):
        return self._c[t].copy()

    def get(self, t):
        A, b, meta = super().get(t)
        b = self._b.copy()
        row = self._rows[t]
        got = dict(matrix_sha256=v.matrix_sha256(v.canonical(A)), rhs_sha256=v.vec_sha256(b), coeff_sha256=meta['coeff_sha256'])
        bad = [k for k in got if got[k] != row[k]]
        if bad:
            self.mismatches.append(dict(t=t, fields=bad))
        meta = dict(meta, rhs_sha256=got['rhs_sha256'], frozen_mismatch=bad)
        return A, b, meta
