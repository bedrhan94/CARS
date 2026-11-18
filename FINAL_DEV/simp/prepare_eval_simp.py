"""SIMP3D evaluation inputs: generate twice (separate processes), compare bit for bit, then fix the per-(instance,
window) targets by the rule in simp_settings.json and record the SHA-256 of the settings and the frozen manifest.
Usage: python prepare_eval_simp.py"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
PY = str(CARS / 'PHASE5_DEV' / '.venv' / 'Scripts' / 'python.exe')
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ev = HERE / 'eval'; ev.mkdir(exist_ok=True)
    for name in ('pass_a', 'pass_b'):
        with open(ev / f'gen_{name}.log', 'w') as f:
            assert subprocess.run([PY, '-B', 'gen_simp.py', str(ev / name), 'eval'], stdout=f, stderr=subprocess.STDOUT,
                                  env=ENV, cwd=str(HERE)).returncode == 0
    fa = {p.relative_to(ev / 'pass_a').as_posix(): sha(p) for p in sorted((ev / 'pass_a').rglob('*')) if p.is_file() and p.name != 'generation.json'}
    fb = {p.relative_to(ev / 'pass_b').as_posix(): sha(p) for p in sorted((ev / 'pass_b').rglob('*')) if p.is_file() and p.name != 'generation.json'}
    diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    if diff:
        raise SystemExit(f'SIMP generation not reproducible: {diff[:5]}')
    s = json.loads((HERE / 'simp_settings.json').read_text())
    man = json.loads((ev / 'pass_a' / 'simp_manifest.json').read_text())
    for key, entry in man['instances'].items():
        for w, win in entry['windows'].items():
            if s['target_rule'].startswith('absolute'):
                tau = 1e-8
            else:
                tau = 1e-8 * entry['history'][win['k0'] - 1]['u_norm']
            s['targets'][f'{key}|{w}'] = float(tau).hex()
    s['eval_frozen_files'] = fa; s['frozen_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (HERE / 'simp_settings.json').write_text(json.dumps(s, indent=1))
    (HERE / 'simp_settings.sha256').write_text(sha(HERE / 'simp_settings.json') + '  simp_settings.json\n')
    print('IDENTICAL', len(fa), 'files; targets', {k: float.fromhex(v) for k, v in s['targets'].items()})


if __name__ == '__main__':
    main()
