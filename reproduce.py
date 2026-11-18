"""Reproduction check of the CARS research artifact (run from the repository root with the locked environment).

Steps
  0  integrity: every file listed in MANIFEST_SHA256.txt is present with the recorded SHA-256 (checked first, before
     any script rewrites its own result file);
  1  decision rule of the pre-specified analysis on synthetic cases;
  2  exact rational test suite of the accurate residual enclosure;
  3  every frozen held-out matrix, right-hand side and coefficient vector rebuilt from the exact hex inputs;
  4  replay of every shipped primary-arm record (CHOLMOD re-proof, bound recomputation);
  5  the frozen confirmatory analysis re-run on all records and compared with analysis10b.json;
  6  independent recomputation of the primary statistic;
  7  analyses of every supplementary and post hoc experiment re-run and compared with the shipped results
     (SIMP; structural envelope; compiled kernels; corrected fallback and its checks; audit extensions;
     full compilation; SIMP with compiled operations; large size).
Not re-run (machine time, solution vectors or Numba needed): the timed cells and their full replays, and the
equivalence tests of the compiled kernels; their records and results are included.
Writes reproduce_result.json; prints ALL_OK."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
FD = ROOT / 'FINAL_DEV'


def run(args, cwd=ROOT, timeout=7200):
    t0 = time.time()
    r = subprocess.run([PY, '-B'] + [str(a) for a in args], cwd=str(cwd), env=ENV, capture_output=True, text=True, timeout=timeout)
    return dict(cmd=' '.join(str(a) for a in args), rc=r.returncode, seconds=round(time.time() - t0, 1),
                tail=(r.stdout + r.stderr)[-500:])


def strip(d, drop=('seconds',)):
    if isinstance(d, dict):
        return {k: strip(v, drop) for k, v in d.items() if k not in drop}
    if isinstance(d, list):
        return [strip(v, drop) for v in d]
    return d


def same_numbers(a, b, path=''):
    """Compare two JSON trees, ignoring path-like strings and timing/environment keys."""
    skip = ('env', 'seconds', 'generated', 'timestamp', 'path', 'file', 'dir', 'host', 'platform')
    if isinstance(a, dict) and isinstance(b, dict):
        diffs = []
        for k in set(a) | set(b):
            if any(s in k.lower() for s in skip):
                continue
            if k not in a or k not in b:
                diffs.append(f'{path}/{k}: missing'); continue
            diffs += same_numbers(a[k], b[k], f'{path}/{k}')
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} != {len(b)}']
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same_numbers(x, y, f'{path}[{i}]')]
    if isinstance(a, float) and isinstance(b, float):
        return [] if a == b or abs(a - b) <= 1e-12 * max(1.0, abs(a)) else [f'{path}: {a} != {b}']
    if isinstance(a, str) and isinstance(b, str) and (('\\' in a or '/' in a) and ('\\' in b or '/' in b)):
        return []
    return [] if a == b else [f'{path}: {str(a)[:60]} != {str(b)[:60]}']


def rerun_compare(script, cwd, result):
    """Run a script (path relative to cwd) and compare its result file (path relative to cwd), ignoring run times."""
    before = strip(json.loads((cwd / result).read_text()))
    s = run([script], cwd=cwd)
    after = json.loads((cwd / result).read_text()) if (cwd / result).exists() else None
    s['identical'] = after is not None and strip(after) == before
    return s


def manifest():
    bad = []
    for line in (ROOT / 'MANIFEST_SHA256.txt').read_text().splitlines():
        h, rel = line.split('  ', 1)
        p = ROOT / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            bad.append(rel)
    return dict(rc=0 if not bad else 1, changed_or_missing=bad[:30], n=len(bad))


def main():
    S = {}
    S['0_manifest'] = manifest()
    S['1_decision_tests'] = run(['FINAL_DEV/phase10b/test_decision10b.py'])
    S['2_eft_exact_tests'] = rerun_compare('PHASE10A_DEV/eft/eft_exact_tests.py', ROOT, 'PHASE10A_DEV/eft/eft_exact_tests.json')
    S['3_check_inputs'] = run(['FINAL_DEV/release_tools/check_inputs.py'])
    ci = FD / 'release_tools/check_inputs_result.json'
    if ci.exists():
        c = json.loads(ci.read_text())
        S['3_check_inputs'].update(checked=c['checked'], keys_with_mismatch=c['keys_with_mismatch'])
        if c['keys_with_mismatch']:
            S['3_check_inputs']['rc'] = 1
    S['4_replay_subset'] = run(['FINAL_DEV/release_tools/replay_subset.py'])
    rs = FD / 'release_tools/replay_subset_result.json'
    if rs.exists():
        S['4_replay_subset']['result'] = json.loads(rs.read_text()).get('totals')
    out10b = ROOT / '_reproduce_analysis10b.json'
    S['5_analysis10b'] = run(['FINAL_DEV/phase10b/analyze10b.py', 'FINAL_DEV/phase10b/runs', out10b])
    if out10b.exists():
        a = json.loads(out10b.read_text()); b = json.loads((FD / 'phase10b/analysis10b.json').read_text())
        keys = ('primary', 'joint_outcome', 'secondary_st_s2', 'secondary_mt_s3', 'secondary_mt_s2', 'session2')
        d = [x for k in keys for x in same_numbers(a.get(k), b.get(k), k)]
        S['5_analysis10b'].update(compared=list(keys), n_differences=len(d), differences=d[:20])
        if d:
            S['5_analysis10b']['rc'] = 1
        out10b.unlink()
    S['6_recompute_primary'] = rerun_compare('FINAL_DEV/audit/recompute_primary.py', ROOT, 'FINAL_DEV/audit/recompute_primary.json')
    simp = FD / 'simp/analysis_simp.json'
    before = json.loads(simp.read_text())
    S['7_simp'] = run(['FINAL_DEV/simp/analyze_simp.py'])
    d = same_numbers(json.loads(simp.read_text()), before)
    S['7_simp'].update(identical=not d, n_differences=len(d))
    st = FD / 'REVISION1/structural'
    S['7_struct_analysis'] = rerun_compare('analyze_struct.py', st, 'analysis_struct.json')
    S['7_envelope_check'] = rerun_compare('check_envelope.py', st, 'check_envelope.json')
    S['7_envelope_check']['ALL_EQUIVALENT'] = json.loads((st / 'check_envelope.json').read_text()).get('ALL_EQUIVALENT')
    r2 = FD / 'REVISION2'
    S['7_compiled_analysis'] = rerun_compare('analyze_compiled.py', r2 / 'compiled', 'analysis_compiled.json')
    S['7_fallback_analysis'] = rerun_compare('analyze_simp_fb.py', r2 / 'fallback', 'analysis_simp_fb.json')
    S['7_fallback_checks'] = rerun_compare('check_fallback.py', r2 / 'fallback', 'check_fallback.json')
    S['7_eft_exact_abs'] = rerun_compare('eft_exact_abs.py', r2 / 'checks', 'eft_exact_abs.json')
    S['7_rump_share'] = rerun_compare('rump_constant_share.py', r2 / 'checks', 'rump_constant_share.json')
    S['7_negctl_coeff'] = rerun_compare('negctl_coeff.py', r2 / 'checks', 'negctl_coeff.json')
    r3 = FD / 'REVISION3'
    S['7_fullc_analysis'] = rerun_compare('analyze_fullc.py', r3 / 'fullc', 'analysis_fullc.json')
    S['7_simpc_analysis'] = rerun_compare('analyze_simpc.py', r3 / 'simpc', 'analysis_simpc.json')
    S['7_large_analysis'] = rerun_compare('analyze_large.py', r3 / 'large', 'analysis_large.json')
    ok = (all(v.get('rc', 0) == 0 for v in S.values()) and all(v.get('identical', True) for v in S.values())
          and S['7_envelope_check']['ALL_EQUIVALENT'] is True)
    res = dict(python=sys.version.split()[0], ALL_OK=bool(ok), steps=S)
    (ROOT / 'reproduce_result.json').write_text(json.dumps(res, indent=1))
    for k, v in S.items():
        print(k, 'rc', v.get('rc'), {kk: vv for kk, vv in v.items()
                                     if kk in ('identical', 'checked', 'keys_with_mismatch', 'result', 'n_differences', 'n', 'ALL_EQUIVALENT')})
    print('ALL_OK', ok)


if __name__ == '__main__':
    main()
