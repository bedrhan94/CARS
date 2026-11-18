"""Phase 10B code and environment freeze (run once, BEFORE the pre-registration freeze record and before any held-out
input exists). Writes reports/PHASE10B/PHASE10B_CODE_FREEZE.json and PHASE10B_ENVIRONMENT.json.

Code set = (i) every Python module that the Phase 10B workers actually import from the CARS tree, discovered by importing
them here and reading sys.modules; (ii) the Phase 10B scripts; (iii) the Phase 10A frozen manifest's files; (iv) the
data files the generator reads (family configuration and element-factor fixtures). Aggregate = SHA-256 of the sorted
JSON of {path: sha256}."""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import ctypes, hashlib, importlib.metadata as md, json, platform, subprocess, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common10b as K                                                # noqa: E402
for p in (K.CARS / 'PHASE10A_DEV' / 'methods', K.CARS / 'PHASE6_DEV' / 'lean6', K.CARS / 'PHASE5_DEV' / 'arith'):
    sys.path.insert(0, str(p))
import runner10a, verify10a, replay6, independent5                    # noqa: E402,F401
import psutil                                                        # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402

OWN = ['common10b.py', 'gen_inputs10b.py', 'compare_inputs10b.py', 'run_cell10b.py', 'replay10b.py', 'independent10b.py',
       'run_plan10b.py', 'analyze10b.py', 'test_decision10b.py', 'freeze_code10b.py', 'freeze_prereg10b.py']


def rel(p):
    return Path(p).resolve().relative_to(K.CARS).as_posix()


def main():
    files = {}
    for name, m in sorted(sys.modules.items()):
        f = getattr(m, '__file__', None)
        if f and Path(f).resolve().is_relative_to(K.CARS) and '.venv' not in Path(f).parts:
            files[rel(f)] = K.sha256_file(f)
    imported = sorted(files)
    for n in OWN:
        files[rel(HERE / n)] = K.sha256_file(HERE / n)
    m10a = json.loads((K.CARS / 'PHASE10A_DEV' / 'FROZEN_CODE_MANIFEST10A.json').read_text())
    for f in list(m10a['phase10a_files']) + list(m10a['frozen_dependencies']):
        files[f] = K.sha256_file(K.CARS / f)
    fam = Family3D('E3-TS', 'S0', seed=0, split='dev')
    fx = Path(sys.modules['cars_m2.families'].PHASE3)
    for f in sorted(fx.glob('*.json')):
        files[rel(f)] = K.sha256_file(f)
    agg = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    a10_changed = {f: dict(frozen_10a=h, now=files[f]) for f, h in {**m10a['phase10a_files'], **m10a['frozen_dependencies']}.items()
                   if files[f] != h}
    code = dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S %z'), aggregate_sha256=agg, n_files=len(files),
                imported_cars_modules=imported, files=files, phase10a_manifest_aggregate=m10a['aggregate_sha256'],
                phase10a_files_changed_since_10a_freeze=a10_changed,
                note=('analyze10a.py differs from the Phase 10A freeze by one line (stage list widened 13 s after that '
                      'freeze, before the 10A benchmark finished); it is not used by Phase 10B. All method, verifier, '
                      'EFT and dependency files are identical to the Phase 10A freeze.'))
    K.REPORTS.mkdir(parents=True, exist_ok=True)
    K.CODE_FREEZE.write_text(json.dumps(code, indent=1))
    # environment
    import numpy as np
    pk = {}
    for d in ('numpy', 'scipy', 'pypardiso', 'mkl', 'python-flint', 'cholespy', 'psutil', 'intel-openmp', 'tbb'):
        try:
            pk[d] = md.version(d)
        except md.PackageNotFoundError:
            pk[d] = None
    lib = runner10a.R6._MKL.get()
    buf = ctypes.create_string_buffer(256)
    try:
        lib.MKL_Get_Version_String(buf, 256); mklv = buf.value.decode(errors='replace')
    except AttributeError:
        mklv = None
    try:
        cpu = subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name'],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        plan = subprocess.run(['powercfg', '/getactivescheme'], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:                                           # recorded, not fatal
        cpu = plan = repr(e)
    env = dict(recorded_at=time.strftime('%Y-%m-%d %H:%M:%S %z'), os=platform.platform(), machine=platform.machine(),
               cpu=cpu, logical_cpus=psutil.cpu_count(), physical_cores=psutil.cpu_count(logical=False),
               ram_GiB=round(psutil.virtual_memory().total / 2 ** 30, 2), power_scheme=plan,
               python=platform.python_version(), python_impl=platform.python_implementation(), packages=pk,
               mkl_version_string=mklv, numpy_config_blas=str(getattr(np, '__config__', None) and 'see numpy.show_config'),
               venv='PHASE5_DEV/.venv (include-system-site-packages = true)',
               thread_modes={k: dict(threads=v[0], affinity=v[1]) for k, v in K.MODES.items()},
               note='Single machine; no second machine is available (see the final reports).')
    (K.REPORTS / 'PHASE10B_ENVIRONMENT.json').write_text(json.dumps(env, indent=1))
    print(agg, len(files), 'imported', len(imported), 'changed_vs_10a', list(a10_changed))
    print(json.dumps(env, indent=1)[:1500])


if __name__ == '__main__':
    main()
