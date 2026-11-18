"""Phase 10A code freeze: SHA-256 of every Phase 10A source file and of the frozen dependencies it imports, written
after validation (V1, V1E) and BEFORE the paired benchmark (P2) and the S3 diagnostic (D3). The same manifest is the
proposed frozen code basis for Phase 10B. Usage: python freeze_code10a.py -> FROZEN_CODE_MANIFEST10A.json"""
import hashlib, json, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CARS = HERE.parent
OWN = sorted([*HERE.glob('methods/*.py'), *HERE.glob('eft/*.py'), *HERE.glob('analysis/*.py'),
              HERE / 'run_plan10a.py', HERE / 'freeze_inputs10a.py', HERE / 'freeze_code10a.py'])
DEPS = ['PHASE6_DEV/lean6/runner6.py', 'PHASE6_DEV/lean6/replay6.py', 'PHASE5_DEV/lean/fastbind.py',
        'PHASE5_DEV/lean/verify5.py', 'PHASE5_DEV/lean/runner5.py', 'PHASE5_DEV/arith/independent5.py',
        'PHASE3B_DEV/work/CARS_M0_M1/cars/_reference_m0.py', 'PHASE3B_DEV/work/CARS_M0_M1/cars_m2/families.py',
        'PHASE3B_DEV/work/CARS_M0_M1/cars_m2/runner3b.py', 'reference_impl/fem3d_ref.py', 'reference_impl/m2_ref.py']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    files = {p.relative_to(CARS).as_posix(): sha(p) for p in OWN if '__pycache__' not in p.parts}
    deps = {d: sha(CARS / d) for d in DEPS if (CARS / d).exists()}
    agg = hashlib.sha256(json.dumps([files, deps], sort_keys=True).encode()).hexdigest()
    out = dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'), phase10a_files=files, frozen_dependencies=deps,
               aggregate_sha256=agg, missing_deps=[d for d in DEPS if not (CARS / d).exists()])
    (HERE / 'FROZEN_CODE_MANIFEST10A.json').write_text(json.dumps(out, indent=1))
    print(agg, len(files), len(deps), out['missing_deps'])


if __name__ == '__main__':
    main()
