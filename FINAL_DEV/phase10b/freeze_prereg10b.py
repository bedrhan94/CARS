"""Phase 10B pre-registration freeze record. Run AFTER freeze_code10b.py and after the final pre-registration and the
pre-data amendments are written, and BEFORE any held-out input is generated. Refuses to run if held-out inputs exist
or if any trace of seeds 200-204 is found in the workspace. Writes reports/PHASE10B/PREREG_FREEZE_RECORD.json."""
import json, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common10b as K                                                # noqa: E402

DOCS = ['reports/PHASE10B/PHASE10B_PREREGISTRATION_FINAL.md', 'reports/PHASE10B/PHASE10B_PREDATA_AMENDMENTS.md',
        'reports/PHASE10B/PHASE10B_CODE_FREEZE.json', 'reports/PHASE10B/PHASE10B_ENVIRONMENT.json',
        'reports/PHASE10A/PHASE10B_PREREGISTRATION_DRAFT.md']


def main():
    if K.FREEZE_RECORD.exists():
        raise SystemExit('already frozen')
    if K.INPUTS.exists():
        raise SystemExit('held-out inputs already exist: freeze refused')
    scan = subprocess.run([sys.executable, '-B', str(K.CARS / 'FINAL_DEV' / 'stage0' / 'heldout_seed_scan.py')],
                          capture_output=True, text=True)
    sres = json.loads((K.CARS / 'FINAL_DEV' / 'stage0' / 'heldout_seed_scan.json').read_text())
    if sres['name_hits'] or sres['record_hits']:
        raise SystemExit('trace of held-out seeds found: freeze refused')
    code = json.loads(K.CODE_FREEZE.read_text())
    rec = dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S %z'), documents={d: K.sha256_file(K.CARS / d) for d in DOCS},
               code_aggregate_sha256=code['aggregate_sha256'], heldout_scan=dict(files_walked=sres['files_walked'],
               json_scanned=sres['json_scanned'], hits=0, scanned_at=sres['scanned_at']),
               statement='No held-out input (seeds 200-204) existed or had been inspected when this record was written.')
    K.FREEZE_RECORD.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
