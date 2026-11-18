"""Phase 10B shared definitions and guards (no computation).

Held-out access is allowed only if the pre-registration freeze record exists and every frozen document and code file
still has its frozen SHA-256 (check_freeze). Benchmark cells additionally require the frozen held-out input manifest
(check_inputs_frozen). Development seeds 0-2 remain usable for harness smoke tests."""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
REPORTS = CARS / 'reports' / 'PHASE10B'
FREEZE_RECORD = REPORTS / 'PREREG_FREEZE_RECORD.json'
CODE_FREEZE = REPORTS / 'PHASE10B_CODE_FREEZE.json'
INPUTS = HERE / 'inputs'
INPUT_MANIFEST = INPUTS / 'inputs_manifest10b.json'

HELDOUT_SEEDS = (200, 201, 202, 203, 204)
DEV_SEEDS = (0, 1, 2)
FAMILIES = ('E3-TS', 'D3-TS')
SIZES = ('S2', 'S3')                      # S3 primary, S2 secondary; S4 is never allowed
MODES = {'ST': (1, [2]), 'MT': (8, [0, 2, 4, 6, 8, 10, 12, 14])}
TARGET = 1e-8
ARMS = [('A', 'RPS|P1F'), ('B1', 'RPS-EFT'), ('B', 'RPS-OPT'), ('B9', 'RPS-OPT-S09'), ('C', 'RPS|M2A'), ('D', 'M2A-EFT'),
        ('D2', 'M2A-EFT-S05')]
PRIMARY = {'E3-TS': ('B9', 'D'), 'D3-TS': ('B', 'D2')}       # (per-system, reuse); fixed from Phase 10A dev S3


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def check_freeze():
    """Raise SystemExit unless the pre-registration freeze record exists and all frozen files are unchanged."""
    if not FREEZE_RECORD.exists():
        raise SystemExit('FREEZE GUARD: no pre-registration freeze record; held-out access refused')
    rec = json.loads(FREEZE_RECORD.read_text())
    bad = [rel for rel, h in rec['documents'].items() if sha256_file(CARS / rel) != h]
    code = json.loads(CODE_FREEZE.read_text())
    bad += [rel for rel, h in code['files'].items() if sha256_file(CARS / rel) != h]
    if bad:
        raise SystemExit('FREEZE GUARD: frozen files changed after the freeze: ' + ', '.join(bad[:10]))
    return rec['code_aggregate_sha256']


def check_inputs_frozen():
    if not INPUT_MANIFEST.exists():
        raise SystemExit('INPUT GUARD: held-out inputs are not frozen')
    return json.loads(INPUT_MANIFEST.read_text())


def seed_guard(seed, split):
    if split == 'heldout' and seed in HELDOUT_SEEDS:
        return 'heldout'
    if split == 'dev' and seed in DEV_SEEDS:
        return 'dev'
    raise SystemExit(f'SEED GUARD: seed {seed} / split {split} refused (seeds 100-104 are never used here)')
