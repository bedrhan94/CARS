"""Revision 2, post hoc (referee comment on a sharper Rump constant): size of the Rump constant c relative to the shift
sigma in every Rump anchor of the confirmatory records (all cells, ST and MT, S2 and S3, all arms). A sharper constant
(Rump 2006, bounds II/III) can raise the certified bound lambda_lo = sigma by at most c (Corollary 2.4: the certified
bound is sigma, and the diagonal reduction is sigma + c), so c/sigma bounds the possible relative gain.
Writes rump_constant_share.json."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
RUNS = CARS / 'FINAL_DEV' / 'phase10b' / 'runs'


def val(z):
    return float.fromhex(z) if isinstance(z, str) else float(z)


def main():
    per = defaultdict(list); seen = set(); n_rec = 0
    for cfg in ('ST_S2', 'ST_S3', 'MT_S2', 'MT_S3'):
        for f in sorted((RUNS / cfg).glob('*/*/*.jsonl')):
            if '_warmup' in f.parts:
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line); n_rec += 1
                for l in r.get('lambda_chain') or []:
                    if l.get('kind') != 'rump' or l.get('c') is None or l.get('sigma') is None:
                        continue
                    key = (l.get('anchor_matrix_sha256') or r['matrix_sha256'], l['sigma'])
                    if key in seen:
                        continue
                    seen.add(key)
                    per[(r['family'], r['size'])].append(val(l['c']) / val(l['sigma']))
    out = {f'{k[0]}|{k[1]}': dict(anchors=len(v), max=float(max(v)), median=float(np.median(v))) for k, v in sorted(per.items())}
    allv = [x for v in per.values() for x in v]
    res = dict(records_scanned=n_rec, distinct_anchors=len(allv), c_over_sigma_max=float(max(allv)),
               c_over_sigma_median=float(np.median(allv)), per_config=out)
    (HERE / 'rump_constant_share.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
