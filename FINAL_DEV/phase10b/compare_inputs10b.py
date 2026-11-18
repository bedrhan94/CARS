"""Phase 10B: compare two independent generation passes bit for bit (every file, byte for byte, apart from the run
metadata in generation.json) and, if identical, write the frozen input manifest with the SHA-256 of every frozen file.
Usage: python compare_inputs10b.py <pass_a> <pass_b> <frozen_manifest_out>"""
import json, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common10b as K                                               # noqa: E402


def files(root):
    return {p.relative_to(root).as_posix(): K.sha256_file(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != 'generation.json'}


def main():
    a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    fa, fb = files(a), files(b)
    ga = json.loads((a / 'generation.json').read_text()); gb = json.loads((b / 'generation.json').read_text())
    diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    man_equal = ga['manifest'] == gb['manifest']
    res = dict(compared_at=time.strftime('%Y-%m-%d %H:%M:%S'), pass_a=a.name, pass_b=b.name, pid_a=ga['pid'], pid_b=gb['pid'],
               files_compared=len(fa), differing_files=diff, manifest_equal=man_equal,
               IDENTICAL=(not diff and man_equal and len(fa) == len(fb)))
    if res['IDENTICAL']:
        res.update(kind=ga['kind'], manifest=ga['manifest'], frozen_files=fa, frozen_dir=a.name)
    out.write_text(json.dumps(res, indent=0))
    print(json.dumps({k: res[k] for k in ('files_compared', 'differing_files', 'manifest_equal', 'IDENTICAL')}))
    if not res['IDENTICAL']:
        raise SystemExit('INPUT GENERATION NOT REPRODUCIBLE: hard stop')


if __name__ == '__main__':
    main()
