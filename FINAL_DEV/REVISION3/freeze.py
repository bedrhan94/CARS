"""Revision 2: hash-freeze a protocol directory before any timed cell. Usage: python freeze.py <dir> <note> <file>...
Writes <dir>/PROTOCOL_FREEZE.json (refuses to overwrite)."""
import hashlib, json, sys, time
from pathlib import Path

d = Path(sys.argv[1]); note = sys.argv[2]; files = sys.argv[3:]
out = d / 'PROTOCOL_FREEZE.json'
if out.exists():
    raise SystemExit('already frozen')
rec = dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S %z'), note=note,
           files={f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in files})
out.write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
