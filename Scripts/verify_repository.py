"""Integrity checks for archived results, using only the Python standard library."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'Data'


def main():
    errors=[]
    manifest=DATA/'SHA256_MANIFEST.csv'
    with manifest.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        p=ROOT/row['path']
        if not p.is_file():
            errors.append(f'MISSING {row["path"]}')
            continue
        raw=p.read_bytes()
        if len(raw)!=int(row['bytes']) or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            errors.append(f'HASH MISMATCH {row["path"]}')

    groups={'environment_4x4_v31':16,'v31_regression_suite':6,'v31_heading40_combo':1}
    with (DATA/'RECORDED_CASES.csv').open(newline='',encoding='utf-8') as f:
        cases=list(csv.DictReader(f))
    if len(cases)!=sum(groups.values()):
        errors.append(f'Expected {sum(groups.values())} case rows but found {len(cases)}')
    for grp,n in groups.items():
        files=list((DATA/'RawRuns'/grp).glob('*/metrics.json'))
        if len(files)!=n:errors.append(f'{grp}: expected {n} metrics files, got {len(files)}')
        for mpath in files:
            m=json.loads(mpath.read_text(encoding='utf-8'))
            if m.get('status')!='SUCCESS':
                errors.append(f'Unexpected status in {mpath}: {m.get("status")}')
            if not (mpath.parent/'trajectory.csv').is_file():
                errors.append(f'Missing trajectory beside {mpath}')
        summary=DATA/'RawRuns'/grp/'summary.json'
        if not summary.exists() or len(json.loads(summary.read_text(encoding='utf-8')))!=n:
            errors.append(f'Incorrect summary for {grp}')

    print(f'Integrity: {len(rows)} archived data files checked; {len(cases)} cases indexed.')
    if errors:
        for e in errors:print('ERROR:',e)
        raise SystemExit(1)
    print('PASS: every archived data hash matches and all 23 recorded cases are present.')

if __name__=='__main__': main()
