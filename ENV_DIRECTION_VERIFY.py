"""Audit recorded environmental sweeps for completeness and no peer access.

Does not execute physics and does not relabel outcomes. Exits nonzero if an
expected case was not run, failed berthing, or violated the information boundary.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from ENV_DIRECTION_SWEEP import build_cases


def verify(folder: Path, expected=None):
    if not (folder/'summary.json').exists():
        raise FileNotFoundError(folder/'summary.json')
    root=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    records={r['name']:r for r in root}
    missing=[];failed=[];boundary=[]
    for scenario in expected if expected is not None else []:
        name=scenario.name
        if name not in records or not (folder/name/'metrics.json').exists():
            missing.append(name);continue
        r=records[name]
        m=json.loads((folder/name/'metrics.json').read_text(encoding='utf-8'))
        if m['status']!='SUCCESS' or m['unsafe'] or not m['success'] or m['position_error_m']>2.5 or m['heading_error_deg']>2.:failed.append((name,m['status']))
        if m['controller_method']!='ship_state_local' or m['peer_state_access'] or m['communication_between_tugs'] or m['centralized_joint_optimizer'] or m['centralized_force_allocator']:
            boundary.append(name)
        if (folder/name/'local_nmpc_decisions.csv').exists():
            with (folder/name/'local_nmpc_decisions.csv').open(newline='',encoding='utf-8') as f:
                rows=csv.DictReader(f)
                if any(float(row['fraction'])>1.000001 or float(row['fraction'])<-1e-6 for row in rows):
                    boundary.append(name+' (thrust fraction)')
    print(f'Expected={len(expected or [])}; Recorded={len(records)}; Missing={len(missing)}; Failed={len(failed)}; Boundary/actuator violations={len(boundary)}')
    for kind,items in [('NOT TESTED',missing),('NOT SUCCESS',failed),('BOUNDARY VIOLATION',boundary)]:
        for item in items:print(f'[{kind}] {item}')
    return not (missing or failed or boundary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='results/environment_4x4_validated')
    p.add_argument('--psi0',type=float,default=20.)
    p.add_argument('--wind-mps',type=float,default=8.)
    p.add_argument('--current-mps',type=float,default=.25)
    p.add_argument('--wind-angles',type=float,nargs='+',default=[0,90,180,270])
    p.add_argument('--current-angles',type=float,nargs='+',default=[0,90,180,270])
    a=p.parse_args()
    cases=build_cases('combined',(a.psi0,),(a.wind_mps,),(a.current_mps,),a.wind_angles,a.current_angles)
    raise SystemExit(0 if verify(Path(a.results),cases) else 1)
