"""Reproducible evaluation of four independent ship-state NMPC controllers.

Runs cases sequentially and writes a summary after EACH case; includes timeouts
and failures. No success label is inferred from a low endpoint error alone.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from run_nmpc import rollout

SCENARIOS={
    'calm20':dict(psi0=20.,target_x=50.,target_y=-100.),
    'calm40':dict(psi0=40.,target_x=50.,target_y=-100.),
    'deep200':dict(psi0=20.,target_x=50.,target_y=-200.),
    'wind20':dict(psi0=20.,target_x=50.,target_y=-100.,wind_mps=8.,wind_to_deg=90.),
    'current20':dict(psi0=20.,target_x=50.,target_y=-100.,current_mps=.25,current_to_deg=90.),
    'combined20':dict(psi0=20.,target_x=50.,target_y=-100.,wind_mps=8.,wind_to_deg=90.,current_mps=.25,current_to_deg=90.),
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',choices=['all',*SCENARIOS],nargs='+',default=['calm20'])
    p.add_argument('--out',default='results/ship_state_suite')
    p.add_argument('--duration',type=float,default=5400.)
    p.add_argument('--maxiter',type=int,default=8)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--plots',action='store_true')
    p.add_argument('--post-success-seconds',type=float,default=0.)
    p.add_argument('--save-every-seconds',type=float,default=300.)
    args=p.parse_args()
    cases=list(SCENARIOS) if 'all' in args.cases else list(dict.fromkeys(args.cases))
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in cases:
        folder=root/name
        print(f'[SHIP-STATE] RUNNING {name}: {SCENARIOS[name]}',flush=True)
        try:
            m=rollout(out=str(folder),controller_method='ship_state_local',
                      max_seconds=args.duration,maxiter=args.maxiter,seed=args.seed,
                      render=args.plots,post_success_seconds=args.post_success_seconds,
                      save_every_seconds=args.save_every_seconds,**SCENARIOS[name])
            record=dict(case=name,status=m['status'],success=bool(m['success']),
                unsafe=bool(m['unsafe']),time_s=m['seconds'],
                position_error_m=m['position_error_m'],
                heading_error_deg=m['heading_error_deg'],
                min_logged_hull_gap_m=m['min_logged_hull_gap_m'],
                output_folder=str(folder))
        except Exception as exc:
            record=dict(case=name,status='RUN_ERROR',success=False,unsafe=None,
                        time_s=None,position_error_m=None,heading_error_deg=None,
                        min_logged_hull_gap_m=None,output_folder=str(folder),error=str(exc))
            print(f'[SHIP-STATE] ERROR {name}: {exc}',flush=True)
        records.append(record)
        (root/'summary.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf8')
        cols=list(dict.fromkeys(key for r in records for key in r))
        with (root/'summary.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=cols)
            writer.writeheader();writer.writerows(records)
        print(f'[SHIP-STATE] DONE {name}: {record["status"]}',flush=True)
    print(f'[SHIP-STATE] Summary: {root/"summary.csv"}',flush=True)

if __name__=='__main__':main()
