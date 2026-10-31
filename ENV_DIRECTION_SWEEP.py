"""Independent ship-state Dec-NMPC environmental direction validation.

Usage: python ENV_DIRECTION_SWEEP.py --preset cardinal --duration 6500 --maxiter 6 --out results/environment_sweep
All directions are TOWARD, relative to earth-fixed X (CCW). A run is
successful only if the simulator's strict berthing monitor reports SUCCESS.
No bad cases are excluded; output is checkpointed after every case.
"""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    psi0: float
    wind_mps: float
    wind_to_deg: float
    current_mps: float
    current_to_deg: float


def build_cases(preset='cardinal', psi0=(20.,), wind=(8.,), current=(0.25,),
                wind_angles=(0.,90.,180.,270.), current_angles=(0.,90.,180.,270.)):
    """Generate unique valid single- and combined-disturbance scenarios."""
    if preset not in ('cardinal','calm','wind','current','combined','all'):
        raise ValueError('Unsupported preset')
    cases=[]
    def add(psi,w,wa,c,ca):
        # Fixed deterministic case IDs; scenario keys include exact parameters.
        def f(v):return f'{float(v):g}'.replace('-','m').replace('.','p')
        kind=('combined' if w and c else 'wind' if w else 'current' if c else 'calm')
        name='_'.join((kind,'psi'+f(psi),'w'+f(w),'wd'+f(wa),'c'+f(c),'cd'+f(ca)))
        cases.append(Scenario(name,float(psi),float(w),float(wa),float(c),float(ca)))
    for p in psi0:
        if preset in ('cardinal','calm','all'):add(p,0,0,0,0)
        if preset in ('cardinal','wind','all'):
            for w,a in product(wind,wind_angles):
                if w>0:add(p,w,a,0,0)
        if preset in ('cardinal','current','all'):
            for c,a in product(current,current_angles):
                if c>0:add(p,0,0,c,a)
        if preset in ('cardinal','combined','all'):
            for w,c,wa,ca in product(wind,current,wind_angles,current_angles):
                if w>0 and c>0:add(p,w,wa,c,ca)
    return list(dict((x.name,x) for x in cases).values())


def write_summary(folder, records):
    (folder/'summary.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    if records:
        names=list(dict.fromkeys(k for row in records for k in row))
        with (folder/'summary.csv').open('w',encoding='utf8',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=names)
            writer.writeheader();writer.writerows(records)
    plot_results(folder, records)


def plot_results(folder, records):
    # Summary chart is deliberately not conditional on --plots.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np
    except ImportError:
        return
    rows=[r for r in records if r.get('status')!='RUN_ERROR']
    if not rows:return
    from collections import Counter
    counts=Counter(r['status'] for r in records)
    fig,ax=plt.subplots(figsize=(7,4))
    keys=['SUCCESS','TIMEOUT','UNSAFE','RUN_ERROR']
    ax.bar(keys,[counts.get(x,0) for x in keys],color=['#21875B','#DD9E3B','#C34B52','#7F7F7F'])
    ax.set_title('Ship-state NMPC: verified environmental cases')
    ax.set_ylabel('Number of cases')
    ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(folder/'status_counts.png',dpi=160);plt.close(fig)
    # For each (heading, wind speed, current speed), show the complete wind/
    # current angle grid. Unrun cells are grey, never treated as successes.
    groups={}
    for r in records:
        if float(r['wind_mps'])>0 and float(r['current_mps'])>0:
            key=(r['psi0_deg'],r['wind_mps'],r['current_mps'])
            groups.setdefault(key,[]).append(r)
    for (psi,w,c), group in groups.items():
        xs=sorted(set(float(r['wind_to_deg']) for r in group))
        ys=sorted(set(float(r['current_to_deg']) for r in group))
        matrix=np.full((len(ys),len(xs)),-1,int)
        for r in group:
            i=ys.index(float(r['current_to_deg']));j=xs.index(float(r['wind_to_deg']))
            matrix[i,j]={'SUCCESS':2,'TIMEOUT':1,'UNSAFE':0,'RUN_ERROR':-1}.get(r['status'],-1)
        fig,ax=plt.subplots(figsize=(max(6,len(xs)*1.2),max(4,len(ys)*.95)))
        cmap=ListedColormap(['#AAAAB1','#D25A57','#E4B24A','#55A77E'])
        ax.imshow(matrix+1,cmap=cmap,vmin=0,vmax=3,aspect='auto')
        ax.set_xticks(range(len(xs)),[f'{x:g}°' for x in xs]);ax.set_yticks(range(len(ys)),[f'{x:g}°' for x in ys])
        ax.set_xlabel('Wind TOWARD direction');ax.set_ylabel('Current TOWARD direction')
        ax.set_title(f'psi0={psi:g}°, wind={w:g} m/s, current={c:g} m/s')
        for i in range(len(ys)):
            for j in range(len(xs)):
                ax.text(j,i,{-1:'—',0:'UNSAFE',1:'TIMEOUT',2:'PASS'}[int(matrix[i,j])],
                        fontsize=8,ha='center',va='center')
        fig.tight_layout()
        fig.savefig(folder/f'angle_matrix_psi{psi:g}_w{w:g}_c{c:g}.png',dpi=160)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=['cardinal','calm','wind','current','combined','all'],default='cardinal')
    parser.add_argument('--psi0',type=float,nargs='+',default=[20.])
    parser.add_argument('--wind-mps',type=float,nargs='+',default=[8.])
    parser.add_argument('--current-mps',type=float,nargs='+',default=[.25])
    parser.add_argument('--wind-angles',type=float,nargs='+',default=[0.,90.,180.,270.])
    parser.add_argument('--current-angles',type=float,nargs='+',default=[0.,90.,180.,270.])
    parser.add_argument('--target-x',type=float,default=50.)
    parser.add_argument('--target-y',type=float,default=-100.)
    parser.add_argument('--duration',type=float,default=6500.)
    parser.add_argument('--maxiter',type=int,default=6)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--out',default='results/environment_direction_sweep')
    parser.add_argument('--max-cases',type=int,default=0,help='Cap number of cases; 0 runs all')
    parser.add_argument('--plots',action='store_true',help='Generate full time-series plots for every case (slower)')
    parser.add_argument('--no-resume',action='store_true',help='Rerun even if a completed metrics.json is present')
    parser.add_argument('--environmental-reserve',choices=['auto','on','off'],default='auto')
    args=parser.parse_args()
    if args.duration<=0 or args.maxiter<=0:parser.error('duration and maxiter must be positive')
    if any(x<0 for x in args.wind_mps+args.current_mps):parser.error('Speeds must be nonnegative')
    cases=build_cases(args.preset,args.psi0,args.wind_mps,args.current_mps,args.wind_angles,args.current_angles)
    if args.max_cases>0:cases=cases[:args.max_cases]
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    (root/'configuration.json').write_text(json.dumps(vars(args),indent=2),encoding='utf8')
    from run_nmpc import rollout, CONTROLLER_VERSION
    records=[]
    for index,case in enumerate(cases,1):
        folder=root/case.name;metrics=folder/'metrics.json'
        print(f'[ENV SWEEP {index}/{len(cases)}] {case.name}',flush=True)
        try:
            if metrics.exists() and not args.no_resume:
                m=json.loads(metrics.read_text(encoding='utf8'))
                if m.get('controller_version')!=CONTROLLER_VERSION or m.get('position_tolerance_m')!=2.5:
                    raise ValueError('Existing results incompatible with current controller; rerun with --no-resume or use a new output directory')
                # Check scenario and run settings; a new sweep with incompatible
                # duration/configuration must not silently reuse old metrics.
                for key,value in [('wind_mps',case.wind_mps),('wind_to_deg',case.wind_to_deg),('current_mps',case.current_mps),('current_to_deg',case.current_to_deg)]:
                    if abs(float(m[key])-value)>1e-6:raise ValueError('Existing metrics incompatible with this case')
            else:
                m=rollout(out=str(folder),controller_method='ship_state_local',
                          psi0=case.psi0,wind_mps=case.wind_mps,wind_to_deg=case.wind_to_deg,
                          current_mps=case.current_mps,current_to_deg=case.current_to_deg,
                          target_x=args.target_x,target_y=args.target_y,
                          seed=args.seed,max_seconds=args.duration,maxiter=args.maxiter,
                          environmental_reserve=args.environmental_reserve,
                          render=args.plots,verbose=False)
            rec=dict(asdict(case),psi0_deg=case.psi0,status=m['status'],success=bool(m['success']),
                     unsafe=bool(m['unsafe']),time_s=m['seconds'],position_error_m=m['position_error_m'],
                     heading_error_deg=m['heading_error_deg'],min_hull_gap_m=m['min_logged_hull_gap_m'],
                     output_folder=str(folder))
        except Exception as ex:
            rec=dict(asdict(case),psi0_deg=case.psi0,status='RUN_ERROR',success=False,unsafe=None,
                     error=f'{type(ex).__name__}: {ex}',output_folder=str(folder))
            print(f'[ENV SWEEP] ERROR: {rec["error"]}',flush=True)
        records.append(rec)
        write_summary(root,records)
        print(f'[ENV SWEEP] {rec["status"]} (position err {rec.get("position_error_m",float("nan")):.2f} m)',flush=True)
    print(f'[ENV SWEEP] Summary: {root/"summary.csv"}',flush=True)


if __name__=='__main__':main()
