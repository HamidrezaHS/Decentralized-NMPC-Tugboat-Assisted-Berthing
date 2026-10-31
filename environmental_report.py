"""Reproducible figures/tables for wind/current robustness comparisons.

Success and unsafe proportions are descriptive for the actually completed runs.
Time-to-success is right-censored for unsuccessful runs; never average it by
assigning the timeout time to failure as if it had succeeded.
"""
from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _read(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def _float(value):
    try:return float(value)
    except (ValueError,TypeError):return np.nan

def _export(fig,folder,name):
    fig.savefig(folder/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def generate_report(directory):
    folder=Path(directory)
    summary=folder/'comparison.csv'
    if not summary.exists():
        # Recover ALL finished cases when an interrupted run was not merged.
        records=[]
        for path in sorted(folder.glob('*/metrics.json')):
            c=path.parent/'configuration.json'
            if not c.exists():continue
            m=json.loads(path.read_text(encoding='utf8'))
            cfg=json.loads(c.read_text(encoding='utf8'))
            records.append(dict(scenario=path.parent.name.split('_h')[0],
                psi0_deg=cfg['psi0'],mode=cfg.get('disturbance_mode','measured'),
                status=m['status'],success=int(m['success']),unsafe=int(m['unsafe']),
                time_s=m['seconds'],position_error_m=m['position_error_m'],
                heading_error_deg=m['heading_error_deg'],wind_mps=cfg.get('wind_mps',0),
                current_mps=cfg.get('current_mps',0),
                integrated_abs_thrust_Ns=m.get('integrated_abs_thrust_Ns',np.nan),
                avg_optimizer_ms=m.get('avg_optimizer_ms',np.nan),
                case_dir=str(path.parent)))
        if not records:raise FileNotFoundError('No completed case available for report')
        with summary.open('w',encoding='utf8',newline='') as f:
            keys=list(dict.fromkeys(k for r in records for k in r))
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
    runs=_read(summary)
    if not runs:raise ValueError('No comparison rows')
    modes=sorted(set(r['mode'] for r in runs),key=lambda a:['none','residual','measured'].index(a))
    scenarios=list(dict.fromkeys(r['scenario'] for r in runs))
    headings=list(dict.fromkeys(r['psi0_deg'] for r in runs))
    groups=defaultdict(list)
    for r in runs:groups[(r['scenario'],r['mode'])].append(r)
    # Empirical aggregate only; avoid claiming significance with one seed.
    rows=[]
    for (case,mode), rr in groups.items():
        successful=[r for r in rr if r['status']=='SUCCESS']
        rows.append(dict(scenario=case,mode=mode,runs=len(rr),
            successes=len(successful),unsafe=sum(r['status']=='UNSAFE' for r in rr),
            timeouts=sum(r['status']=='TIMEOUT' for r in rr),
            success_fraction=len(successful)/len(rr),
            mean_final_position_error_m=float(np.mean([_float(r['position_error_m']) for r in rr])),
            mean_final_heading_error_deg=float(np.mean([_float(r['heading_error_deg']) for r in rr])),
            mean_success_time_s=(float(np.mean([_float(r['time_s']) for r in successful])) if successful else ''),
            mean_thrust_integral_Ns=float(np.nanmean([_float(r.get('integrated_abs_thrust_Ns')) for r in rr])),
            mean_total_abs_thrust_N=float(np.nanmean([_float(r.get('mean_total_abs_thrust_N',
                _float(r.get('integrated_abs_thrust_Ns'))/max(_float(r.get('time_s')),1.))) for r in rr])),
            mean_optimizer_ms=float(np.nanmean([_float(r.get('avg_optimizer_ms')) for r in rr]))))
    with (folder/'comparison_aggregate.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    # Matrix: successes / run count; show sample count in each cell.
    fig,ax=plt.subplots(figsize=(max(8,1.6*len(modes)+2),max(4.5,.55*len(scenarios)+2)),constrained_layout=True)
    mat=np.full((len(scenarios),len(modes)),np.nan)
    for i,case in enumerate(scenarios):
        for j,mode in enumerate(modes):
            rr=groups.get((case,mode),[])
            if rr:mat[i,j]=sum(r['status']=='SUCCESS' for r in rr)/len(rr)
    ax.imshow(np.ma.masked_invalid(mat),vmin=0,vmax=1,cmap='viridis',aspect='auto')
    for i,case in enumerate(scenarios):
        for j,mode in enumerate(modes):
            rr=groups.get((case,mode),[])
            txt=f'{sum(r["status"]=="SUCCESS" for r in rr)}/{len(rr)}' if rr else 'n/a'
            ax.text(j,i,txt,ha='center',va='center',color='white' if not rr or mat[i,j]<.65 else 'black',fontsize=10)
    ax.set_xticks(range(len(modes)),modes)
    ax.set_yticks(range(len(scenarios)),scenarios)
    ax.set_title('Verified safe successes / completed runs (not a statistical claim)')
    ax.set_xlabel('Disturbance compensation method')
    _export(fig,folder,'01_success_matrix')

    # Terminal errors and effort across scenarios/modes.
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    xs=np.arange(len(scenarios));width=.8/max(len(modes),1)
    metrics=[('mean_final_position_error_m','Final position error (m)'),
             ('mean_final_heading_error_deg','Final heading error (deg)'),
             ('mean_total_abs_thrust_N','Mean total |thrust| (MN)'),
             ('mean_optimizer_ms','Per-tug optimizer runtime (ms)')]
    for axis,(key,label) in zip(axs.flat,metrics):
        for j,mode in enumerate(modes):
            vals=[]
            for c in scenarios:
                row=next((r for r in rows if r['scenario']==c and r['mode']==mode),None)
                val=_float(row[key]) if row else np.nan
                if key=='mean_total_abs_thrust_N':val/=1e6
                vals.append(val)
            axis.bar(xs-.4+width*(j+.5),vals,width,label=mode)
        axis.set_ylabel(label);axis.set_xticks(xs,scenarios,rotation=20,ha='right')
        axis.grid(axis='y',alpha=.2)
    axs[0,0].legend(ncol=len(modes),fontsize=9)
    fig.suptitle('Independent NMPC robustness, actuation and compute cost')
    _export(fig,folder,'02_errors_effort_compute')

    # Representative path/time-history overlays: same heading and scenario;
    # compare compensation methods on identical environment.
    if runs:
        firsthead=headings[0]
        sample_case=next((c for c in scenarios if c!='calm'),'calm')
        selections=[r for r in runs if r['scenario']==sample_case and r['psi0_deg']==firsthead]
        fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for r in selections:
            file=Path(r['case_dir'])/'trajectory.csv'
            if not file.exists():
                file=Path(__file__).resolve().parent/str(r['case_dir'])/'trajectory.csv'
            if not file.exists():
                file=folder/str(r['case_dir'])/'trajectory.csv'
            if not file.exists():continue
            traj=_read(file)
            t=np.array([_float(q['time_s']) for q in traj])/60
            xx=np.array([_float(q['x_m']) for q in traj]);yy=np.array([_float(q['y_m']) for q in traj])
            epos=np.array([_float(q['position_error_m']) for q in traj]);hd=np.array([_float(q['heading_error_deg']) for q in traj])
            label=f"{r['mode']} ({r['status']})"
            axs[0,0].plot(xx,yy,label=label)
            axs[0,1].plot(t,epos,label=label)
            axs[1,0].plot(t,hd,label=label)
            axs[1,1].plot(t,np.array([_float(q['hull_gap_m']) for q in traj]),label=label)
        axs[0,0].set(xlabel='East X (m)',ylabel='North Y (m)')
        axs[0,0].set_aspect('equal',adjustable='box')
        axs[0,1].set(xlabel='Time (min)',ylabel='Position error (m)')
        axs[1,0].set(xlabel='Time (min)',ylabel='Heading error (deg)')
        axs[1,1].set(xlabel='Time (min)',ylabel='Hull–quay clearance (m)')
        for a in axs.flat:
            a.grid(alpha=.25)
            if a.get_legend_handles_labels()[0]:a.legend(fontsize=8)
        fig.suptitle(f'Same disturbance, identical start: {sample_case} / heading {firsthead} deg')
        _export(fig,folder,'03_same_environment_comparison')
    (folder/'REPORT_NOTES.txt').write_text(
        'All curves/tables use logged simulator data. TIMEOUT and UNSAFE are NOT successes.\\n'
        'Success time is averaged only among successful runs (other times censored).\\n'
        'Use >=5 random seeds per setup before claims about variance or superiority.\\n'
        'Wind/current headings denote TO direction; convert wind FROM by +180 degrees.\\n'
        'Environmental coefficients and contact model are modeling assumptions, not sea-trial calibration.\\n',
        encoding='utf8')
    print(f'Paper report ready: {folder} ({len(runs)} actual runs)',flush=True)
    return folder

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/nmpc_environment')
    a=p.parse_args();generate_report(a.input)
