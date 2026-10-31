"""Rebuild reproducible environmental-sweep figures without rerunning physics.

Run: python ENV_DIRECTION_REPORT.py --results results/environment_direction_sweep
Only completed simulations count; not-yet-run combinations are NOT_TESTED.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def report(root: Path):
    summary=root/'summary.json'
    if not summary.exists():
        raise FileNotFoundError(f'Cannot find {summary}; run ENV_DIRECTION_SWEEP.py first')
    records=json.loads(summary.read_text(encoding='utf-8'))
    configuration=root/'configuration.json'
    opts=json.loads(configuration.read_text(encoding='utf-8')) if configuration.exists() else {}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
    except ImportError as exc:
        raise RuntimeError('matplotlib and numpy are needed for figures') from exc
    counts=Counter(r.get('status','UNKNOWN') for r in records)
    # The sweep's compact summary intentionally omits most physical safety
    # metrics. Rejoin them from original per-run metrics without inferring any
    # success label or dropping failed runs.
    detail=[]
    for r in records:
        row=dict(r)
        metrics_path=root/r.get('name','__not_a_run__')/'metrics.json'
        if metrics_path.exists():
            m=json.loads(metrics_path.read_text(encoding='utf-8'))
            for field in ('max_push_heading_error_deg','max_interaction_force_N',
                          'integrated_abs_thrust_Ns','max_substep_close_speed_mps',
                          'max_logged_yaw_rate_deg_s','cpu_elapsed_seconds',
                          'path_length_ratio','solver_success_rate'):
                if field in m:
                    row[field]=m[field]
        detail.append(row)
    if detail:
        keys=list(dict.fromkeys(k for row in detail for k in row))
        with (root/'summary_details.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader();writer.writerows(detail)
    fig,ax=plt.subplots(figsize=(7,4))
    keys=['SUCCESS','TIMEOUT','UNSAFE','RUN_ERROR']
    ax.bar(keys,[counts.get(x,0) for x in keys],color=['#238264','#C49E43','#B9575C','#8B8B8B'])
    ax.set_ylabel('Number of completed cases')
    ax.set_title('Environmental validation: strict berthing outcome')
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(root/'status_counts.png',dpi=160);plt.close(fig)
    groups={}
    for r in records:
        if float(r.get('wind_mps',0))>0 and float(r.get('current_mps',0))>0:
            key=(float(r.get('psi0_deg',r['psi0'])),float(r['wind_mps']),float(r['current_mps']))
            groups.setdefault(key,[]).append(r)
    for (psi,w,c),rows in groups.items():
        xs=sorted(set(map(float,opts.get('wind_angles',[]))) | {float(r['wind_to_deg']) for r in rows})
        ys=sorted(set(map(float,opts.get('current_angles',[]))) | {float(r['current_to_deg']) for r in rows})
        if not xs or not ys:continue
        values=np.full((len(ys),len(xs)),0,dtype=int)  # 0 = NOT TESTED
        for r in rows:
            i=ys.index(float(r['current_to_deg']));j=xs.index(float(r['wind_to_deg']))
            values[i,j]={'SUCCESS':3,'TIMEOUT':2,'UNSAFE':1,'RUN_ERROR':4}.get(r['status'],4)
        cmap=ListedColormap(['#D0D1D5','#CF5D56','#E5B756','#59A784','#8B8B8B'])
        fig,ax=plt.subplots(figsize=(max(7,1.35*len(xs)),max(3.8,.95*len(ys))))
        ax.imshow(values,cmap=cmap,vmin=-0.5,vmax=4.5,aspect='auto')
        ax.set_xticks(range(len(xs)),[f'{x:g}°' for x in xs]);ax.set_yticks(range(len(ys)),[f'{y:g}°' for y in ys])
        ax.set_xlabel('Wind toward (earth frame)');ax.set_ylabel('Current toward (earth frame)')
        ax.set_title(f'psi0 {psi:g}°  |  wind {w:g} m/s  |  current {c:g} m/s')
        labels=['NOT TESTED','UNSAFE','TIMEOUT','SUCCESS','RUN ERROR']
        for i in range(len(ys)):
            for j in range(len(xs)):
                z=values[i,j]
                ax.text(j,i,labels[z],ha='center',va='center',fontsize=7.5,color='#1B2630')
        fig.tight_layout()
        fig.savefig(root/f'angle_matrix_psi{psi:g}_w{w:g}_c{c:g}.png',dpi=160)
        plt.close(fig)
        with (root/f'angle_matrix_psi{psi:g}_w{w:g}_c{c:g}.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['current_toward_deg / wind_toward_deg',*xs])
            for i,a in enumerate(ys):writer.writerow([a,*[labels[v] for v in values[i]]])
    finished=len(records)
    print(f'Completed: {finished} | Success: {counts.get("SUCCESS",0)} | Timeout: {counts.get("TIMEOUT",0)} | Unsafe: {counts.get("UNSAFE",0)} | Errors: {counts.get("RUN_ERROR",0)}')
    print(f'Figures written to {root}')
    return counts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results/environment_direction_sweep')
    args=parser.parse_args()
    report(Path(args.results))
