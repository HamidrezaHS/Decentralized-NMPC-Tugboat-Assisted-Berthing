"""Ocean Engineering reproducible apples-to-apples berthing comparison.

Methods: unchanged Dec-NMPC, central joint NMPC, consensus ADMM-NMPC.
All algorithms use the SAME physical coupled ship/tug plant, berth target,
force limits, safety monitor and environmental forcing for a given scenario.

Use --smoke for software validation. These runs are NOT paper successes.
Use --suite core for four environmental cases x two headings x three methods.
Use --suite extended for a more severe and opposite-direction study.
Run --report-only to rebuild the two main figures and two summary tables.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_nmpc import rollout

METHODS=('decentralized','central','admm')
CASES={
    'calm':dict(wind_mps=0.,wind_to_deg=-90.,current_mps=0.,current_to_deg=-90.),
    'wind6':dict(wind_mps=6.,wind_to_deg=-90.,current_mps=0.,current_to_deg=-90.),
    'current02':dict(wind_mps=0.,wind_to_deg=-90.,current_mps=.2,current_to_deg=-90.),
    'combined6_02':dict(wind_mps=6.,wind_to_deg=-90.,current_mps=.2,current_to_deg=-90.),
    'literature4_05':dict(wind_mps=4.,wind_to_deg=-90.,current_mps=.5,current_to_deg=-90.),
    'harsh10_05':dict(wind_mps=10.,wind_to_deg=-90.,current_mps=.5,current_to_deg=-90.),
    'offshore8_03':dict(wind_mps=8.,wind_to_deg=90.,current_mps=.3,current_to_deg=90.),
    'gust7_02':dict(wind_mps=7.,wind_to_deg=-90.,current_mps=.2,current_to_deg=-90.,
                    gust_mps=2.,current_osc_mps=.1,gust_period_s=360.),
}
SUITES={
    'smoke':(['calm'],[20.]),
    'core':(['calm','wind6','current02','combined6_02'],[20.,40.]),
    'extended':(['calm','wind6','current02','combined6_02',
                 'literature4_05','harsh10_05','offshore8_03','gust7_02'],[20.,40.]),
}
LABELS={'decentralized':'Dec-NMPC (no peer communication)',
        'central':'Centralized joint NMPC',
        'admm':'Consensus ADMM-NMPC (exchanging plans)'}
FIELDS=['case','psi0_deg','seed','method','status','success','unsafe',
        'time_s','time_to_success_s','position_error_m','heading_error_deg',
        'min_hull_gap_m','max_close_speed_mps','peak_interaction_N',
        'integrated_abs_thrust_Ns','integrated_thrust_change_N',
        'avg_optimizer_ms','mean_decision_wall_ms','cpu_elapsed_seconds','communication_bytes',
        'max_push_heading_error_deg','push_guard_interventions',
        'path_length_m','path_length_ratio','max_along_target_overshoot_m','case_path']


def _row(name,psi,seed,method,folder,m):
    return dict(case=name,psi0_deg=psi,seed=seed,method=method,
        status=m['status'],success=int(m['success']),unsafe=int(m['unsafe']),
        time_s=m['seconds'],time_to_success_s=(m['seconds'] if m['success'] else ''),
        position_error_m=m['position_error_m'],heading_error_deg=m['heading_error_deg'],
        min_hull_gap_m=m.get('min_logged_hull_gap_m'),
        max_close_speed_mps=m.get('max_substep_close_speed_mps'),
        peak_interaction_N=m.get('max_interaction_force_N'),
        integrated_abs_thrust_Ns=m.get('integrated_abs_thrust_Ns'),
        integrated_thrust_change_N=m.get('integrated_abs_thrust_change_N'),
        avg_optimizer_ms=m.get('avg_optimizer_ms'),
        mean_decision_wall_ms=m.get('mean_decision_wall_ms',
                                   4.*m.get('avg_optimizer_ms',0.)),
        cpu_elapsed_seconds=m.get('cpu_elapsed_seconds'),
        communication_bytes=m.get('coordination_communication_bytes',0),
        max_push_heading_error_deg=(max(m.get('max_push_heading_error_deg') or [float('nan')])),
        push_guard_interventions=sum(m.get('push_heading_guard_interventions') or [0]),
        path_length_m=m.get('path_length_m'),path_length_ratio=m.get('path_length_ratio'),
        max_along_target_overshoot_m=m.get('max_along_target_overshoot_m'),
        case_path=str(folder))


def _write_csv(path,rows,headers):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers)
        writer.writeheader();writer.writerows(rows)


def load_results(root):
    entries=[]
    for folder in sorted(root.glob('case_*')):
        mf=folder/'metrics.json';cf=folder/'configuration.json';tf=folder/'trajectory.csv'
        if not (mf.exists() and cf.exists() and tf.exists()):continue
        m=json.loads(mf.read_text(encoding='utf-8'))
        c=json.loads(cf.read_text(encoding='utf-8'))
        case=c.get('benchmark_case')
        if not case:
            # Folders are named by the runner; fail closed if metadata missing.
            continue
        entries.append(_row(case,float(c['psi0']),int(c['seed']),c['controller_method'],folder,m))
    return entries


def _figure_trajectories(rows,root,case='combined6_02',psi0=20.):
    selected=[r for r in rows if r['case']==case and float(r['psi0_deg'])==float(psi0)]
    fig,ax=plt.subplots(figsize=(7.0,5.8),constrained_layout=True)
    colors={'decentralized':'#126d89','central':'#b66e18','admm':'#7a5795'}
    for r in selected:
        with (Path(r['case_path'])/'trajectory.csv').open(encoding='utf-8',newline='') as f:
            traj=list(csv.DictReader(f))
        xx=np.asarray([float(v['x_m']) for v in traj]);yy=np.asarray([float(v['y_m']) for v in traj])
        ax.plot(xx,yy,label=f"{LABELS[r['method']]} ({r['status']})",lw=1.8,color=colors[r['method']])
        ax.scatter(xx[-1],yy[-1],color=colors[r['method']],s=30)
    if selected:
        ax.scatter([50],[-100],marker='*',s=160,c='black',label='Berth target')
        ax.scatter([0],[0],marker='o',facecolors='none',edgecolors='black',s=60,label='Initial ship center')
    ax.set(xlabel='East / x (m)',ylabel='North / y (m)',title=f'Ship-center paths: {case}, initial heading {psi0:g}°')
    ax.grid(alpha=.20);ax.axis('equal')
    if selected:ax.legend(fontsize=7.6)
    for fmt in ('png','pdf'):fig.savefig(root/f'Figure_1_Trajectories.{fmt}',dpi=190)
    plt.close(fig)


def _figure_tracking(rows,root,case='combined6_02',psi0=20.):
    selected=[r for r in rows if r['case']==case and float(r['psi0_deg'])==float(psi0)]
    fig,axs=plt.subplots(2,1,figsize=(7.5,6.1),sharex=True,constrained_layout=True)
    for r in selected:
        with (Path(r['case_path'])/'trajectory.csv').open(encoding='utf-8',newline='') as f:
            traj=list(csv.DictReader(f))
        t=np.asarray([float(q['time_s'])/60 for q in traj])
        ep=np.asarray([float(q['position_error_m']) for q in traj])
        vn=np.asarray([float(q.get('north_speed_mps') or 0.) for q in traj])
        lbl=LABELS[r['method']]
        axs[0].plot(t,ep,label=lbl,lw=1.7)
        axs[1].plot(t,vn,lw=1.7)
    axs[0].axhline(2.5,ls=':',color='#777777',lw=.9,label='Berthing tolerance')
    axs[1].axhline(-.045,ls=':',color='#777777',lw=.9,label='Nominal approach-speed threshold')
    axs[0].set_ylabel('Position error (m)')
    axs[1].set(ylabel='Earth-fixed North velocity (m/s)',xlabel='Simulation time (min)')
    if selected:axs[0].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.suptitle(f'Comparative response: {case}, ψ0={psi0:g}°')
    for fmt in ('png','pdf'):fig.savefig(root/f'Figure_2_Tracking.{fmt}',dpi=190)
    plt.close(fig)


def report(root,representative='combined6_02',heading=20.):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rows=load_results(root)
    _write_csv(root/'Table_1_All_Cases.csv',rows,FIELDS)
    bymethod=defaultdict(list)
    for r in rows:bymethod[r['method']].append(r)
    agg=[]
    for method in METHODS:
        group=bymethod[method]
        if not group:continue
        successes=[r for r in group if r['success']]
        agg.append(dict(method=method,n=len(group),successful=len(successes),
            unsafe=sum(r['unsafe'] for r in group),timeouts=sum(r['status']=='TIMEOUT' for r in group),
            success_rate=sum(r['success'] for r in group)/len(group),
            mean_time_to_success_s=(float(np.mean([float(r['time_s']) for r in successes])) if successes else ''),
            median_terminal_position_error_m=float(np.median([float(r['position_error_m']) for r in group])),
            mean_peak_interaction_N=float(np.mean([float(r['peak_interaction_N']) for r in group])),
            mean_total_thrust_Ns=float(np.mean([float(r['integrated_abs_thrust_Ns']) for r in group])),
            mean_max_longitudinal_overshoot_m=float(np.mean([float(r['max_along_target_overshoot_m']) for r in group if r['max_along_target_overshoot_m'] is not None])) if any(r['max_along_target_overshoot_m'] is not None for r in group) else None,
            mean_decision_wall_ms=float(np.mean([float(r['mean_decision_wall_ms']) for r in group])),
            mean_max_push_heading_error_deg=float(np.mean([float(r['max_push_heading_error_deg']) for r in group])),
            total_communication_bytes=sum(int(r['communication_bytes']) for r in group)))
    if agg:_write_csv(root/'Table_2_Method_Aggregate.csv',agg,list(agg[0]))
    representative_rows=[r for r in rows if r['case']==representative and float(r['psi0_deg'])==float(heading)]
    actual_case,actual_heading=(representative,heading)
    if not representative_rows and rows:
        actual_case,actual_heading=rows[0]['case'],float(rows[0]['psi0_deg'])
    _figure_trajectories(rows,root,actual_case,actual_heading)
    _figure_tracking(rows,root,actual_case,actual_heading)
    with (root/'REPORT_STATUS.txt').open('w',encoding='utf8') as f:
        f.write('Runs with timeout are NOT successes. Empty mean_time_to_success means no success.\n')
        f.write('These are numerical experiments, not reproductions of Chen/Huynh/Du.\n')
        f.write(f'Cases collected: {len(rows)}. Representative case={actual_case}, psi0={actual_heading}\n')
        for r in rows:
            f.write(f"{r['case']:18s} {r['method']:14s} ψ0={r['psi0_deg']:g} {r['status']:8s} "
                    f"err={float(r['position_error_m']):.3f}m t={float(r['time_s']):.1f}s\n")
    print(f'OE report generated using {len(rows)} completed runs in {root}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=list(SUITES),default='core')
    p.add_argument('--out',default='results/ocean_engineering_benchmark')
    p.add_argument('--headings',type=float,nargs='+')
    p.add_argument('--cases',choices=list(CASES),nargs='+')
    p.add_argument('--methods',choices=METHODS,nargs='+',default=list(METHODS))
    p.add_argument('--seeds',type=int,nargs='+',default=[123])
    p.add_argument('--guidance-mode',choices=['mission','legacy'],default='mission')
    p.add_argument('--max-seconds',type=float,default=21600.,help='Emergency guard; success and unsafe stop naturally.')
    p.add_argument('--maxiter',type=int,default=10)
    p.add_argument('--admm-rounds',type=int,default=3)
    p.add_argument('--admm-inner-iterations',type=int,default=4)
    p.add_argument('--admm-rho',type=float,default=.5)
    p.add_argument('--disable-push-heading-guard',action='store_true',
                   help='For paired ablation: turn off local fast safety layer for all methods.')
    p.add_argument('--render-each',action='store_true')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--rerun',action='store_true')
    p.add_argument('--representative',default='combined6_02')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.report_only:
        report(out,representative=a.representative);return
    scenarios,headings=SUITES[a.suite]
    scenarios=a.cases or scenarios; headings=a.headings or headings
    guard_enabled=not a.disable_push_heading_guard
    experiment=dict(push_heading_guard=guard_enabled,guidance_mode=a.guidance_mode,suite=a.suite,scenarios=scenarios,headings=headings,methods=a.methods,
                    seeds=a.seeds,max_seconds=a.max_seconds,maxiter=a.maxiter,
                    admm_rounds=a.admm_rounds,admm_inner_iterations=a.admm_inner_iterations,admm_rho=a.admm_rho,
                    physical_plant='identical coupled 5-vessel',condition='success/unsafe/emergency guard')
    (out/'experiment.json').write_text(json.dumps(experiment,indent=2),encoding='utf8')
    for scenario in scenarios:
        for psi in headings:
            for seed in a.seeds:
                for method in a.methods:
                    name=f'case_{scenario}_h{psi:g}_s{seed}_{method}'
                    folder=out/name
                    cfgfile=folder/'configuration.json'
                    expected=dict(psi0=psi,seed=seed,max_seconds=a.max_seconds,
                        controller_method=method,push_heading_guard=guard_enabled,max_iterations=a.maxiter,
                        guidance_mode=a.guidance_mode,
                        precision=True,controller_version='oe-v4-controller-audited',
                        admm_rounds=a.admm_rounds,admm_rho=a.admm_rho,
                        admm_inner_iterations=a.admm_inner_iterations,**CASES[scenario])
                    reuse=False
                    if not a.rerun and cfgfile.exists() and (folder/'metrics.json').exists() and (folder/'trajectory.csv').exists():
                        old=json.loads(cfgfile.read_text(encoding='utf8'))
                        reuse=(all(old.get(k)==v for k,v in expected.items()) and
                               json.loads((folder/'metrics.json').read_text(encoding='utf8')).get('status')
                               in ('SUCCESS','UNSAFE','TIMEOUT'))
                    if reuse:
                        print('REUSE',name,flush=True)
                        continue
                    print('RUN',name,flush=True)
                    m=rollout(out=str(folder),psi0=psi,seed=seed,
                        max_seconds=a.max_seconds,maxiter=a.maxiter,render=a.render_each,
                        verbose=False,controller_method=method,push_heading_guard=guard_enabled,
                        guidance_mode=a.guidance_mode,
                        admm_rounds=a.admm_rounds,admm_inner_iterations=a.admm_inner_iterations,
                        admm_rho=a.admm_rho,**CASES[scenario])
                    cfg=json.loads(cfgfile.read_text(encoding='utf8'))
                    cfg['benchmark_case']=scenario
                    cfgfile.write_text(json.dumps(cfg,indent=2),encoding='utf8')
                    print('END',name,m['status'],f"pos_error={m['position_error_m']:.3f}m",flush=True)
                    # Avoid expensive figure regeneration for each long rollout.
                    # A terminated benchmark can still be summarized using --report-only.
    report(out,representative=a.representative)


if __name__=='__main__':main()
