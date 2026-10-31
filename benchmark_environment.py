"""Reproducible environmental NMPC benchmark. No fabricated cases or successes.

Usage:
  python benchmark_environment.py --suite quick --out results/environment_quick
  python benchmark_environment.py --suite paper --out results/environment_paper
  python benchmark_environment.py --report-only --out results/environment_paper

Every case has its own actual simulator trajectory and a row in the report.
Modes: none = no disturbance observer/feedforward; residual = own ship-state
residual observer; measured = own tug's local environmental sensor + residual.
"""
from __future__ import annotations
import argparse,csv,json
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
from run_nmpc import rollout, CONTROLLER_VERSION, DEFAULT_EMERGENCY_SIM_SECONDS
from environmental_report import generate_report

@dataclass(frozen=True)
class Scenario:
    name:str
    wind_mps:float=0.
    wind_to_deg:float=-90.
    current_mps:float=0.
    current_to_deg:float=-90.
    gust_mps:float=0.
    gust_period_s:float=360.
    current_osc_mps:float=0.

SCENARIOS={
    'calm':Scenario('calm'),
    'wind6':Scenario('wind6',wind_mps=6.),
    'wind10':Scenario('wind10',wind_mps=10.),
    'current02':Scenario('current02',current_mps=.2),
    'current05':Scenario('current05',current_mps=.5),
    'combined':Scenario('combined',wind_mps=8.,current_mps=.3),
    'wind6current02':Scenario('wind6_current02',wind_mps=6.,current_mps=.2),
    'gust':Scenario('gust',wind_mps=7.,gust_mps=2.,current_mps=.2,current_osc_mps=.1),
    'offshore':Scenario('offshore',wind_mps=8.,wind_to_deg=90.,current_mps=.3,current_to_deg=90.),
}
SUITES={
    'quick':(['calm','wind6','current02','combined'],[20.],['residual','measured']),
    'paper':(['calm','wind6','wind10','current02','current05','combined','gust','offshore'],[20.,40.],['none','residual','measured']),
}

def make_row(case,psi,mode,seed,out,metrics):
    result={'scenario':case.name,'psi0_deg':psi,'mode':mode,'seed':seed,'case_dir':str(out),
            'status':metrics['status'],'success':int(metrics['success']),
            'unsafe':int(metrics['unsafe']),'time_s':metrics['seconds'],
            'position_error_m':metrics['position_error_m'],'heading_error_deg':metrics['heading_error_deg'],
            'wind_mps':case.wind_mps,'wind_to_deg':case.wind_to_deg,
            'current_mps':case.current_mps,'current_to_deg':case.current_to_deg,
            'gust_mps':case.gust_mps,'current_osc_mps':case.current_osc_mps,
            'min_logged_hull_gap_m':metrics.get('min_logged_hull_gap_m'),
            'max_close_speed_mps':metrics.get('max_logged_close_speed_mps'),
            'integrated_abs_thrust_Ns':metrics.get('integrated_abs_thrust_Ns'),
            'mean_total_abs_thrust_N':metrics.get('mean_total_abs_thrust_N',
                metrics.get('integrated_abs_thrust_Ns',0.)/max(metrics['seconds'],1.)),
            'solver_success_rate':metrics.get('solver_success_rate'),
            'avg_optimizer_ms':metrics.get('avg_optimizer_ms'),
            'cpu_elapsed_seconds':metrics['cpu_elapsed_seconds']}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['quick','paper'],default='quick')
    p.add_argument('--out',default='results/nmpc_environment')
    p.add_argument('--scenarios',nargs='+',choices=list(SCENARIOS))
    p.add_argument('--headings',nargs='+',type=float)
    p.add_argument('--modes',nargs='+',choices=['none','residual','measured'])
    p.add_argument('--seeds',nargs='+',type=int,default=[123])
    p.add_argument('--max-seconds',type=float,default=DEFAULT_EMERGENCY_SIM_SECONDS,
                   help='Common emergency simulated-time cap per case. Longer runs take substantially more CPU time.')
    p.add_argument('--maxiter',type=int,default=10)
    p.add_argument('--position-tolerance-m',type=float,default=2.5)
    p.add_argument('--wind-sensor-std-mps',type=float,default=0.)
    p.add_argument('--current-sensor-std-mps',type=float,default=0.)
    p.add_argument('--max-fraction-step',type=float,default=.38)
    p.add_argument('--max-offset-step-deg',type=float,default=24.)
    p.add_argument('--slew-weight',type=float,default=.10)
    p.add_argument('--render-each',action='store_true')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--rerun',action='store_true')
    args=p.parse_args()
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        generate_report(dest)
        return
    cases,hds,modes=SUITES[args.suite]
    cases=args.scenarios or cases
    hds=args.headings or hds
    modes=args.modes or modes
    experiment={'suite':args.suite,'scenarios':cases,'headings':hds,'modes':modes,
                'seeds':args.seeds,'max_seconds':args.max_seconds,'maxiter':args.maxiter,
                'position_tolerance_m':args.position_tolerance_m,
                'wind_sensor_std_mps':args.wind_sensor_std_mps,
                'current_sensor_std_mps':args.current_sensor_std_mps,
                'max_fraction_step':args.max_fraction_step,
                'max_offset_step_deg':args.max_offset_step_deg,
                'slew_weight':args.slew_weight}
    (dest/'experiment.json').write_text(json.dumps(experiment,indent=2),encoding='utf8')
    entries=[]
    for seed in args.seeds:
        for name in cases:
            case=SCENARIOS[name]
            for hd in hds:
                for mode in modes:
                    folder=dest/f'{case.name}_h{hd:g}_{mode}_seed{seed}'
                    metricsfile=folder/'metrics.json'
                    # Only reuse results generated with the exact active
                    # scenario/sensor/tolerance and compute budget.
                    configfile=folder/'configuration.json'
                    expected={'controller_version':CONTROLLER_VERSION,'psi0':hd,'max_seconds':args.max_seconds,'seed':seed,
                        'max_iterations':args.maxiter,'wind_mps':case.wind_mps,
                        'wind_to_deg':case.wind_to_deg,'current_mps':case.current_mps,
                        'current_to_deg':case.current_to_deg,'gust_mps':case.gust_mps,
                        'gust_period_s':case.gust_period_s,
                        'current_osc_mps':case.current_osc_mps,'disturbance_mode':mode,
                        'wind_sensor_std_mps':args.wind_sensor_std_mps,
                        'current_sensor_std_mps':args.current_sensor_std_mps,
                        'position_tolerance_m':args.position_tolerance_m,
                        'max_fraction_step':args.max_fraction_step,
                        'max_offset_step_deg':args.max_offset_step_deg,
                        'slew_weight':args.slew_weight}
                    reuse=False
                    if not args.rerun and metricsfile.exists() and configfile.exists() and (folder/'trajectory.csv').exists():
                        old=json.loads(configfile.read_text(encoding='utf8'))
                        reuse=all(old.get(k)==v for k,v in expected.items())
                    if reuse:
                        print('REUSE',folder,flush=True)
                        metrics=json.loads(metricsfile.read_text(encoding='utf8'))
                    else:
                        print('RUN',folder,flush=True)
                        metrics=rollout(out=str(folder),psi0=hd,max_seconds=args.max_seconds,
                            seed=seed,maxiter=args.maxiter,render=args.render_each,verbose=False,
                            wind_mps=case.wind_mps,wind_to_deg=case.wind_to_deg,
                            current_mps=case.current_mps,current_to_deg=case.current_to_deg,
                            gust_mps=case.gust_mps,gust_period_s=case.gust_period_s,
                            current_osc_mps=case.current_osc_mps,
                            disturbance_mode=mode,
                            wind_sensor_std_mps=args.wind_sensor_std_mps,
                            current_sensor_std_mps=args.current_sensor_std_mps,
                            position_tolerance_m=args.position_tolerance_m,
                            max_fraction_step=args.max_fraction_step,
                            max_offset_step_deg=args.max_offset_step_deg,
                            slew_weight=args.slew_weight)
                    entries.append(make_row(case,hd,mode,seed,folder,metrics))
                    print('RESULT',case.name,hd,mode,metrics['status'],
                          round(metrics['position_error_m'],2),flush=True)
                    with (dest/'comparison.csv').open('w',newline='',encoding='utf8') as f:
                        w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
    generate_report(dest)

if __name__=='__main__':main()
