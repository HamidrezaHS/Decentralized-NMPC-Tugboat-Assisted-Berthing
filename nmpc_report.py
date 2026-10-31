"""Create paper-ready diagnostics from a completed independent NMPC rollout.

Only uses the logged measured simulator trajectory and optimizer diagnostics;
no smoothed or fabricated trajectories are substituted.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path,encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError(f'Empty CSV: {path}')
    return rows


def field(rows,key,default=0.):
    return np.asarray([float(q.get(key) or default) for q in rows])


def export(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=185,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def report(folder):
    root=Path(folder)
    rows=load_csv(root/'trajectory.csv')
    decisions=load_csv(root/'local_nmpc_decisions.csv')
    config=json.loads((root/'configuration.json').read_text(encoding='utf-8'))
    metrics=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    t=field(rows,'time_s')/60
    fig,axs=plt.subplots(3,2,figsize=(12.5,10),constrained_layout=True)
    ax=axs[0,0]
    ax.plot(t,field(rows,'x_m'),label='X ship (m)')
    ax.plot(t,field(rows,'y_m'),label='Y ship (m)')
    ax.axhline(config['target_x'],ls='--',color='gray',lw=1,label='X target')
    ax.axhline(config['target_y'],ls=':',color='gray',lw=1,label='Y target')
    ax.set_ylabel('Position (m)');ax.legend(fontsize=8,ncol=2)
    axs[0,1].plot(t,field(rows,'position_error_m'))
    axs[0,1].axhline(2.5,ls='--',color='gray',lw=1)
    axs[0,1].set_ylabel('Position error (m)')
    axs[1,0].plot(t,field(rows,'heading_error_deg'))
    axs[1,0].axhline(2.,ls='--',color='gray',lw=1)
    axs[1,0].set_ylabel('Abs heading error (deg)')
    axs[1,1].plot(t,field(rows,'u_mps'),label='u')
    axs[1,1].plot(t,field(rows,'v_mps'),label='v')
    axs[1,1].set_ylabel('Ship body velocity (m/s)');axs[1,1].legend()
    axs[2,0].plot(t,field(rows,'r_deg_s'))
    axs[2,0].axhline(.035,ls='--',color='gray',lw=1)
    axs[2,0].axhline(-.035,ls='--',color='gray',lw=1)
    axs[2,0].set_ylabel('Yaw rate (deg/s)')
    axs[2,1].plot(t,field(rows,'hull_gap_m'))
    axs[2,1].axhline(0.,ls='--',color='gray',lw=1)
    axs[2,1].set_ylabel('Hull–quay clearance (m)')
    for ax in axs.flat:ax.set_xlabel('Time (min)');ax.grid(alpha=.2)
    fig.suptitle('Decentralized NMPC: 3-DOF vessel tracking and safety')
    export(fig,root/'nmpc_tracking')

    fig,axs=plt.subplots(3,1,figsize=(12,9),constrained_layout=True,sharex=True)
    palette=['#0891b2','#2563eb','#d97706','#c026d3']
    for i,c in enumerate(palette,1):
        axs[0].plot(t,field(rows,f'tug{i}_fraction'),label=f'T{i}',color=c)
        axs[1].plot(t,field(rows,f'tug{i}_thrust_N')/1e6,label=f'T{i}',color=c)
        axs[2].plot(t,field(rows,f'tug{i}_offset_deg'),label=f'T{i}',color=c)
    axs[0].set_ylabel('Command thrust fraction')
    axs[1].set_ylabel('Actual signed thrust (MN)')
    axs[2].set_ylabel('Local target offset (deg)')
    for ax in axs:ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8)
    axs[-1].set_xlabel('Time (min)')
    fig.suptitle('Independent NMPC actions and measured physical actuator responses')
    export(fig,root/'nmpc_commands')

    fig,axs=plt.subplots(2,1,figsize=(12,7),constrained_layout=True,sharex=True)
    for i,c in enumerate(palette,1):
        r=[rr for rr in decisions if int(rr['agent'])==i]
        axs[0].plot(field(r,'observation_time_s')/60,field(r,'objective'),label=f'T{i}',color=c)
        axs[1].plot(field(r,'observation_time_s')/60,field(r,'iterations'),label=f'T{i}',color=c)
    axs[0].set_ylabel('Local finite-horizon objective')
    axs[1].set_ylabel('L-BFGS-B iterations')
    axs[1].set_xlabel('Time (min)')
    for ax in axs:ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8)
    fig.suptitle('Decentralized nonlinear optimization diagnostics')
    export(fig,root/'nmpc_solver')

    # Environment and precision-specific plots preserve the logged actual values.
    fig,axs=plt.subplots(3,1,figsize=(11.5,9),sharex=True,constrained_layout=True)
    axs[0].plot(t,field(rows,'wind_mps'),label='Wind to heading in configuration')
    axs[0].set_ylabel('Wind speed (m/s)')
    axs[1].plot(t,field(rows,'current_mps'),label='Current to heading in configuration')
    axs[1].set_ylabel('Current speed (m/s)')
    if 'approach_speed_mps' in rows[0]:
        closing=field(rows,'approach_speed_mps')
    else:
        psi=np.deg2rad(field(rows,'psi_deg'))
        closing=np.maximum(0.,-(np.sin(psi)*field(rows,'u_mps')+
                                np.cos(psi)*field(rows,'v_mps')))
    axs[2].plot(t,closing,label='Quay approach speed')
    axs[2].axhline(.045,color='tab:red',ls='--',lw=1,label='Terminal lateral threshold')
    axs[2].set_ylabel('Approach speed (m/s)')
    for ax in axs:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axs[-1].set_xlabel('Time (min)')
    fig.suptitle('Environmental forcing and hull approach speed (logged simulation)')
    export(fig,root/'nmpc_environment')

    # Compare the actual final trajectory with the strict target tolerances.
    fig,axs=plt.subplots(2,2,figsize=(11.5,7.8),constrained_layout=True)
    tail_start=max(0.,t[-1]-12.)
    mask=t>=tail_start
    axs[0,0].plot(t[mask],field(rows,'position_error_m')[mask]);axs[0,0].axhline(2.5,ls='--',lw=1,color='gray')
    axs[0,0].axhline(1.0,ls=':',lw=1,color='tab:orange')
    axs[0,0].set_ylabel('Position error (m)')
    axs[0,1].plot(t[mask],field(rows,'heading_error_deg')[mask]);axs[0,1].axhline(2.,ls='--',lw=1,color='gray')
    axs[0,1].set_ylabel('Heading error (deg)')
    speed=(field(rows,'ship_speed_mps') if 'ship_speed_mps' in rows[0] else
           np.hypot(field(rows,'u_mps'),field(rows,'v_mps')))
    axs[1,0].plot(t[mask],speed[mask]);axs[1,0].axhline(.045,ls='--',lw=1,color='gray')
    axs[1,0].set_ylabel('Ship speed (m/s)')
    axs[1,1].plot(t[mask],np.abs(field(rows,'r_deg_s')[mask]));axs[1,1].axhline(.035,ls='--',lw=1,color='gray')
    axs[1,1].set_ylabel('Abs yaw rate (deg/s)')
    for ax in axs.flat:ax.set_xlabel('Time (min)');ax.grid(alpha=.25)
    fig.suptitle('Terminal convergence – observed, not extrapolated')
    export(fig,root/'nmpc_terminal_zoom')

    with (root/'nmpc_kpi.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','value'])
        for k,v in metrics.items():w.writerow([k,v])
    return root


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/nmpc_h20')
    args=p.parse_args();root=report(args.input)
    print('Created nmpc_tracking, nmpc_commands, nmpc_solver in PNG/PDF and nmpc_kpi.csv:',root)

if __name__=='__main__':main()
