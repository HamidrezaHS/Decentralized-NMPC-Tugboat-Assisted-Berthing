"""Export publication-oriented MARL plots.

Outputs:
- trajectory.png : ship-centre path with sampled ship/tug footprints
- position_error.png / heading_error.png / hull_gap.png
- contact_forces.png / signed_thrust.png
- berthing_final.png : final physical plan view
- berthing_four_stages.png : 2x2 staged figure similar to the main project renderer

For the staged visuals, the ship and tug outlines are loaded from the DXF
geometry files already included in the project.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from geom.constraints import Constraints
from model.ship import Ship
from model.tug import Tug
from physical_v4.plant import UnilateralPlant, ContactConfig, rot
from decentralized_game.tow_safety import free_side_from_quay
from visualization.marine_renderer import scene, get_hull

TUG_COLORS=['tab:orange','tab:green','tab:red','tab:purple']


def _read_rows(path: str | Path):
    with open(path,newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty trajectory')
    return rows


def _f(row,key,default=0.0):
    value=row.get(key,default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _values(rows,key):
    return np.array([_f(r,key) for r in rows],dtype=float)


def _transform(points,x,y,heading_deg):
    ang=np.deg2rad(float(heading_deg))
    c,s=np.cos(ang),np.sin(ang)
    pts=np.asarray(points,float)
    return pts @ np.array([[c,s],[-s,c]]) + np.array([x,y])


def _sample_indices(count,max_samples):
    max_samples=max(1,min(max_samples,count))
    return np.unique(np.linspace(0,count-1,max_samples,dtype=int))


def _draw_orientation_arrow(ax,x,y,heading_deg,length,color,alpha=.9,z=10):
    ang=np.deg2rad(float(heading_deg))
    dx,dy=length*np.cos(ang),length*np.sin(ang)
    ax.annotate('',xy=(x+dx,y+dy),xytext=(x,y),
                arrowprops=dict(arrowstyle='-|>',lw=1.2,color=color,alpha=alpha),
                zorder=z)


def _load_tug_outline():
    root=Path(__file__).resolve().parent
    try:
        return np.asarray(Constraints(str(root/'geom'/'waterline_tug.dxf')).points,float)
    except Exception:
        L=22.943/2;B=8.5/2
        return np.array([
            [-L,-0.65*B],[-0.86*L,-B],[0.55*L,-B],[L,-0.50*B],[L,0.0],
            [L,0.50*B],[0.55*L,B],[-0.86*L,B],[-L,0.65*B]
        ])


def _cfg_from_args_and_rows(args, rows):
    return {
        'target_x': float(args.target_x),
        'target_y': float(args.target_y),
        'target_psi': float(args.target_psi_deg),
        'x0': _f(rows[0],'x_m'),
        'y0': _f(rows[0],'y_m'),
    }


def _make_plant(target_psi_deg: float):
    root=Path(__file__).resolve().parent
    hull=Constraints(str(root/'geom'/'waterline_ship.dxf'))
    side=free_side_from_quay(np.deg2rad(target_psi_deg), np.array([0.,-1.]))
    tug=Tug()
    tug.bollard_pull_N=100.*9806.65
    tug.command_force_limit_N=tug.bollard_pull_N*.9
    plant=UnilateralPlant(Ship(), tug, hull, side, ContactConfig())
    return plant, side


def _convert_rows_for_renderer(raw_rows, cfg):
    plant,_side=_make_plant(cfg['target_psi'])
    out=[]
    for rr in raw_rows:
        ship=np.array([_f(rr,'x_m'),_f(rr,'y_m'),np.deg2rad(_f(rr,'psi_deg')),0.,0.,0.],float)
        attachments=plant.attachment(ship)
        row={
            't': _f(rr,'time_s'),
            'x': ship[0],
            'y': ship[1],
            'psi_deg': _f(rr,'psi_deg'),
            'ue': _f(rr,'u_mps'),
            've': _f(rr,'v_mps'),
            'fender_gap_m': _f(rr,'hull_gap_m'),
            'approach_speed_mps': _f(rr,'approach_speed_mps'),
            'wind_mps': _f(rr,'wind_mps'),
            'wind_to_deg': _f(rr,'wind_to_deg'),
            'current_mps': _f(rr,'current_mps'),
            'current_to_deg': _f(rr,'current_to_deg'),
        }
        for i in range(1,5):
            tx=_f(rr,f'tug{i}_x_m');ty=_f(rr,f'tug{i}_y_m');tpsi=_f(rr,f'tug{i}_psi_deg')
            arm=np.array([plant.ap if i<3 else plant.at,0.0])
            tug_anchor=np.array([tx,ty])+rot(np.deg2rad(tpsi))@arm
            contact=attachments[i-1]
            row.update({
                f'tug{i}_x_m': tx,
                f'tug{i}_y_m': ty,
                f'tug{i}_psi_deg': tpsi,
                f'tug{i}_contact_x_m': float(contact[0]),
                f'tug{i}_contact_y_m': float(contact[1]),
                f'tug{i}_rope_anchor_x_m': float(tug_anchor[0]),
                f'tug{i}_rope_anchor_y_m': float(tug_anchor[1]),
                f'thrust{i}_N': _f(rr,f'tug{i}_thrust_N'),
                f'tug{i}_hull_overlap': 0.0,
                f'tug{i}_hull_clearance_m': 99.0,
            })
        row['push1_normal_N']=_f(rr,'push1_N')
        row['push2_normal_N']=_f(rr,'push2_N')
        row['tow3_tension_N']=_f(rr,'tow3_N')
        row['tow4_tension_N']=_f(rr,'tow4_N')
        out.append(row)
    return out


def plot_trajectory_with_footprints(rows,dest,target=(50.0,-100.0),target_psi_deg=0.0,samples=7,show_tug_tracks=True):
    x=_values(rows,'x_m');y=_values(rows,'y_m');psi=_values(rows,'psi_deg')
    ship_outline=get_hull();tug_outline=_load_tug_outline()
    fig,ax=plt.subplots(figsize=(10,7.8))
    ax.plot(x,y,linewidth=2.3,color='tab:blue',label='Ship centre path',zorder=2)
    ax.scatter([x[0]],[y[0]],marker='o',s=35,color='tab:blue',zorder=6,label='Start')
    ax.scatter([target[0]],[target[1]],marker='x',s=90,color='black',zorder=6,label='Target')
    target_poly=_transform(ship_outline,target[0],target[1],target_psi_deg)
    ax.add_patch(Polygon(target_poly,closed=True,fill=False,edgecolor='tab:green',
                         linestyle='--',linewidth=1.4,zorder=3,label='Target hull'))
    if show_tug_tracks:
        for i,color in enumerate(TUG_COLORS,1):
            tx=_values(rows,f'tug{i}_x_m');ty=_values(rows,f'tug{i}_y_m')
            ax.plot(tx,ty,color=color,linewidth=1.0,alpha=.28,zorder=1,label=f'Tug {i} track')
    idxs=_sample_indices(len(rows),samples)
    for order,k in enumerate(idxs):
        alpha=0.18+0.60*(order+1)/len(idxs)
        ship_poly=_transform(ship_outline,x[k],y[k],psi[k])
        ax.add_patch(Polygon(ship_poly,closed=True,facecolor='tab:blue',edgecolor='black',
                             linewidth=0.8,alpha=alpha,zorder=4))
        _draw_orientation_arrow(ax,x[k],y[k],psi[k],28,'black',alpha=min(1,alpha+.1),z=5)
        row=rows[k]
        for i,color in enumerate(TUG_COLORS,1):
            tx=_f(row,f'tug{i}_x_m');ty=_f(row,f'tug{i}_y_m');tpsi=_f(row,f'tug{i}_psi_deg')
            tug_poly=_transform(tug_outline,tx,ty,tpsi)
            ax.add_patch(Polygon(tug_poly,closed=True,facecolor=color,edgecolor='black',
                                 linewidth=0.6,alpha=alpha*.92,zorder=6))
            _draw_orientation_arrow(ax,tx,ty,tpsi,8,'black',alpha=min(1,alpha+.05),z=7)
    handles,labels=ax.get_legend_handles_labels()
    seen=set();uh=[];ul=[]
    for h,l in zip(handles,labels):
        if l not in seen:
            seen.add(l);uh.append(h);ul.append(l)
    ax.legend(uh,ul,fontsize=8,ncol=2,loc='best')
    ax.set_xlabel('East / X (m)')
    ax.set_ylabel('North / Y (m)')
    ax.set_title('Ship trajectory with sampled ship and tug footprints')
    ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(dest/'trajectory.png',dpi=180);plt.close(fig)


def plot_four_stages(renderer_rows, cfg, dest, stage_fracs=(0.0,0.33,0.66,1.0)):
    hull=get_hull()
    # Keep publication panels on identical world-coordinate limits.
    allx=[q['x'] for q in renderer_rows]+[q[f'tug{i}_x_m'] for q in renderer_rows for i in range(1,5)]
    ally=[q['y'] for q in renderer_rows]+[q[f'tug{i}_y_m'] for q in renderer_rows for i in range(1,5)]
    cfg=dict(cfg)
    cfg['_visual_bounds']=[max(250.,float(np.max(np.abs(allx)))+40.),
                           max(180.,float(np.max(np.abs(ally)))+40.)]
    fig,axes=plt.subplots(2,2,figsize=(15.5,10.3),facecolor='white',constrained_layout=True)
    for ax,frac in zip(axes.flat,stage_fracs):
        idx=int(round((len(renderer_rows)-1)*frac))
        idx=max(0,min(len(renderer_rows)-1,idx))
        scene(ax,renderer_rows[idx],cfg,hull,history=renderer_rows[:idx+1],light=True)
        ax.set_title(
            f"t = {renderer_rows[idx]['t']:.0f} s   |   position error = "
            f"{np.hypot(renderer_rows[idx]['x']-cfg['target_x'], renderer_rows[idx]['y']-cfg['target_y']):.1f} m",
            fontsize=12,fontweight='bold',color='#1a405b')
    fig.savefig(dest/'berthing_four_stages.png',dpi=180,facecolor='white')
    plt.close(fig)


def plot_final_view(renderer_rows, cfg, dest):
    hull=get_hull()
    fig,ax=plt.subplots(figsize=(10.5,7.2),facecolor='white',constrained_layout=True)
    scene(ax,renderer_rows[-1],cfg,hull,history=renderer_rows,light=True)
    ax.set_title('Final berthing geometry view',fontsize=13,fontweight='bold',color='#1a405b')
    fig.savefig(dest/'berthing_final.png',dpi=180,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results/marl_eval/trajectory.csv')
    p.add_argument('--out',default='results/marl_eval')
    p.add_argument('--target-x',type=float,default=50.0)
    p.add_argument('--target-y',type=float,default=-100.0)
    p.add_argument('--target-psi-deg',type=float,default=0.0)
    p.add_argument('--footprint-samples',type=int,default=7,
                   help='Number of sampled poses used to draw ship/tug footprints on trajectory plot.')
    p.add_argument('--hide-tug-tracks',action='store_true')
    args=p.parse_args();dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)

    raw_rows=_read_rows(args.input)
    cfg=_cfg_from_args_and_rows(args,raw_rows)
    renderer_rows=_convert_rows_for_renderer(raw_rows,cfg)

    def v(key): return _values(raw_rows,key)
    t=v('time_s')/60.

    plot_trajectory_with_footprints(raw_rows,dest,target=(args.target_x,args.target_y),
                                    target_psi_deg=args.target_psi_deg,
                                    samples=args.footprint_samples,
                                    show_tug_tracks=not args.hide_tug_tracks)
    plot_four_stages(renderer_rows,cfg,dest)
    plot_final_view(renderer_rows,cfg,dest)

    figs=[('position_error',t,v('position_error_m'),'Time (min)','Position error (m)'),
          ('heading_error',t,v('heading_error_deg'),'Time (min)','Heading error (deg)'),
          ('hull_gap',t,v('hull_gap_m'),'Time (min)','Hull–quay clearance (m)')]
    for name,xx,yy,xlab,ylab in figs:
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(xx,yy,linewidth=2)
        ax.set_xlabel(xlab);ax.set_ylabel(ylab);ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(dest/f'{name}.png',dpi=170);plt.close(fig)

    force_keys=[k for k in ('push1_N','push2_N','tow3_N','tow4_N') if k in raw_rows[0]]
    if force_keys:
        fig,ax=plt.subplots(figsize=(8,4))
        labels=['T1: push','T2: push','T3: tow','T4: tow']
        for i,key in enumerate(force_keys):
            ax.plot(t,v(key)/1e6,label=labels[i])
        ax.set_xlabel('Time (min)');ax.set_ylabel('Normal force / tension (MN)')
        ax.grid(alpha=.3);ax.legend(ncol=2);fig.tight_layout()
        fig.savefig(dest/'contact_forces.png',dpi=170);plt.close(fig)

    thrust_keys=[f'tug{i}_thrust_N' for i in range(1,5) if f'tug{i}_thrust_N' in raw_rows[0]]
    if thrust_keys:
        fig,ax=plt.subplots(figsize=(8,4))
        for i,key in enumerate(thrust_keys,1):
            ax.plot(t,v(key)/1e6,label=f'T{i}')
        ax.set_xlabel('Time (min)');ax.set_ylabel('Signed propulsive thrust (MN)')
        ax.legend(ncol=4);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(dest/'signed_thrust.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    main()
