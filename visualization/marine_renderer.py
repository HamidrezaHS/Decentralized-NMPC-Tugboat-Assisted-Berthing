"""Publication-quality visualization of the logged physical tug/ship geometry.

Crucial conventions:
* PUSH tug (1,2): draw contact pads, never a fictitious towline.
* TOW tug (3,4): draw a rope between hull attachment and tug attachment.
* A taut line is shown for geometry, even if the upstream rigid/bilateral
  constraint model calculates inadmissible negative tension. Such a line MUST
  be drawn dashed RED, never represented as physically valid slack behavior.
* No reconstruction of hidden tug states from ship-only telemetry: required
  positions are logged by the physical plant, not given to any controller.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from functools import lru_cache
import numpy as np

COLORS=['#22d3ee','#60a5fa','#fbbf24','#f472b6']
NAVY='#071b2f'
WATER='#0c2b40'
TEXT='#e6f2ff'


def _coerce_csv_value(key,value):
    if key.endswith('_mode'):
        return value
    if value in ('', None):
        return float('nan')
    return float(value)


def load_run(output):
    output=Path(output)
    with (output/'trajectory.csv').open(encoding='utf-8',newline='') as f:
        rows=[{k:_coerce_csv_value(k,v) for k,v in r.items()} for r in csv.DictReader(f)]
    if not rows:raise ValueError('No simulation data')
    needed=['tug1_x_m','tug4_rope_anchor_y_m']
    missing=[k for k in needed if k not in rows[0]]
    if missing:raise ValueError(f'Geometry telemetry missing: {missing}. Rerun with the updated run_decentralized.py.')
    cfg=json.loads((output/'configuration.json').read_text(encoding='utf-8'))
    return rows,cfg


def transforms(points, x, y, heading_deg):
    angle=np.deg2rad(heading_deg)
    c,s=np.cos(angle),np.sin(angle)
    pts=np.asarray(points,float)
    return pts@np.array([[c,s],[-s,c]])+np.array([x,y])


def local_tug_outline(length=22.943,width=8.5):
    """Load the actual tug waterline from project DXF (with simple fallback)."""
    return _cached_tug_waterline()


@__import__('functools').lru_cache(maxsize=1)
def _cached_tug_waterline():
    root=Path(__file__).resolve().parents[1]
    path=root/'geom'/'waterline_tug.dxf'
    try:
        from geom.constraints import Constraints
        return np.asarray(Constraints(str(path)).points,float)
    except (FileNotFoundError,ValueError,ImportError):
        length,width=22.943,8.5
        L=length/2;B=width/2
        return np.array([[-L,-B*.65],[-L*.86,-B], [L*.55,-B],
                         [L,-B*.50],[L,0],[L,B*.50],[L*.55,B],
                         [-L*.86,B],[-L,B*.65]])

@lru_cache(maxsize=1)
def get_hull():
    from geom.constraints import Constraints
    return np.asarray(Constraints(str(Path(__file__).resolve().parents[1]/'geom'/'waterline_ship.dxf')).points,float)


@lru_cache(maxsize=1)
def get_tug_hull():
    from geom.constraints import Constraints
    return np.asarray(Constraints(str(Path(__file__).resolve().parents[1]/'geom'/'waterline_tug.dxf')).points,float)


def _direction_vector_towards(angle_deg: float):
    a=np.deg2rad(float(angle_deg))
    return np.cos(a),np.sin(a)


def _draw_environment_overlay(ax, row, cfg, light=False):
    """Unobtrusive earth-fixed arrows: CURRENT left, WIND right.

    Arrows indicate where wind/current are going *TO*, in the same XY axes
    used for the manoeuvre. Use axes coordinates so diagram limits and geometry
    cannot be altered by the annotations.
    """
    from matplotlib import patheffects
    txt = '#193b55' if light else '#ebf5ff'
    def arrow(label, speed, direction, x, y, color):
        if not np.isfinite(speed) or speed < 1e-8:
            return
        u, v = _direction_vector_towards(direction)
        ax.annotate('', xy=(x+0.042*u,y+0.052*v), xytext=(x,y),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle='-|>', lw=2.15,color=color,
                                    shrinkA=0,shrinkB=0), zorder=39)
        line=ax.text(x+0.063,y,f'{label} {speed:g} m/s',
            transform=ax.transAxes,color=txt,fontsize=8.5,fontweight='bold',
            ha='left',va='center',zorder=40)
        line.set_path_effects([patheffects.withStroke(linewidth=2.8,
            foreground='#f4f9fb' if light else '#0c2b40')])
    # Current: upper left; Wind: upper right. Neither adds boxes or axes.
    arrow('Current',float(row.get('current_mps',cfg.get('current_mps',0.)) or 0.),
          float(row.get('current_to_deg',cfg.get('current_to_deg',0.)) or 0.),
          0.048,0.900,'#0e9774')
    arrow('Wind',float(row.get('wind_mps',cfg.get('wind_mps',0.)) or 0.),
          float(row.get('wind_to_deg',cfg.get('wind_to_deg',0.)) or 0.),
          0.722,0.900,'#0284c7')


def scene(ax,row,cfg,hull,history=None,light=False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Circle,FancyArrowPatch,Rectangle
    ax.set_facecolor('#f4f9fb' if light else WATER)
    txt='#163047' if light else TEXT
    hull_col='#a7bac5' if light else '#b6d2e0'
    hull_edge='#274359' if light else '#d3e6f4'
    x,y,ps=row['x'],row['y'],row['psi_deg']
    tx,ty,tpsi=cfg['target_x'],cfg['target_y'],np.deg2rad(cfg['target_psi'])
    half_b=(np.max(hull[:,1])-np.min(hull[:,1]))/2
    normal=np.array([-np.sin(tpsi),np.cos(tpsi)])
    # Quay side must be fixed from INITIAL mission geometry, not flipped
    # as the vessel crosses/overshoots the goal during simulation.
    x0,y0=cfg['x0'],cfg['y0']
    if np.dot(np.array([tx-x0,ty-y0]),normal)<0:normal=-normal
    wall=np.array([tx,ty])+normal*(half_b+6.)
    tangent=np.array([np.cos(tpsi),np.sin(tpsi)])
    # Pier extends out from approach side; geometric wall is the outer face.
    pier=np.array([wall-275*tangent,wall+275*tangent,
                   wall+275*tangent+44*normal,wall-275*tangent+44*normal])
    ax.add_patch(Polygon(pier,closed=True,facecolor='#8f9fac' if light else '#6c7f8d',edgecolor='none',alpha=.80,zorder=1))
    edge=np.array([wall-275*tangent,wall+275*tangent]);ax.plot(edge[:,0],edge[:,1],color='#facc15',lw=3,solid_capstyle='round',zorder=4)
    for offset in np.arange(-250,251,24):
        pos=wall+offset*tangent
        ax.plot([pos[0],pos[0]+7*normal[0]],[pos[1],pos[1]+7*normal[1]],color='#1e3a4d',lw=2,alpha=.7,zorder=2)
    ax.text(*(wall+normal*25), 'QUAY / FENDER LINE',ha='center',va='center',fontsize=9,fontweight='bold',color=txt,zorder=8)
    target_shape=transforms(hull,tx,ty,cfg['target_psi'])
    ax.add_patch(Polygon(target_shape,closed=True,fill=False,lw=1.6,edgecolor='#10b981',ls='--',alpha=.88,zorder=4))
    if history is not None and len(history)>1:
        track=np.array([[q['x'],q['y']] for q in history]);ax.plot(track[:,0],track[:,1],color='#5eead4',lw=1.9,alpha=.8,zorder=4)
    ship_polygon=transforms(hull,x,y,ps)
    ax.add_patch(Polygon(ship_polygon,closed=True,facecolor=hull_col,edgecolor=hull_edge,lw=1.7,zorder=9))
    # Ship deck, cargo blocks, midship line, wheelhouse aft.
    cargo=[(-95,-63),(-53,-21),(-11,21),(31,63),(73,105)]
    for l,r in cargo:
        poly=transforms([[l,-12],[r,-12],[r,12],[l,12]],x,y,ps)
        ax.add_patch(Polygon(poly,facecolor='#627e92' if not light else '#72899a',edgecolor='#d0e2ee' if not light else '#38566b',lw=.6,zorder=10))
    deck=transforms([[-130,-14],[-112,-14],[-112,14],[-130,14]],x,y,ps)
    ax.add_patch(Polygon(deck,facecolor='#e1ebf2',edgecolor='#32506c',lw=.8,zorder=11))
    # Heading glyph.
    bow=transforms([[106,0],[132,0]],x,y,ps);ax.annotate('',xy=bow[-1],xytext=bow[0],arrowprops=dict(arrowstyle='-|>',lw=2,color='#0f766e'),zorder=12)
    tugoutline=get_tug_hull()
    for i,color in enumerate(COLORS,1):
        pos=(row[f'tug{i}_x_m'],row[f'tug{i}_y_m'])
        angle=row[f'tug{i}_psi_deg']
        polygon=transforms(tugoutline,*pos,angle)
        ax.add_patch(Polygon(polygon,closed=True,facecolor=color,edgecolor='#071b2f' if not light else '#263d50',lw=1.4,zorder=17))
        house=transforms([[-4,-2.7],[4,-2.7],[4,2.7],[-4,2.7]],*pos,angle)
        ax.add_patch(Polygon(house,facecolor='#eef7fc',edgecolor='#1d3650',lw=.8,zorder=18))
        px,py=row[f'tug{i}_contact_x_m'],row[f'tug{i}_contact_y_m']
        if i>=3:
            ax0=row[f'tug{i}_rope_anchor_x_m'];ay0=row[f'tug{i}_rope_anchor_y_m']
            tension=row[f'tow{i}_tension_N']
            valid=tension>=-1000
            if row.get(f'tug{i}_hull_overlap',0) or row.get(f'tug{i}_hull_clearance_m',999)<2:
                ax.scatter(pos[0],pos[1],marker='x',s=170,color='#ef4444',zorder=25)
            ax.plot([px,ax0],[py,ay0],lw=2.2,ls='-' if valid else '--',
                    color='#f6d38a' if valid else '#fb7185',zorder=14,alpha=.96)
            if not valid:ax.scatter([(px+ax0)/2],[(py+ay0)/2],marker='x',s=23,color='#fb7185',zorder=16)
            ax.scatter([px,ax0],[py,ay0],s=14,color='#f6d38a' if valid else '#fb7185',zorder=16)
        else:
            ax.add_patch(Circle((px,py),radius=3.1,color=color,alpha=.85,zorder=14))
            ax.plot([px,pos[0]],[py,pos[1]],color=color,ls=':',lw=1.8,zorder=14)
        ux=np.cos(np.deg2rad(angle));uy=np.sin(np.deg2rad(angle))
        ax.annotate('',xy=(pos[0]+ux*20,pos[1]+uy*20),xytext=(pos[0]+ux*8,pos[1]+uy*8),
                    arrowprops=dict(arrowstyle='-|>',color='#091d30',lw=1.3),zorder=19)
        # Blue propulsion arrow for astern tow. Black arrow = bow orientation.
        # These are intentionally different vectors, not a turned-around tug.
        if i>=3 and row[f'thrust{i}_N'] < -1.0:
            ax.annotate('',xy=(pos[0]-ux*26,pos[1]-uy*26),
                        xytext=(pos[0]-ux*9,pos[1]-uy*9),
                        arrowprops=dict(arrowstyle='-|>',color='#2563eb',lw=2),zorder=20)
        label=transforms([[0,14]],*pos,angle)[0]
        ax.text(label[0],label[1],f'T{i} '+('PUSH' if i<3 else 'TOW'),fontsize=8.2,
                color=txt,fontweight='bold',ha='center',va='bottom',zorder=20,
                bbox=dict(facecolor='#071b2f' if not light else '#ffffff',alpha=.80,edgecolor='none',boxstyle='round,pad=0.2'))
    # Expand bounds to retain all tugs even with displaced/rotated initial pose.
    coords=np.array([[x,y],[tx,ty],wall]+[[row[f'tug{i}_x_m'],row[f'tug{i}_y_m']] for i in range(1,5)])
    halfx=max(230.,float(np.max(abs(coords[:,0])))+38.)
    halfy=max(170.,float(np.max(abs(coords[:,1])))+38.)
    if '_visual_bounds' in cfg:
        halfx,halfy=cfg['_visual_bounds']
    ax.set_xlim(-halfx,halfx);ax.set_ylim(-halfy,halfy)
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('East / X  [m]',color=txt,fontsize=10)
    ax.set_ylabel('North / Y  [m]',color=txt,fontsize=10)
    ax.tick_params(colors=txt,labelsize=9)
    ax.grid(color='#bfd2df' if light else '#aec1cc',alpha=.12,lw=.7)
    for spine in ax.spines.values():spine.set_color('#afc0cb' if light else '#446478')
    _draw_environment_overlay(ax,row,cfg,light=light)
    return wall


def _telemetry_lines(row,cfg):
    import math
    err=math.hypot(row['x']-cfg['target_x'],row['y']-cfg['target_y'])
    speed=math.hypot(row['ue'],row['ve'])
    return [('ELAPSED',f"{row['t']:.1f} s"),('DISTANCE ERROR',f'{err:.2f} m'),
            ('HEADING ERROR',f"{(row['psi_deg']-cfg['target_psi']+180)%360-180:+.2f} deg"),
            ('SHIP SPEED',f'{speed:.3f} m/s'),('FENDER GAP',f"{row['fender_gap_m']:.2f} m"),
            ('APPROACH SPEED',f"{row['approach_speed_mps']:+.3f} m/s")]


def dashboard(fig,ax_side,row,cfg):
    ax_side.set_facecolor(NAVY);ax_side.set_xlim(0,1);ax_side.set_ylim(0,1);ax_side.axis('off')
    ax_side.text(.06,.958,'BERTHING TELEMETRY',fontsize=13,fontweight='bold',color=TEXT,va='top')
    ax_side.text(.06,.910,'Decentralized game-based control  /  4 players',fontsize=8.5,color='#9db6c6',va='top')
    for i,(key,value) in enumerate(_telemetry_lines(row,cfg)):
        ypos=.84-i*.085
        ax_side.text(.06,ypos,key,fontsize=9.5,color='#93b4c5',va='center')
        ax_side.text(.94,ypos,value,fontsize=11.3,color='#f8fafc',ha='right',va='center',fontweight='semibold')
    ax_side.axhline(.285,xmin=.06,xmax=.94,color='#3c617a',lw=1)
    ax_side.text(.06,.253,'TUG FORCES / COMMANDS',fontsize=9.2,color='#a6d0e4',fontweight='bold')
    maxT=900000
    for i,c in enumerate(COLORS,1):
        yy=.203-(i-1)*.045
        val=row[f'thrust{i}_N']
        ax_side.text(.06,yy,f'T{i}',color=c,fontsize=10,va='center',fontweight='bold')
        ax_side.add_patch(__import__('matplotlib').patches.Rectangle((.18,yy-.012),.52,.024,fc='#203c51',ec='none'))
        ax_side.add_patch(__import__('matplotlib').patches.Rectangle((.18,yy-.012),.52*min(1,abs(val)/maxT),.024,fc=c,ec='none'))
        ax_side.text(.95,yy,f'{val/1000:+.0f} kN',ha='right',va='center',color=TEXT,fontsize=8.5)
    ax_side.text(.06,.012,'T3/T4: negative thrust = ASTERN / bow toward ship',fontsize=7.3,color='#9edbc7')


def draw_frame(row,cfg,hull,history,fig,ax,ax_side):
    ax.clear();scene(ax,row,cfg,hull,history)
    ax.set_title('COOPERATIVE SHIP BERTHING  |  PHYSICAL PLAN VIEW',fontsize=13,fontweight='bold',color=TEXT,pad=12)
    ax_side.clear();dashboard(fig,ax_side,row,cfg)


def render(output,mode='all',fps=12,frames=90,dpi=120):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,FFMpegWriter,PillowWriter
    rows,cfg=load_run(output);output=Path(output)
    hull=get_hull()
    # Fix the same extent across all times, so animations do not change scale.
    xall=np.array([r['x'] for r in rows]+[r[f'tug{i}_x_m'] for r in rows for i in (1,2,3,4)])
    yall=np.array([r['y'] for r in rows]+[r[f'tug{i}_y_m'] for r in rows for i in (1,2,3,4)])
    cfg['_visual_bounds']=[max(230.,float(np.max(abs(xall)))+42.,abs(cfg['target_x'])+185.),
                           max(170.,float(np.max(abs(yall)))+42.,abs(cfg['target_y'])+105.)]
    metrics_file=output/'metrics.json'
    metrics=json.loads(metrics_file.read_text(encoding='utf-8')) if metrics_file.exists() else {}
    fig=plt.figure(figsize=(15.6,8.2),facecolor=NAVY)
    gs=fig.add_gridspec(1,2,width_ratios=[3.05,1],wspace=.08,left=.055,right=.975,bottom=.10,top=.90)
    ax=fig.add_subplot(gs[0,0]);side=fig.add_subplot(gs[0,1])
    indexes=np.unique(np.linspace(0,len(rows)-1,min(frames,len(rows)),dtype=int))
    def update(j):
        k=int(indexes[j]);draw_frame(rows[k],cfg,hull,rows[:k+1],fig,ax,side)
        return ()
    update(len(indexes)-1)
    if metrics.get('simulation_stop_reason'):
        fig.text(.53,.965,'SAFETY TERMINATION — '+metrics['simulation_stop_reason'],
          ha='center',fontsize=9,color='#fda4af',fontweight='bold')
    fig.savefig(output/'berthing_final.png',dpi=160,facecolor=fig.get_facecolor())
    if mode in ('all','gif','mp4'):
        animation=FuncAnimation(fig,update,frames=len(indexes),interval=1000/fps,blit=False,repeat=True)
        if mode in ('all','mp4'):
            try:animation.save(str(output/'berthing_animation.mp4'),writer=FFMpegWriter(fps=fps,bitrate=2000),dpi=dpi)
            except (RuntimeError,FileNotFoundError,OSError) as exc:
                print('Could not create MP4 (ffmpeg optional):',exc)
        if mode in ('all','gif'):
            # Palette-compressed FFmpeg GIF is far faster than rendering a second
            # frame-by-frame Pillow pass if an MP4 and ffmpeg are available.
            import shutil, subprocess
            movie=output/'berthing_animation.mp4'
            ffmpeg=shutil.which('ffmpeg')
            if ffmpeg is not None and movie.exists():
                vf='fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5'
                result=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y',
                    '-i',str(movie),'-vf',vf,'-loop','0',str(output/'berthing_animation.gif')],
                    check=False,capture_output=True,text=True)
                if result.returncode:print('GIF encoding fallback:',result.stderr[-300:])
                gif_done=result.returncode==0
            else:gif_done=False
            if not gif_done:
                animation.save(str(output/'berthing_animation.gif'),writer=PillowWriter(fps=fps),dpi=82)
    plt.close(fig)
    # Four-stage publication-quality figure (four separate panels).
    fig,axes=plt.subplots(2,2,figsize=(15.5,10.3),facecolor='white',constrained_layout=True)
    for ax,frac in zip(axes.flat,[0,.33,.66,1]):
        idx=int((len(rows)-1)*frac)
        scene(ax,rows[idx],cfg,hull,history=rows[:idx+1],light=True)
        ax.set_title(f"t = {rows[idx]['t']:.0f} s   |   position error = "
                     f"{np.hypot(rows[idx]['x']-cfg['target_x'],rows[idx]['y']-cfg['target_y']):.1f} m",
                     fontsize=12,fontweight='bold',color='#1a405b')
    if metrics.get('simulation_stop_reason'):
        fig.suptitle('SAFETY-TERMINATED RUN — '+metrics['simulation_stop_reason'],
            color='#b91c1c',fontsize=10,fontweight='bold')
    fig.savefig(output/'berthing_four_stages.png',dpi=170,facecolor='white')
    plt.close(fig)
    print('Visualization files created in',output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/calm_900')
    p.add_argument('--mode',choices=['all','static','gif','mp4'],default='all')
    p.add_argument('--fps',type=int,default=12)
    p.add_argument('--frames',type=int,default=90)
    args=p.parse_args()
    render(args.input,mode=args.mode,fps=args.fps,frames=args.frames)

if __name__=='__main__':main()
