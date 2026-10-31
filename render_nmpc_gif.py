"""Generate a diagnostic GIF from recorded NMPC simulation (no invented frames)."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from plot_marl import _read_rows, _convert_rows_for_renderer
from visualization.marine_renderer import scene,get_hull
import json


def create(folder,frames=55,fps=7):
    folder=Path(folder)
    raw=_read_rows(folder/'trajectory.csv')
    cfg0=json.loads((folder/'configuration.json').read_text(encoding='utf8'))
    cfg=dict(target_x=cfg0['target_x'],target_y=cfg0['target_y'],target_psi=cfg0['target_psi_deg'],
             x0=float(raw[0]['x_m']),y0=float(raw[0]['y_m']))
    rows=_convert_rows_for_renderer(raw,cfg)
    allx=[q['x'] for q in rows]+[q[f'tug{i}_x_m'] for q in rows for i in range(1,5)]
    ally=[q['y'] for q in rows]+[q[f'tug{i}_y_m'] for q in rows for i in range(1,5)]
    cfg['_visual_bounds']=[max(250.,float(np.max(np.abs(allx)))+40.),
                           max(180.,float(np.max(np.abs(ally)))+40.)]
    idxs=np.unique(np.linspace(0,len(rows)-1,min(frames,len(rows)),dtype=int))
    fig,ax=plt.subplots(figsize=(9.5,6.5),facecolor='white')
    status=json.loads((folder/'metrics.json').read_text(encoding='utf8'))['status']
    def update(j):
        ax.clear()
        idx=int(idxs[j]);q=rows[idx]
        scene(ax,q,cfg,get_hull(),history=rows[:idx+1],light=True)
        error=float(np.hypot(q['x']-cfg['target_x'],q['y']-cfg['target_y']))
        ax.set_title(f'Decentralized NMPC  |  t = {q["t"]:.0f} s  |  error = {error:.1f} m',
                     fontsize=11,fontweight='bold')
        return ()
    ani=FuncAnimation(fig,update,frames=len(idxs),interval=1000./fps,blit=False,repeat=True)
    dest=folder/'nmpc_berthing.gif'
    ani.save(str(dest),writer=PillowWriter(fps=fps),dpi=95)
    plt.close(fig)
    print('Animation:',dest,'| terminal status:',status)
    return dest


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/nmpc_h20')
    p.add_argument('--frames',type=int,default=55);p.add_argument('--fps',type=int,default=7)
    a=p.parse_args();create(a.input,frames=a.frames,fps=a.fps)

if __name__=='__main__':main()
