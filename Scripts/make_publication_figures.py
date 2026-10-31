"""Reproduce manuscript drawings and plots directly from archived simulation logs.

Dependencies: numpy, matplotlib. No image synthesis is used. Vessel outlines are
serialized from the provided simulator waterline geometry (CSV in Data/).
Run: python Scripts/make_publication_figures.py
"""
from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch, FancyArrowPatch, Arc
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'Data'; FIG=ROOT/'Figures'; FIG.mkdir(parents=True,exist_ok=True)
ship=np.loadtxt(DATA/'ship_hull_xy_m.csv',delimiter=',',skiprows=1)
tug=np.loadtxt(DATA/'tug_hull_xy_m.csv',delimiter=',',skiprows=1)
with (DATA/'directional_matrix.csv').open(newline='') as f:
    records=[{k:(float(v) if k not in ('status',) else v) for k,v in r.items()}
             for r in csv.DictReader(f)]
# Plot style: restrained, print legible, figure text intentionally sparse.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.8,'axes.titlesize':10.4,
                     'axes.labelsize':9.5,'xtick.labelsize':8.7,'ytick.labelsize':8.7,
                     'legend.fontsize':8.7,'pdf.fonttype':42,'ps.fonttype':42,
                     'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':360,'axes.linewidth':0.8})
NAVY='#203C52'; BLUE='#24699A'; TEAL='#0C8D84'; ORANGE='#C47B28'; PURPLE='#7A60A8';
RED='#B94C4C'; GREY='#64798A'; PALE='#E7EFF4'; QUAY='#B8C6CF'
COLORS=[BLUE,TEAL,ORANGE,PURPLE]
DASH=['-','--','-.',':']

def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=0.10)
    fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=0.10)
    plt.close(fig)

def transformed(shape,x,y,deg):
    a=np.deg2rad(float(deg)); c,s=np.cos(a),np.sin(a)
    return shape @ np.array([[c,s],[-s,c]])+np.array([x,y])

def run(name):
    with (DATA/f'case_{name}_trajectory.csv').open(newline='') as f:
        out=[]
        for r in csv.DictReader(f):
            out.append({k:(float(v) if v not in ('',None) else np.nan) for k,v in r.items()})
    return out
R={'0/0':run('w0_c0'),'90/90':run('w90_c90'),'270/270':run('w270_c270')}


def draw_vessels(ax,r,labels=False,alpha=1.,line_thickness=1.1,ship_face='#B7C8D5'):
    x,y,psi=r['x_m'],r['y_m'],r['psi_deg']
    hull=transformed(ship,x,y,psi)
    ax.add_patch(Polygon(hull,closed=True,facecolor=ship_face,edgecolor=NAVY,
                         lw=line_thickness,alpha=alpha,zorder=5))
    for l,h in [(-110,-75),(-69,-34),(-28,7),(13,48),(54,89)]:
        cargo=transformed(np.array([[l,-11],[h,-11],[h,11],[l,11]]),x,y,psi)
        ax.add_patch(Polygon(cargo,closed=True,facecolor='#718DA2',edgecolor='#426075',
                             lw=0.5,alpha=alpha,zorder=6))
    for i,col in enumerate(COLORS,1):
        tx,ty,tpsi=(r[f'tug{i}_x_m'],r[f'tug{i}_y_m'],r[f'tug{i}_psi_deg'])
        poly=transformed(tug,tx,ty,tpsi)
        ax.add_patch(Polygon(poly,closed=True,facecolor=col,edgecolor=NAVY,lw=0.75,
                             alpha=alpha,zorder=8))
        anchor=np.array([r[f'tug{i}_contact_x_m'],r[f'tug{i}_contact_y_m']])
        if i>=3:
            rope=np.array([r[f'tug{i}_rope_anchor_x_m'],r[f'tug{i}_rope_anchor_y_m']])
            ax.plot([anchor[0],rope[0]],[anchor[1],rope[1]],color=ORANGE,lw=1.5,
                    alpha=alpha,zorder=7)
        else:
            ax.plot([anchor[0],tx],[anchor[1],ty],color=col,lw=1.1,
                    ls=':',alpha=alpha,zorder=7)
        if labels:
            ax.annotate(f'{"P" if i<3 else "T"}{i}',(tx,ty),xytext=(0,9),
                        textcoords='offset points',fontsize=9,color=NAVY,
                        ha='center',fontweight='bold',zorder=10,
                        bbox=dict(boxstyle='round,pad=0.1',facecolor='white',edgecolor='none',alpha=.78))

def quay_target(ax,track=True):
    q=-135.0
    ax.add_patch(Rectangle((-250,-200),520,65,facecolor=QUAY,alpha=0.42,zorder=-5))
    ax.plot([-250,270],[q,q],color='#617789',lw=2.1,zorder=2)
    target=transformed(ship,50.,-100.,0.)
    ax.add_patch(Polygon(target,fill=False,edgecolor=TEAL,lw=1.4,ls=(0,(5,3)),zorder=3))
    if track:
        ax.plot(50,-100,marker='+',markersize=10,color=TEAL,markeredgewidth=1.7,zorder=3)

def world_axes(ax,labels=True):
    ax.set_xlim(-230,240); ax.set_ylim(-179,141)
    ax.set_aspect('equal',adjustable='box')
    if labels:
        ax.set_xlabel('Earth-fixed X (m)'); ax.set_ylabel('Earth-fixed Y (m)')
    ax.grid(color='#CBD5DD',alpha=.42,lw=.5)
    ax.tick_params(length=2.5)

# Figure 1: original run geometry at the initial instant + accurate body-frame station view
r0=R['90/90'][0]
fig,axs=plt.subplots(1,2,figsize=(12.0,4.9),gridspec_kw={'width_ratios':[1.28,1]},layout='constrained')
ax=axs[0];quay_target(ax);draw_vessels(ax,r0,labels=True)
world_axes(ax)
ax.annotate(r'$\psi_0=20^\circ$',xy=(15,8),xytext=(-35,105),
            arrowprops=dict(arrowstyle='-',color=GREY,lw=0.75),fontsize=9.4,color=NAVY)
ax.text(53,-157,'Quay / fender line',fontsize=9.0,color=NAVY,ha='center')
ax.text(68,-93,'Desired hull pose',fontsize=8.5,color=TEAL)
ax.set_title('(a) Recorded initial ship--tug configuration',loc='left',pad=9,fontweight='bold',color=NAVY)
ax=axs[1]
ax.add_patch(Polygon(ship,closed=True,facecolor='#B7C8D5',edgecolor=NAVY,lw=1.25))
for l,h in [(-110,-75),(-69,-34),(-28,7),(13,48),(54,89)]:
    poly=np.array([[l,-11],[h,-11],[h,11],[l,11]])
    ax.add_patch(Polygon(poly,facecolor='#718DA2',edgecolor='#426075',lw=.5))
ca,sa=np.cos(np.deg2rad(-20)),np.sin(np.deg2rad(-20))
def bodyxy(x,y):
    return np.array([ca*x-sa*y,sa*x+ca*y])
for i,col in enumerate(COLORS,1):
    tx,ty=bodyxy(r0[f'tug{i}_x_m'],r0[f'tug{i}_y_m'])
    p=transformed(tug,tx,ty,r0[f'tug{i}_psi_deg']-20)
    ax.add_patch(Polygon(p,facecolor=col,edgecolor=NAVY,lw=.85))
    ax.annotate(f'{"P" if i<3 else "T"}{i}',(tx,ty),xytext=(0,9),textcoords='offset points',
                fontsize=9,color=NAVY,ha='center',fontweight='bold')
    cx,cy=bodyxy(r0[f'tug{i}_contact_x_m'],r0[f'tug{i}_contact_y_m'])
    if i>=3:
        a1,a2=bodyxy(r0[f'tug{i}_rope_anchor_x_m'],r0[f'tug{i}_rope_anchor_y_m'])
        ax.plot([cx,a1],[cy,a2],color=ORANGE,lw=1.25)
    else:ax.plot([cx,tx],[cy,ty],color=col,ls=':',lw=1.25)
ax.scatter([-96,96],[29,29],s=25,color=BLUE,edgecolors='white',zorder=10)
ax.scatter([-160,160],[29,29],s=25,color=ORANGE,edgecolors='white',zorder=10)
ax.annotate('',xy=(60,0),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color=TEAL,lw=1.5))
ax.annotate('',xy=(0,45),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color=TEAL,lw=1.5))
ax.text(64,-3,r'$x_b$',color=TEAL,fontsize=10)
ax.text(3,47,r'$y_b$',color=TEAL,fontsize=10)
ax.set(xlim=(-211,211),ylim=(-86,110),xlabel='Body-fixed longitudinal position (m)',
       ylabel='Body-fixed transverse position (m)')
ax.set_aspect('equal');ax.grid(alpha=.20,lw=.5)
ax.set_title('(b) Relative attachment geometry',loc='left',pad=9,fontweight='bold',color=NAVY)
save(fig,'Fig01_ProblemConfiguration')

# Figure 2: exact direction convention, without crowded labels
fig,axs=plt.subplots(1,2,figsize=(10.8,3.4),layout='constrained')
for ax in axs:
    ax.set_aspect('equal');ax.set_xlim(-155,155);ax.set_ylim(-112,112)
    ax.axis('off')
# show hull outline in target pose and orientation
for ax in axs:
    ax.add_patch(Polygon(ship*.45,facecolor='#C5D4DF',edgecolor=NAVY,lw=1.2))
    ax.annotate('',xy=(121,0),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=1.15,color=GREY))
    ax.annotate('',xy=(0,101),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=1.15,color=GREY))
    ax.text(125,5,r'$+X_E$',color=GREY,fontsize=9)
    ax.text(5,98,r'$+Y_N$',color=GREY,fontsize=9)
ax=axs[0]
for end,lab,col,offset in [((113,0),'0°',BLUE,(2,-19)),((0,85),'90°',TEAL,(-32,1)),
                           ((-113,0),'180°',ORANGE,(-5,-18)),((0,-88),'270°',RED,(9,-1))]:
    ax.annotate('',xy=end,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color=col,lw=2))
    ax.annotate(lab,end,xytext=offset,textcoords='offset points',ha='center',fontsize=9.8,
                color=col,fontweight='bold')
ax.set_title('(a) Global angle convention',loc='left',pad=8,fontweight='bold',color=NAVY)
ax=axs[1]
ax.annotate('',xy=(-37,90),xytext=(-37,-15),arrowprops=dict(arrowstyle='-|>',lw=3,color=BLUE))
ax.annotate('',xy=(37,-90),xytext=(37,15),arrowprops=dict(arrowstyle='-|>',lw=3,color=TEAL))
ax.text(-70,82,r'Wind  $\theta_w=90^\circ$',ha='right',color=BLUE,fontweight='bold',fontsize=9.6)
ax.text(56,-78,r'Current  $\theta_c=270^\circ$',ha='left',color=TEAL,fontweight='bold',fontsize=9.6)
ax.set_title('(b) Example of independent wind/current directions',loc='left',pad=8,fontweight='bold',color=NAVY)
save(fig,'Fig02_DisturbanceDirection')

# Figure 3: independent controller blocks and shared plant, orthogonal arrows
fig,ax=plt.subplots(figsize=(11.7,4.35));ax.set_xlim(0,12);ax.set_ylim(0,7.2);ax.axis('off')
def block(ax,x,y,w,h,title,subtitle='',fc='white',ec=NAVY,tc=NAVY,fs=9.5):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07,rounding_size=0.12',lw=1.25,edgecolor=ec,facecolor=fc)
    ax.add_patch(p)
    ax.text(x+w/2,y+h*(.65 if subtitle else .52),title,fontsize=fs,fontweight='bold',
            ha='center',va='center',color=tc)
    if subtitle:ax.text(x+w/2,y+h*.27,subtitle,fontsize=8.5,ha='center',va='center',color=tc)

def arrow(ax,start,end,c=GREY,lw=1.25):
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',lw=lw,color=c,shrinkA=0,shrinkB=0))

block(ax,4.0,6.1,4.0,.78,'Shared ship-state estimate',r'$x_s,y_s,\psi_s,u_s,v_s,r_s$  |  Berth reference',fc=PALE,ec=BLUE)
xs=[1.5,4.3,7.1,9.9]
ax.plot([1.5,9.9],[5.55,5.55],color=BLUE,lw=1.4)
arrow(ax,(6.0,6.09),(6.0,5.55),BLUE)
for i,x in enumerate(xs):
    arrow(ax,(x,5.55),(x,5.05),BLUE)
    block(ax,x-1.12,3.68,2.24,1.33,f'Tug {i+1}: independent NMPC',
          'Own state + local predictor',fc='#EAF3F7',ec=COLORS[i],fs=8.8)
    block(ax,x-1.03,2.45,2.06,.55,'Thrust / azimuth',fc='white',ec=COLORS[i],fs=8.5)
    arrow(ax,(x,3.66),(x,3.05),COLORS[i])
    arrow(ax,(x,2.44),(x,1.35),COLORS[i])
block(ax,.65,.27,10.7,1.04,'Coupled ship--tug dynamics',
      'Five 3-DOF vessels  |  unilateral contact  |  towline tension  |  environment',fc='#EEF2F4',ec='#91A9B8',fs=10)
# ship state feedback loop outside data bus; does not cross controller boxes
ax.plot([.65,.16,.16,3.95],[.74,.74,6.5,6.5],color=GREY,lw=1.2,ls='--')
arrow(ax,(3.95,6.5),(4.0,6.5),GREY)
ax.text(.31,5.8,'Feedback',rotation=90,fontsize=8.4,color=GREY,ha='left')
fig.text(.5,.008,'Local optimizations receive no peer-tug telemetry or planned actions.',
         ha='center',fontsize=9.2,color=GREY)
save(fig,'Fig03_DecentralizedArchitecture')

# Figure 4: algorithm flow, shows new decision mechanisms without crossing
fig,ax=plt.subplots(figsize=(10.4,5.4));ax.set_xlim(0,10);ax.set_ylim(0,9.2);ax.axis('off')
steps=[(7.72,'Measure ship and own-tug states','and local wind/current'),
       (6.20,'Update acceleration residual','from observed ship response'),
       (4.68,'Predict ship trajectory','six-state reduced-order nonlinear model'),
       (3.16,'Optimize two command segments','90-s horizon; bounded local NMPC'),
       (1.64,'Execute first command','thrust, azimuth and tug heading')]
for i,(y,head,desc) in enumerate(steps):
    block(ax,.45,y,5.55,1.0,head,desc,fc=PALE if i<3 else '#E5F2ED',ec=TEAL if i>=3 else BLUE,fs=10)
    if i>0:arrow(ax,(3.22,y+1.47),(3.22,y+1.07),GREY)
block(ax,.45,.18,5.55,.97,'Observe coupled plant response','updated ship/tug pose and velocity',fc='#F3F5F6',ec=GREY,fs=9.6)
arrow(ax,(3.22,1.58),(3.22,1.20),GREY)
# controls applied to optimization: no crossing with main sequence
block(ax,6.85,5.25,2.78,.95,'Adaptive push reserve','local input bound under adverse load',fc='#FFF3E7',ec=ORANGE,fs=9.5)
block(ax,6.85,3.63,2.78,.95,'Progress-triggered penalty','discourage counterproductive force',fc='#F0EDF7',ec=PURPLE,fs=9.5)
# horizontal arrows enter optimizer at right side distinct y levels
arrow(ax,(6.83,5.65),(6.02,4.02),ORANGE)
arrow(ax,(6.83,4.06),(6.02,3.48),PURPLE)
# feedback outside to top
ax.plot([.46,.12,.12,.46],[.68,.68,8.23,8.23],color=GREY,lw=1.0,ls='--')
arrow(ax,(.17,8.23),(.44,8.23),GREY)
ax.text(.20,4.80,'Receding-horizon feedback',rotation=90,fontsize=8.3,color=GREY,ha='center',va='center')
save(fig,'Fig04_LocalControlCycle')

# Figure 5: complete wind/current matrix (exact actual logged values)
angs=[0,90,180,270]
def matrix(key,fac=1.):
    return np.array([[next(r[key] for r in records if r['wind_toward_deg']==w and r['current_toward_deg']==c)*fac
                      for w in angs] for c in angs])
T=matrix('berthing_time_s',1/60); Q=matrix('integrated_abs_thrust_Ns',1e-9)
fig,axs=plt.subplots(1,2,figsize=(10.8,4.35),layout='constrained')
for ax,arr,ttl,cmap,unit,precision in [
    (axs[0],T,'(a) Time to terminal acceptance','YlGnBu','min',1),
    (axs[1],Q,'(b) Integrated absolute thrust','YlOrBr','GN s',2)]:
    im=ax.imshow(arr,cmap=cmap,vmin=arr.min()-.1*(arr.max()-arr.min()),vmax=arr.max())
    ax.set(xticks=range(4),yticks=range(4),xticklabels=[f'{x}°' for x in angs],
           yticklabels=[f'{x}°' for x in angs],xlabel='Wind toward direction',ylabel='Current toward direction')
    ax.set_title(ttl,loc='left',fontweight='bold',pad=8,color=NAVY)
    norm=Normalize(vmin=arr.min(),vmax=arr.max())
    for j in range(4):
        for k in range(4):
            val=arr[j,k]
            ax.text(k,j,f'{val:.{precision}f}',ha='center',va='center',fontsize=9.3,
                    color='white' if norm(val)>.70 else '#1A3448',fontweight='bold')
    cb=fig.colorbar(im,ax=ax,shrink=.76,pad=.026);cb.set_label(unit)
save(fig,'Fig05_DirectionalPerformance')

# Figure 6: tradeoff/constraint diagnostics; physical and optimization factors
fig,axs=plt.subplots(1,3,figsize=(12.0,3.9),layout='constrained')
cm={0:BLUE,90:TEAL,180:ORANGE,270:PURPLE}
mk={0:'o',90:'s',180:'^',270:'D'}
for r in records:
    wind=int(r['wind_toward_deg']);current=int(r['current_toward_deg']);color=cm[wind];marker=mk[current]
    t=r['berthing_time_s']/60; effort=r['integrated_abs_thrust_Ns']/1e9
    axs[0].scatter(t,effort,c=color,marker=marker,s=58,alpha=.88,edgecolors='white',linewidths=.65)
    axs[1].scatter(r['min_hull_gap_m'],r['peak_interaction_force_N']/1e3,c=color,marker=marker,s=58,
                   alpha=.88,edgecolors='white',linewidths=.65)
    axs[2].scatter(t,r['solver_success_rate'],c=color,marker=marker,s=58,alpha=.88,
                   edgecolors='white',linewidths=.65)
axs[0].set(xlabel='Berthing time (min)',ylabel='Integrated thrust (GN s)')
axs[1].set(xlabel='Minimum hull clearance (m)',ylabel='Peak interaction load (kN)')
axs[2].set(xlabel='Berthing time (min)',ylabel='Optimizer-success fraction')
for i,ttl in enumerate(['(a) Duration–effort trade-off','(b) Interaction load and clearance',
                         '(c) Numerical convergence']):
    axs[i].set_title(ttl,loc='left',fontweight='bold',color=NAVY,pad=9)
    axs[i].grid(alpha=.21,lw=.5)
axs[2].set_ylim(.57,1.02)
colors=[Line2D([],[],marker='o',lw=0,color=cm[a],markersize=7,label=f'Wind {a}°') for a in angs]
markers=[Line2D([],[],marker=mk[a],lw=0,color=GREY,markersize=7,label=f'Current {a}°') for a in angs]
fig.legend(handles=colors+markers,loc='lower center',bbox_to_anchor=(.5,-.09),ncol=8,frameon=False,fontsize=8)
save(fig,'Fig06_OperationalDiagnostics')

# Figure 7: actual recorded response across three disturbance directions
fig,axs=plt.subplots(2,2,figsize=(11.0,7.2),layout='constrained')
for (label,rr),col,ls in zip(R.items(),[BLUE,TEAL,ORANGE],['-','--','-.']):
    tm=np.array([r['time_s'] for r in rr])/60
    def v(k):return np.array([r[k] for r in rr],float)
    axs[0,0].plot(v('x_m'),v('y_m'),label=f'Wind/current {label}°',color=col,ls=ls,lw=1.8)
    axs[0,1].plot(tm,v('position_error_m'),color=col,ls=ls,lw=1.8)
    axs[1,0].plot(tm,v('heading_error_deg'),color=col,ls=ls,lw=1.8)
    axs[1,1].plot(tm,v('ship_speed_mps'),color=col,ls=ls,lw=1.8)
axs[0,0].plot(50,-100,marker='*',markersize=12,color=RED)
axs[0,0].set(xlabel='Earth-fixed X (m)',ylabel='Earth-fixed Y (m)')
axs[0,0].axis('equal');axs[0,0].legend(loc='best',frameon=False,fontsize=8.2)
axs[0,1].axhline(2.5,color=RED,ls=':',lw=1.15)
axs[1,0].axhline(2,color=RED,ls=':',lw=1.15)
axs[1,1].axhline(.045,color=RED,ls=':',lw=1.15)
axs[0,1].set(xlabel='Time (min)',ylabel='Position error (m)')
axs[1,0].set(xlabel='Time (min)',ylabel='Heading error (deg)')
axs[1,1].set(xlabel='Time (min)',ylabel='Ship speed (m/s)')
for ax,ttl in zip(axs.flat,['(a) Ship-centre path','(b) Position convergence',
                             '(c) Heading convergence','(d) Speed during berthing']):
    ax.set_title(ttl,loc='left',color=NAVY,fontweight='bold',pad=8)
    ax.grid(alpha=.20,lw=.5)
save(fig,'Fig07_RepresentativeTrajectories')

# Figure 8: physical four-stage reconstruction using simulated positions, not an illustration
rr=R['270/270'];inds=np.linspace(0,len(rr)-1,4,dtype=int)
fig,axs=plt.subplots(2,2,figsize=(10.7,7.9),layout='constrained')
for a,idx in zip(axs.flat,inds):
    quay_target(a);world_axes(a)
    a.plot([r['x_m'] for r in rr[:idx+1]],[r['y_m'] for r in rr[:idx+1]],
           color=TEAL,lw=1.3,alpha=.85)
    draw_vessels(a,rr[idx],labels=True)
    a.set_title(f"t = {rr[idx]['time_s']/60:.1f} min   |   position error = {rr[idx]['position_error_m']:.1f} m",
                fontsize=9.8,color=NAVY,pad=6,fontweight='bold')
save(fig,'Fig08_FourStageBerthing')

# Figure 9: force and actuator histories from the difficult recorded case
rr=R['270/270'];tm=np.array([r['time_s'] for r in rr])/60
fig,axs=plt.subplots(2,2,figsize=(11.3,7.2),sharex=True,layout='constrained')
for i,col in enumerate(COLORS,1):
    def get(k):return np.array([r[k] for r in rr],float)
    axs[0,0].plot(tm,get(f'tug{i}_fraction'),color=col,lw=1.3,label=f'Tug {i}')
    axs[0,1].plot(tm,get(f'tug{i}_thrust_N')/1e6,color=col,lw=1.3)
    axs[1,0].plot(tm,get(f'tug{i}_offset_deg'),color=col,lw=1.2)
    force=(get(f'push{i}_N') if i<=2 else get(f'tow{i}_N'))/1e3
    axs[1,1].plot(tm,force,color=col,lw=1.3)
for a,y,ttl in zip(axs.flat,
                   ['Normalized command','Signed propulsor thrust (MN)','Azimuth offset (deg)','Contact / towline load (kN)'],
                   ['(a) NMPC command','(b) Actual propulsion','(c) Propulsion direction','(d) Transmitted interaction force']):
    a.set_ylabel(y); a.set_title(ttl,loc='left',color=NAVY,pad=8,fontweight='bold')
    a.grid(alpha=.2,lw=.5);a.set_xlabel('Time (min)')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.075),
           ncol=4,frameon=False)
save(fig,'Fig09_TugActuationAndLoads')

# Figure 10: near-quay metrics in the hard case; derived only from logged data
rr=R['270/270'];tm=np.array([r['time_s'] for r in rr])/60
err=np.array([r['position_error_m'] for r in rr]);tm_s=np.array([r['time_s'] for r in rr]);
progress= np.interp(tm_s-200,tm_s,err,left=err[0]) - err
fig,axs=plt.subplots(2,1,figsize=(9.6,5.15),sharex=True,layout='constrained')
axs[0].plot(tm,[r['hull_gap_m'] for r in rr],color=BLUE,lw=1.65,label='Minimum hull clearance')
axs[0].plot(tm,err,color=TEAL,lw=1.5,label='Position error')
axs[0].set_ylabel('Distance (m)')
axs[0].legend(loc='upper right',frameon=False,ncol=2)
axs[1].plot(tm,progress,color=ORANGE,lw=1.45,label='Change in position error over preceding 200 s')
axs[1].axhline(0,ls=':',lw=1.0,color=GREY)
axs[1].set(xlabel='Time (min)',ylabel='Error reduction (m / 200 s)')
for ax in axs:ax.grid(alpha=.22,lw=.5)
axs[0].set_title('(a) Terminal geometry and tracking error',loc='left',fontweight='bold',color=NAVY)
axs[1].set_title('(b) Measured progress indicator (computed from the logged trajectory)',
                 loc='left',fontweight='bold',color=NAVY)
save(fig,'Fig10_ApproachProgress')

# Separate graphical abstract: simple and accurate, no external images
fig,ax=plt.subplots(figsize=(12.5,5.0));ax.set_xlim(0,12.5);ax.set_ylim(0,5);ax.axis('off')
for bx,w in [(.2,3.7),(4.25,4.0),(8.60,3.55)]:
    ax.add_patch(FancyBboxPatch((bx,.55),w,3.9,boxstyle='round,pad=.08',
                                edgecolor='#B2C3CD',facecolor='#F4F8FA',lw=1.4))
# left hull+tugs
ax.add_patch(Polygon(ship*.010 + np.array([2,2.35]),facecolor='#B7C8D5',edgecolor=NAVY,lw=1.5))
for j,(x,y) in enumerate([(1.45,2.79),(2.55,2.79),(1.11,3.12),(2.89,3.12)]):
    ax.add_patch(Polygon(tug*.013+np.array([x,y]),facecolor=COLORS[j],edgecolor=NAVY,lw=.9))
ax.text(2.05,3.9,'Coupled ship–tug dynamics',ha='center',color=NAVY,fontweight='bold',fontsize=11)
ax.text(2.05,1.2,'Two push + two tow',ha='center',color=GREY,fontsize=10)
# middle controls
ax.text(6.25,3.9,'Independent local NMPC',ha='center',color=NAVY,fontweight='bold',fontsize=11)
for i,(x,y) in enumerate([(5.0,2.9),(7.0,2.9),(5.0,1.9),(7.0,1.9)]):
    ax.add_patch(FancyBboxPatch((x-.72,y-.27),1.44,.54,boxstyle='round,pad=.04',
                                facecolor=COLORS[i],edgecolor='none'))
    ax.text(x,y,f'Tug {i+1}',ha='center',va='center',color='white',fontsize=10,fontweight='bold')
ax.text(6.25,1.25,'Shared ship state · no peer commands',ha='center',color=GREY,fontsize=9)
# right simple values
ax.text(10.4,3.9,'Directional evaluation',ha='center',color=NAVY,fontweight='bold',fontsize=11)
ax.text(10.4,2.7,'16 / 16',ha='center',color=TEAL,fontweight='bold',fontsize=28)
ax.text(10.4,2.15,'tested cases reached the terminal envelope',ha='center',color=NAVY,fontsize=8.8)
ax.text(10.4,1.45,'37.6–96.1 min',ha='center',color=GREY,fontsize=11)
for a,b in [((3.95,2.5),(4.20,2.5)),((8.28,2.5),(8.52,2.5))]:
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=2,color=TEAL))
save(fig,'GraphicalAbstract')
print('Generated figures:',len(list(FIG.glob('*.pdf'))),'PDF and',len(list(FIG.glob('*.png'))),'PNG')
