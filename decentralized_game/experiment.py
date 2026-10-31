"""Physical four-tug experiment, separate from all four decision makers.

The environment must see every plant state in order to solve the coupled DAE;
no agent is passed this environment or any other agent's state/action.
"""
from __future__ import annotations
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from geom.constraints import Constraints
from model.ship import Ship
from model.tug import Tug
from simulation.simulator import Sim
from .types import TugObservation, wrap
from .local_agent import AgentConfig, LocalGameAgent
from .equilibrium import DistributedPotentialPlayer, DistributedGameSession
from .tow_safety import free_side_from_quay, lead_error, heading_error, tug_corners


@dataclass
class ExperimentConfig:
    game_mode: str = 'local_best_response'  # or potential_equilibrium
    max_equilibrium_sweeps: int = 30
    equilibrium_tol: float = 1e-7
    duration: float = 300.
    dt: float = .1
    x0: float = 0.
    y0: float = 0.
    psi0: float = 0.
    target_x: float = 0.
    target_y: float = -60.
    target_psi: float = 0.
    wind: float = 0.
    wind_direction_deg: float = 0.
    current: float = 0.
    current_direction_deg: float = 0.
    tug_bp_tf: float = 100.
    usable_fraction: float = .9
    ramp_seconds: float = 20.
    game_period: float = 10.
    lookahead: float = 90.
    azimuth_rate_deg_s: float = .35
    max_speed: float = .17
    disable_push_stabilizer: bool = False
    tow_rope_lead_limit_deg: float = 50.
    tow_heading_limit_deg: float = 75.
    min_tow_hull_clearance_m: float = 2.
    # Halts on a hull incursion rather than generating physically invalid frames.
    stop_on_tow_hull_incursion: bool = True
    stop_on_tow_rope_lead_violation: bool = True
    # A tracking aid inherited from the original model; not a centralized ship controller.
    # It is NOT part of the agent's local action and is disclosed in the report.
    random_seed: int = 7
    output: str = 'results/smoke'
    no_plots: bool = False


def contact_y(hull, x, side=1):
    points=np.asarray(hull.get_y(float(x)),float)
    if not len(points): raise ValueError('No hull section at x=%f'%x)
    return float(np.max(points[:,1]) if side==1 else np.min(points[:,1]))


def hull_gap(hull_points, pose, wall, normal):
    x,y,psi=pose
    rotation=np.array([[np.cos(psi),-np.sin(psi)], [np.sin(psi),np.cos(psi)]])
    points=hull_points@rotation.T+[x,y]
    distances=(wall-points)@normal
    idx=int(np.argmin(distances))
    return float(distances[idx]),points[idx]


def fender_reaction(ship,state,hull,wall,normal):
    gap,point=hull_gap(hull,state[:3],wall,normal)
    ship.external_tau=np.zeros(3)
    if gap>=0:return gap,0.
    x,y,psi,u,v,r,ue,ve=state
    rx,ry=point-np.array([x,y])
    velocity_normal=float(np.dot([ue-r*ry,ve+r*rx],normal))
    magnitude=float(min(2e6,max(0.,2.5e5*(-gap)+6e6*max(0.,velocity_normal))))
    force_world=-magnitude*normal
    c,s=np.cos(psi),np.sin(psi)
    body_rot=np.array([[c,s],[-s,c]])
    force_body=body_rot@force_world
    arm_body=body_rot@np.array([rx,ry])
    ship.external_tau=np.array([*force_body,float(np.cross(arm_body,force_body))])
    return gap,magnitude


def make_observation(t,state,own,applied_thrust,applied_delta,target,gap):
    return TugObservation(time=float(t),ship_pose=tuple(map(float,state[:3])),
         ship_body_velocity=tuple(map(float,state[3:6])),
         ship_earth_velocity=tuple(map(float,state[6:8])),
         own_tug_state=tuple(map(float,own)),
         own_applied_thrust=float(applied_thrust),
         own_applied_azimuth=float(applied_delta),
         target_pose=tuple(map(float,target)),near_wall_gap_m=float(gap))


def _plot(records, target, wall, normal, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=np.array([r['t'] for r in records])
    x=np.array([r['x'] for r in records]); y=np.array([r['y'] for r in records]);
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(x,y,label='Ship center');ax.plot(x[0],y[0],'go',label='Start')
    ax.plot(target[0],target[1],'rx',ms=9,label='Target')
    along=np.array([np.cos(target[2]),np.sin(target[2])]);
    line=np.array([wall-150*along,wall+150*along]);ax.plot(line[:,0],line[:,1],'k--',label='Quay')
    ax.axis('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.grid(alpha=.25);ax.legend();
    fig.tight_layout();fig.savefig(output/'trajectory.png',dpi=160);plt.close(fig)
    for name,cols in [('motion',['ue','ve','psi_deg']),
                      ('forces',['thrust1_N','thrust2_N','thrust3_N','thrust4_N']),
                      ('contact',['push1_normal_N','push2_normal_N','tow3_tension_N','tow4_tension_N']),
                      ('safety',['fender_gap_m','approach_speed_mps'])]:
        fig,ax=plt.subplots(figsize=(9,4))
        for col in cols:ax.plot(a,[r[col] for r in records],label=col)
        ax.set_xlabel('Time (s)');ax.grid(alpha=.3);ax.legend(loc='best');fig.tight_layout()
        fig.savefig(output/(name+'.png'),dpi=160);plt.close(fig)


def run(config: ExperimentConfig):
    if config.game_mode not in ('local_best_response','potential_equilibrium'):
        raise ValueError('Invalid game_mode')
    if config.max_equilibrium_sweeps<1 or config.equilibrium_tol<=0:
        raise ValueError('Invalid equilibrium solver settings')
    if config.duration <= 0 or config.dt <= 0 or config.game_period < config.dt:
        raise ValueError('Invalid simulation time settings')
    if config.ramp_seconds<0:raise ValueError('ramp_seconds must be nonnegative')
    out=Path(config.output);out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    hull_obj=Constraints(str(root/'geom'/'waterline_ship.dxf'))
    hull=np.asarray(hull_obj.points,float)
    ship=Ship();tugs=[Tug() for _ in range(4)]
    for tug in tugs:
        tug.bollard_pull_N=config.tug_bp_tf*9806.65
        tug.command_force_limit_N=tug.bollard_pull_N*config.usable_fraction
    max_forces=[t.command_force_limit_N for t in tugs]
    state=np.array([config.x0,config.y0,np.deg2rad(config.psi0),0.,0.,0.,0.,0.],float)
    target=np.array([config.target_x,config.target_y,np.deg2rad(config.target_psi)],float)
    n=np.array([-np.sin(target[2]),np.cos(target[2])]);
    if (target[:2]-state[:2])@n<0:n=-n
    # This is a MISSION CONSTANT, shared as chart knowledge, not an action of
    # any centralized controller. The renderer reuses exactly this sign.
    free_side=free_side_from_quay(float(target[2]),n)
    # Keep physically distinct work stations: 0.2 Lpp separates push and tow.
    positions=[-.30*ship.Lpp,.30*ship.Lpp,-.50*ship.Lpp,.50*ship.Lpp]
    psi0=state[2]
    local_states=[]
    for x in positions:
        y=contact_y(hull_obj,x,free_side)
        local_states.append(dict(lp=float(np.hypot(x,y)),psi_p=float(np.arctan2(y,x)),
               u=0.,v=0.,r=0.,psi=wrap(psi0-free_side*np.pi/2),
               theta_dot=0.,theta=0.,ue=0.,ve=0.))
    wall=target[:2]+n*(.5*np.ptp(hull[:,1])+6.)
    env=(config.wind,np.deg2rad(config.wind_direction_deg),
         config.current,np.deg2rad(config.current_direction_deg))
    agent_conf=AgentConfig(decision_period_s=config.game_period,lookahead_s=config.lookahead,
      delta_rate_deg_s=config.azimuth_rate_deg_s,desired_speed_mps=config.max_speed,
      tow_rope_lead_limit_deg=config.tow_rope_lead_limit_deg,
      tow_heading_limit_deg=config.tow_heading_limit_deg)
    agents=[LocalGameAgent(i,'push' if i<2 else 'tow',max_forces[i],agent_conf,env,free_side=free_side) for i in range(4)]
    plant=Sim(config.dt,ship,push_Tugs=tugs[:2],tow_Tugs=tugs[2:],
              delta_rate=np.deg2rad(config.azimuth_rate_deg_s),enable_tug_wind_shadow=True)
    # Plant-only states. Never make this container visible to LocalGameAgent.
    applied_deltas=[a.base_azimuth for a in agents]
    applied_thrusts=[0.]*4
    rows=[];decisions=[];equilibria=[];unilateral=[];potential_rounds=[]
    from shapely.geometry import Polygon
    ship_poly=Polygon(hull).buffer(0)
    stop_reason=None
    worst_clearance=float('inf')
    n_steps=int(round(config.duration/config.dt))
    for k in range(n_steps):
        t=k*config.dt
        gap,fender=fender_reaction(ship,state,hull,wall,n)
        own=[]
        for i,s in enumerate(local_states):
            own.append(np.array([s['lp'],s['psi_p'],s['u'],s['v'],s['r'],s['psi'],
                 *([s['theta_dot'],s['theta']] if i>=2 else [])],float))
        # Independent sense-decide calls, no other tug's state or action in observations.
        # Simulated scheduler is sequential for reproducibility; all observations are
        # captured BEFORE any decision, so update is effectively synchronous.
        observations=[make_observation(t,state,own[i],applied_thrusts[i],
                applied_deltas[i],target,gap) for i in range(4)]
        for agent,obs in zip(agents,observations):agent.observe(obs)
        if config.game_mode=='potential_equilibrium':
            if t-agents[0].last_decision_time>=config.game_period-1e-7:
                # Shared ship NAVIGATION estimate, obtained from observations.
                # All player observers see the same ship time series. Neither
                # private tug state nor co-player control is read by this source.
                public_acc=np.asarray(agents[0].measured_acc,float).copy()
                try:
                    players=[DistributedPotentialPlayer(i,
                        a.prepare_equilibrium_bids(observations[i],public_acc),
                        observations[i],public_acc,
                        lookahead_s=config.lookahead,
                        desired_speed_mps=config.max_speed,
                        max_near_speed_mps=agent_conf.max_near_speed_mps,
                        safety_near_gap_m=agent_conf.safety_near_gap_m)
                        for i,a in enumerate(agents)]
                except (ValueError,np.linalg.LinAlgError) as exc:
                    stop_reason=f'No admissible local stage game at t={t:.1f}s: {exc}'
                    if not rows:raise RuntimeError(stop_reason) from exc
                    break
                session=DistributedGameSession(players)
                eq=session.solve(config.max_equilibrium_sweeps,config.equilibrium_tol)
                equilibria.append(dict(time_s=t,equilibrium=eq['equilibrium'],
                   max_unilateral_gain=eq['max_unilateral_gain'],
                   sweeps=eq['sweeps'],moves=eq['moves'],potential=eq['potential'],
                   agent1_gain=eq['per_agent_unilateral_gain'][0],
                   agent2_gain=eq['per_agent_unilateral_gain'][1],
                   agent3_gain=eq['per_agent_unilateral_gain'][2],
                   agent4_gain=eq['per_agent_unilateral_gain'][3]))
                # Offline *audit only*: enumerate each frozen-game player's
                # unilateral costs AFTER negotiation; never used as a controller.
                for p in players:
                    selected_potential=p._stage_cost(p.selected)
                    for j in p.feasible_ids:
                        unilateral.append(dict(time_s=t,agent_id=p.id+1,
                             selected_action_index=p.selected,candidate_action_index=j,
                             selected_potential=selected_potential,
                             deviated_potential=p._stage_cost(j),
                             unilateral_gain=selected_potential-p._stage_cost(j)))
                for sweep,score in enumerate(eq['potential_by_sweep'],1):
                    potential_rounds.append(dict(time_s=t,sweep=sweep,potential=score))
                # Commit only commands selected privately by each player.
                for i,(agent,p) in enumerate(zip(agents,players)):
                    bid=p.bids[p.selected]
                    agent.action=bid.action
                    agent.last_cost=eq['potential']
                    agent.last_decision_time=t
                    agent.decision_count+=1
                    decisions.append(dict(time_s=t,agent_id=i+1,mode=agent.mode,
                        thrust_N=bid.action.thrust,
                        azimuth_deg=np.rad2deg(bid.action.azimuth),
                        local_utility_cost=bid.local_cost,
                        local_predicted_approach_mps=float('nan'),
                        observed_earth_acc_x=agent.measured_acc[0],
                        observed_earth_acc_y=agent.measured_acc[1],
                        observed_yaw_acc=agent.measured_acc[2],
                        cumulative_local_evaluations=agent.proposals_evaluated,
                        rejected_tow_actions=agent.rejected_tow_actions,
                        tow_no_feasible_action=agent.tow_no_feasible_action,
                        tow_lead_error_deg=agent.last_tow_lead_error_deg,
                        equilibrium_unilateral_gain=eq['per_agent_unilateral_gain'][i],
                        equilibrium_stage_converged=eq['equilibrium']))
        else:
            for i,agent in enumerate(agents):
                last_count=agent.decision_count
                agent.decide(observations[i])
                if agent.decision_count>last_count:
                    decisions.append(dict(time_s=t,agent_id=i+1,mode=agent.mode,
                       thrust_N=agent.action.thrust,azimuth_deg=np.rad2deg(agent.action.azimuth),
                       local_utility_cost=agent.last_cost,
                       local_predicted_approach_mps=agent.last_predicted_approach_speed,
                       observed_earth_acc_x=agent.measured_acc[0],
                       observed_earth_acc_y=agent.measured_acc[1],
                       observed_yaw_acc=agent.measured_acc[2],
                       cumulative_local_evaluations=agent.proposals_evaluated,
                       rejected_tow_actions=agent.rejected_tow_actions,
                       tow_no_feasible_action=agent.tow_no_feasible_action,
                       tow_lead_error_deg=agent.last_tow_lead_error_deg))
        ramp_x=min(1.,t/max(1e-9,config.ramp_seconds))
        ramp=ramp_x*ramp_x*(3.-2.*ramp_x)
        inputs=[(a.action.thrust*ramp,a.action.azimuth) for a in agents]
        sol,applied_deltas,block_sizes,_=plant.solve_one_step(
             [state[3],state[4],state[5],state[2],*env],own[:2],own[2:],inputs,applied_deltas)
        sb,pb,tb=Sim.split_solution(sol,block_sizes,2)
        state[3:6]+=config.dt*sb[:3]
        state[2]=wrap(state[2]+config.dt*state[5])
        state[6:8]+=config.dt*sb[3:5]
        state[:2]+=config.dt*state[6:8]
        for i,b in enumerate(pb+tb):
            s=local_states[i]
            s['u']+=config.dt*b[0];s['v']+=config.dt*b[1];s['r']+=config.dt*b[2]
            s['psi']=wrap(s['psi']+config.dt*s['r'])
            s['ue']+=config.dt*b[3];s['ve']+=config.dt*b[4]
            if i<2 and not config.disable_push_stabilizer:
                # INHERITED tracking aid: an artificial push-tug heading stabilizer
                # from upstream model. Disabled via --disable-push-stabilizer.
                desired=state[5]+np.clip(wrap(state[2]-free_side*np.pi/2-s['psi'])/10.,
                                   -np.deg2rad(.6),np.deg2rad(.6))
                s['r']+=config.dt/4.*(desired-s['r'])
                s['psi']=wrap(s['psi']+config.dt*s['r'])
            if i>=2:
                s['theta_dot']+=config.dt*b[5]
                s['theta']=wrap(s['theta']+config.dt*s['theta_dot'])
        # These forces are diagnostics from the complete PLANT ONLY; never exposed to agents.
        push_normal=[-float(b[5]) for b in pb]
        push_tangent=[-float(b[6]) for b in pb]
        tow_tension=[float(b[6]) for b in tb]
        applied_thrusts=[inp[0] for inp in inputs]
        # Plant-only geometry telemetry: measured tug centre, hull contact and rope
        # anchor. Only written to the output CSV AFTER control decisions; never
        # made accessible through TugObservation or LocalGameAgent.
        R=np.array([[np.cos(state[2]),-np.sin(state[2])],
                    [np.sin(state[2]), np.cos(state[2])]])
        tug_geometry={}
        tow_clearances=[];tow_overlaps=[]
        for i,s in enumerate(local_states):
            kind='push' if i<2 else 'tow'
            local=np.array([s['lp'],s['psi_p'],s['u'],s['v'],s['r'],s['psi'],
                            *([s['theta_dot'],s['theta']] if i>=2 else [])],float)
            relative=plant.coupled._tug_center_in_ship_body(kind,local,tugs[i],state[2])
            center=state[:2]+R@relative
            contact=state[:2]+R@np.array([s['lp']*np.cos(s['psi_p']),
                                            s['lp']*np.sin(s['psi_p'])])
            # For a tow the tug-side rope attachment is ati forward of the tug
            # centre. A push has no rope and a single contact point.
            attach=center+(tugs[i].ati*np.array([np.cos(s['psi']),np.sin(s['psi'])])
                           if kind=='tow' else np.zeros(2))
            j=i+1
            if kind=='tow':
                from shapely.geometry import Polygon
                # Geometry evaluated in ship coordinates (isometry-invariant).
                tug_poly=Polygon(tug_corners(relative,s['psi']-state[2]))
                clearance=float(ship_poly.distance(tug_poly))
                overlap=bool(ship_poly.intersects(tug_poly))
                tow_clearances.append(clearance)
                tow_overlaps.append(overlap)
                tug_geometry.update({f'tug{j}_rope_lead_error_deg':
                   float(np.rad2deg(lead_error(state[2],s['psi'],s['theta'],free_side))),
                   f'tug{j}_heading_error_deg':
                   float(np.rad2deg(heading_error(state[2],s['psi'],free_side))),
                   f'tug{j}_hull_clearance_m':clearance,
                   f'tug{j}_hull_overlap':int(overlap)})
            tug_geometry.update({f'tug{j}_x_m':float(center[0]),
                f'tug{j}_y_m':float(center[1]),
                f'tug{j}_psi_deg':float(np.rad2deg(s['psi'])),
                f'tug{j}_contact_x_m':float(contact[0]),
                f'tug{j}_contact_y_m':float(contact[1]),
                f'tug{j}_rope_anchor_x_m':float(attach[0]),
                f'tug{j}_rope_anchor_y_m':float(attach[1])})
        row=dict(t=t+config.dt,x=float(state[0]),y=float(state[1]),
          psi_deg=float(np.rad2deg(state[2])),u=float(state[3]),v=float(state[4]),
          r_deg_s=float(np.rad2deg(state[5])),ue=float(state[6]),ve=float(state[7]),
          fender_gap_m=float(gap),fender_force_N=float(fender),
          approach_speed_mps=float(state[6:8]@n),
          push1_normal_N=push_normal[0],push2_normal_N=push_normal[1],
          push1_tangent_N=push_tangent[0],push2_tangent_N=push_tangent[1],
          tow3_tension_N=tow_tension[0],tow4_tension_N=tow_tension[1],
          **{f'thrust{i+1}_N':float(inputs[i][0]) for i in range(4)},
          **{f'azimuth{i+1}_deg':float(np.rad2deg(applied_deltas[i])) for i in range(4)},
          **tug_geometry)
        rows.append(row)
        worst_clearance=min(worst_clearance,*tow_clearances)
        if config.stop_on_tow_rope_lead_violation and any(
                abs(tug_geometry[f'tug{i}_rope_lead_error_deg']) > config.tow_rope_lead_limit_deg
                for i in (3,4)):
            stop_reason=(f'Tow rope-lead safety envelope exceeded at t={t+config.dt:.1f} s: '
               +', '.join(f'T{i}={tug_geometry[f"tug{i}_rope_lead_error_deg"]:+.2f} deg'
                           for i in (3,4)))
            break
        if config.stop_on_tow_hull_incursion and any(
                overlap or clearance < config.min_tow_hull_clearance_m
                for overlap,clearance in zip(tow_overlaps,tow_clearances)):
            stop_reason=f'Tow hull-clearance guard triggered at t={t+config.dt:.1f} s: '+(
               ', '.join(f'T{i+3}={tow_clearances[i]:.2f}m' for i in range(2)))
            break
        if not np.all(np.isfinite(state)) or max(abs(state[0]),abs(state[1]))>1e5:
            raise RuntimeError(f'Coupled DAE diverged at t={t} s')
    with (out/'trajectory.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'agent_decisions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(decisions[0]));w.writeheader();w.writerows(decisions)
    if config.game_mode=='potential_equilibrium' and equilibria:
        with (out/'game_equilibrium.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(equilibria[0]));w.writeheader();w.writerows(equilibria)
    if config.game_mode=='potential_equilibrium':
        for basename,items in [('game_unilateral_deviations.csv',unilateral),
                               ('game_potential_sweeps.csv',potential_rounds)]:
            if items:
                with (out/basename).open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    last=rows[-1];hold=rows[-max(1,int(round(5./config.dt))):]
    passes_hold=bool(stop_reason is None and all(
          np.hypot(r['x']-target[0],r['y']-target[1])<=2.5
          and abs(np.rad2deg(wrap(np.deg2rad(r['psi_deg'])-target[2])))<=2.
          and np.hypot(r['ue'],r['ve'])<=.045
          and abs(r['r_deg_s'])<=.035 for r in hold))
    contact_violations=sum(1 for r in rows if (r['push1_normal_N'] < -1000
        or r['push2_normal_N'] < -1000 or r['tow3_tension_N'] < -1000
        or r['tow4_tension_N'] < -1000))
    friction_violations=sum(1 for r in rows if any(
        abs(r[f'push{i}_tangent_N'])> .15*max(0,r[f'push{i}_normal_N'])+1e3
        for i in (1,2)))
    near_speed_violations=sum(r['fender_gap_m']<45 and r['approach_speed_mps']>.095 for r in rows)
    collision_violations=sum(r['fender_gap_m']<-.01 for r in rows)
    tow_side_violations=sum(1 for r in rows if any(
        abs(r[f'tug{i}_rope_lead_error_deg'])>config.tow_rope_lead_limit_deg
        for i in (3,4)))
    tow_hull_violations=sum(1 for r in rows if any(
        r[f'tug{i}_hull_overlap'] or r[f'tug{i}_hull_clearance_m']<config.min_tow_hull_clearance_m
        for i in (3,4)))
    stats=dict(method=('finite_exact_potential_distributed_strict_best_response' if config.game_mode=='potential_equilibrium' else 'decentralized_belief_based_local_best_response_dynamic_game'),
       game_mode=config.game_mode,
       stage_game_equilibria=int(sum(e['equilibrium'] for e in equilibria)),
       stage_games_solved=len(equilibria),
       stage_equilibrium_rate=(float(np.mean([e['equilibrium'] for e in equilibria])) if equilibria else None),
       max_observed_unilateral_gain=(float(max(e['max_unilateral_gain'] for e in equilibria)) if equilibria else None),
       max_equilibrium_sweeps=config.max_equilibrium_sweeps,
       no_rl=True,no_force_allocator=True,
       decision_information='own_tug_state + ship_kinematics + berth_goal + inferred_acceleration',
       agents=4,models_per_agent='one_ship_one_tug_coupled_DAE',
       coupled_four_tug_plant=True,initial=asdict(config),
       free_side_ship_body=free_side,
       actual_simulated_duration_s=float(rows[-1]['t']),
       simulation_stop_reason=stop_reason,
       tow_min_hull_clearance_m=float(worst_clearance),
       tow_hull_incursion_samples=tow_hull_violations,
       tow_rope_lead_violation_samples=tow_side_violations,
       tow_rejected_candidate_actions=[a.rejected_tow_actions for a in agents],
       tow_no_feasible_decisions=[a.tow_no_feasible_action for a in agents],
       tow_safety_guard_status='halt_on_unsafe_geometry; NOT a proven invariant controller',
       final_position_error_m=float(np.linalg.norm(state[:2]-target[:2])),
       final_heading_error_deg=float(abs(np.rad2deg(wrap(state[2]-target[2])))),
       final_speed_mps=float(np.linalg.norm(state[6:8])),
       final_yaw_rate_deg_s=float(np.rad2deg(state[5])),
       max_approach_speed_mps=float(max(0,max(r['approach_speed_mps'] for r in rows))),
       final_fender_gap_m=last['fender_gap_m'],
       negative_push_or_rope_samples=contact_violations,
       push_friction_cone_violation_samples=friction_violations,
       near_quay_approach_speed_violation_samples=int(near_speed_violations),
       fender_penetration_samples=int(collision_violations),
       terminal_5s_hold=passes_hold,
       conservative_safe_success=bool(stop_reason is None and tow_hull_violations==0
          and tow_side_violations==0 and passes_hold and contact_violations==0
          and friction_violations==0 and near_speed_violations==0 and collision_violations==0),
       decisions_per_agent=[a.decision_count for a in agents],
       local_DAE_evaluations_per_agent=[a.proposals_evaluated for a in agents],
       safety_caveat='Action filtering and halt-on-incursion do not guarantee physical safety; original bilateral tow DAE still permits negative tension; no unilateral contact or collision impulse solver.',
       theory_caveat=('Pure Nash certificates refer only to each fixed finite identical-interest surrogate stage game, not continuous ship-tug dynamics or closed-loop stability.' if config.game_mode=='potential_equilibrium' else 'Approximate game best responses with local beliefs; no Nash convergence or formal safety guarantee.'),
       push_tug_tracking_aid_enabled=not config.disable_push_stabilizer)
    (out/'metrics.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'configuration.json').write_text(json.dumps(asdict(config),indent=2),encoding='utf-8')
    if not config.no_plots:_plot(rows,target,wall,n,out)
    return stats
