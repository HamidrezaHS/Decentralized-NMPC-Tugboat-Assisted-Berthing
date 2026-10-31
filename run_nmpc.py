"""Reproducible four-agent decentralized NMPC evaluation on full coupled plant.

Each controller receives *only* the public ship state, its own tug state,
and its own actuator readings. The runner combines the four independently
selected decisions solely as the physical simulator's actuation input.
No central maneuver planner, generalized-force allocator, inter-tug messages,
MARL network or game-theoretic solver is used in this stage.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import math
import numpy as np
from marl_game.env import EnvConfig, BerthingGameEnv
from decentralized_nmpc import NMPCConfig,LocalNMPC,common_ship_observation
from decentralized_nmpc.coordination_baselines import JointNMPC,CoordinationSettings
from contact_local import ContactNMPC,ContactNMPCConfig,contact_observation
from ship_state_local import ShipStateLocalSensor, ship_state_policy_vector, SHIP_STATE_VECTOR_COLUMNS

CONTROLLER_VERSION='oe-v4.6-coriolis-corrected+ship-state-env-reserve-v3.1'
DEFAULT_EMERGENCY_SIM_SECONDS = 21600.0  # Six simulated hours; 0 disables the cap


def _atomic_csv(path: Path, rows: list[dict]):
    """Persist complete logged observations, including initial rows with blank fields."""
    if not rows:
        return
    tmp=path.with_name(path.name+'.tmp')
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with tmp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _save_progress(out: Path, rows: list[dict], decisions: list[dict], status: str, sim_time: float):
    """Save completed history. This is not a restartable simulator checkpoint."""
    _atomic_csv(out/'trajectory.csv',rows)
    _atomic_csv(out/'local_nmpc_decisions.csv',decisions)
    progress={'status':status,'simulation_time_s':float(sim_time),
              'recorded_states':len(rows),'recorded_decisions':len(decisions),
              'note':'Diagnostic snapshot only; restart requires a new simulation.'}
    path=out/'progress.json'
    tmp=out/'progress.json.tmp'
    tmp.write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def rollout(out='results/nmpc_h20',psi0=20.,max_seconds=DEFAULT_EMERGENCY_SIM_SECONDS,decision_seconds=10.,dt=.2,
            seed=123,render=True,horizon=90.,maxiter=10,verbose=True,noise=0.,
            target_x=50.,target_y=-100.,target_psi_deg=0.,start_x=0.,start_y=0.,
            wind_mps=0.,wind_to_deg=0.,current_mps=0.,current_to_deg=0.,
            gust_mps=0.,gust_period_s=360.,current_osc_mps=0.,
            disturbance_mode='measured',wind_sensor_std_mps=0.,
            current_sensor_std_mps=0.,position_tolerance_m=2.5,
            precision=True,max_fraction_step=.38,max_offset_step_deg=24.,slew_weight=.10,
            anti_stall_gain=1.3,anti_stall_window_s=200.,save_every_seconds=300.,
            guidance_mode='mission',
            controller_method='decentralized',environmental_reserve='auto',push_heading_guard=True,admm_rounds=3,admm_rho=.5,
            admm_inner_iterations=4,post_success_seconds=0.,
            eval_position_jitter_m=0.,eval_heading_jitter_deg=0.,
            physics_mode='energy_consistent',export_local_demos=False,export_ship_state_demos=False):
    if physics_mode not in ('energy_consistent','legacy'):
        raise ValueError('physics_mode must be energy_consistent or legacy')
    from model.ship import Ship
    Ship.DEFAULT_CORIOLIS_MODE = physics_mode
    if guidance_mode not in ('mission','legacy'):
        raise ValueError('guidance_mode must be mission or legacy')
    if abs(target_psi_deg)>1e-8:
        raise ValueError('The current plant models a globally east-west quay; target_psi_deg must be 0 for physical comparisons')
    if environmental_reserve not in ('auto','on','off'):
        raise ValueError('environmental_reserve must be auto, on or off')
    if controller_method not in ('decentralized','ship_state_local','central','admm','contact_local'):
        raise ValueError('controller_method must be decentralized, ship_state_local, central, admm or contact_local')
    if controller_method in ('central','admm') and (wind_sensor_std_mps>0 or current_sensor_std_mps>0):
        raise ValueError('Matched joint-controller sensor noise is not yet implemented; use zero sensor noise for controlled comparisons')
    if admm_rounds<1 or admm_inner_iterations<1:
        raise ValueError('ADMM rounds and inner iterations must be positive')
    if max_seconds < 0 or not np.isfinite(max_seconds):
        raise ValueError('max_seconds must be finite and nonnegative (0 means no simulation-time cap)')
    if save_every_seconds < 0 or not np.isfinite(save_every_seconds):
        raise ValueError('save_every_seconds must be finite and nonnegative')
    if post_success_seconds < 0 or not np.isfinite(post_success_seconds):
        raise ValueError('post_success_seconds must be finite and nonnegative')
    if any(not np.isfinite(z) or z < 0 for z in (eval_position_jitter_m, eval_heading_jitter_deg)):
        raise ValueError('evaluation jitter parameters must be finite nonnegative')
    if post_success_seconds>0 and abs(post_success_seconds/dt-round(post_success_seconds/dt))>1e-6:
        raise ValueError('post_success_seconds must be a multiple of the physical time step')
    output=Path(out)
    output.mkdir(parents=True,exist_ok=True)
    # Zero means success/safety only. Use a very large internal clock guard
    # rather than changing EnvConfig semantics for other MARL applications.
    effective_env_limit=(max_seconds+post_success_seconds+decision_seconds
                         if max_seconds>0 else 1.0e12)
    config=EnvConfig(train=False,initial_psi_deg=psi0,max_seconds=effective_env_limit,
                     dt=dt,decision_seconds=decision_seconds,seed=seed,observation_noise=noise,
                     post_success_seconds=post_success_seconds,
                     eval_position_jitter_m=eval_position_jitter_m,
                     eval_heading_jitter_deg=eval_heading_jitter_deg,
                     target_x=target_x,target_y=target_y,target_psi_deg=target_psi_deg,
                     eval_start_x_m=start_x,eval_start_y_m=start_y,
                     stop_on_unsafe=True,hold_s=5.,push_heading_guard=(push_heading_guard and controller_method!='contact_local'),
                     contact_only=(controller_method=='contact_local'),
                     eval_wind_mps=wind_mps,eval_wind_direction_deg=wind_to_deg,
                     eval_current_mps=current_mps,eval_current_direction_deg=current_to_deg)
    env=BerthingGameEnv(config)
    env.reset()
    # Online success tolerance; identical in monitor and reported diagnostics.
    from physical_v4.termination import BerthingMonitor, BerthingCriteria
    env.monitor=BerthingMonitor(BerthingCriteria(position_m=position_tolerance_m,
                                                hold_s=config.hold_s))
    reserve_enabled=(controller_method=='ship_state_local' if environmental_reserve=='auto' else environmental_reserve=='on')
    local_cfg=NMPCConfig(horizon_s=horizon,max_iterations=maxiter,
                         environmental_thrust_reserve=reserve_enabled,
                         disturbance_mode=disturbance_mode,
                         mission_guidance=(guidance_mode=='mission'),
                         max_fraction_step=max_fraction_step,
                         max_offset_step_deg=max_offset_step_deg,
                         slew_weight=slew_weight,
                         anti_stall_gain=anti_stall_gain,
                         anti_stall_window_s=anti_stall_window_s,
                         berth_position_tolerance_m=position_tolerance_m,
                         terminal_position_weight=.45 if precision else 0.,
                         terminal_heading_weight=.08 if precision else 0.,
                         precision_radius_m=25. if precision else 0.)
    sensor_rngs=[np.random.default_rng(seed+10001+37*i) for i in range(4)]
    ship_state_sensors=[ShipStateLocalSensor(i,measure_environment=(disturbance_mode=='measured'),
        wind_sensor_std_mps=wind_sensor_std_mps,current_sensor_std_mps=current_sensor_std_mps,seed=seed)
        for i in range(4)]
    def environment_at(t):
        # Directions are TOWARD in earth-fixed frame. Gust and current
        # variations are deterministic and applied to the physical plant.
        w=max(0.,wind_mps+gust_mps*math.sin(2*math.pi*t/gust_period_s)) if gust_mps else wind_mps
        c=max(0.,current_mps+current_osc_mps*math.sin(2*math.pi*t/gust_period_s)) if current_osc_mps else current_mps
        return (float(w),math.radians(wind_to_deg),float(c),math.radians(current_to_deg))
    env.plant.env=environment_at(0.)
    if controller_method=='contact_local':
        controllers=[ContactNMPC(i,ContactNMPCConfig(
              horizon_s=horizon,max_iterations=maxiter,
              max_fraction_step=max_fraction_step,
              max_offset_step_deg=max_offset_step_deg,
              slew_weight=slew_weight)) for i in range(4)]
        for i,controller in enumerate(controllers):
            controller.configure_berth(env.plant.attach_local[i],env.quay_y,
                                       env.plant._ship_points)
    elif controller_method in ('decentralized','ship_state_local'):
        controllers=[LocalNMPC(i,env.goal,env.hull.points,env.plant.attach_local[i],
                           env.side,env.plant.max_force,local_cfg) for i in range(4)]
    else:controllers=[]
    joint=(JointNMPC(env,local_cfg,CoordinationSettings(method=controller_method,
                 admm_rounds=admm_rounds,admm_rho=admm_rho,
                 admm_inner_iterations=admm_inner_iterations))
           if controller_method in ('central','admm') else None)
    start=time.perf_counter()
    rows=[];decisions=[];local_demos=[];ship_state_demos=[];ret=0.0
    peak_approach_speed=0.
    min_gap=float('inf')
    max_yaw_rate=0.
    max_substep_close_speed=0.
    max_interaction_force_N=0.
    integrated_abs_thrust=0.
    integrated_thrust_change=0.
    last_thrust=np.zeros(4)

    reason='TIMEOUT';stop_cause='MAX_SIM_TIME_GUARD';info={}
    first_success=None
    next_save_time=float(save_every_seconds) if save_every_seconds>0 else float('inf')
    def record(t,command=None):
        sh=env.ship
        d=env.plant.geometry(sh,env.tugs)
        gap=env.hull_gap()
        target_err=float(np.linalg.norm(sh[:2]-env.goal[:2]))
        v_earth=np.array([math.cos(sh[2])*sh[3]-math.sin(sh[2])*sh[4],
                          math.sin(sh[2])*sh[3]+math.cos(sh[2])*sh[4]])
        return dict(time_s=round(float(t),4),x_m=float(sh[0]),y_m=float(sh[1]),
            psi_deg=float(np.rad2deg(sh[2])),u_mps=float(sh[3]),v_mps=float(sh[4]),
            r_deg_s=float(np.rad2deg(sh[5])),
            position_error_m=target_err,
            heading_error_deg=float(abs(np.rad2deg((sh[2]-env.goal[2]+np.pi)%(2*np.pi)-np.pi))),
            hull_gap_m=gap,
            ship_speed_mps=float(np.linalg.norm(sh[3:5])),
            east_speed_mps=float(v_earth[0]),north_speed_mps=float(v_earth[1]),
            approach_speed_mps=max(0.,float(-v_earth[1])),
            wind_mps=float(env.plant.env[0]),wind_to_deg=float(np.rad2deg(env.plant.env[1])),
            current_mps=float(env.plant.env[2]),current_to_deg=float(np.rad2deg(env.plant.env[3])),
            **{f'push{i+1}_heading_error_deg':float(np.rad2deg(abs(                (sh[2]-env.side*np.pi/2-env.tugs[i,2]+np.pi)%(2*np.pi)-np.pi))) for i in range(2)},
            **{f'push{i+1}_guard_interventions':int(env.heading_guards[i].n_interventions) for i in range(2)},
            **{f'tug{i+1}_thrust_N':float(env.act.thrust[i]) for i in range(4)},
            **{f'tug{i+1}_offset_deg':(float(command[i,1])*20. if command is not None else 0.) for i in range(4)},
            **d)
    rows.append(record(0.))
    try:
        step=0
        while True:
            if max_seconds>0 and env.monitor.first_success_time_s is None and env.time >= max_seconds-1.0e-7:
                stop_cause='MAX_SIM_TIME_GUARD'
                break
            # The following loop reads NO peer tug state from any controller's
            # observation, and each controller is called independently.
            env.plant.env=environment_at(env.time)
            if controller_method in ('decentralized','ship_state_local','contact_local'):
                acts=[]
                step_reports=[]
                observed_ship_states=[]
                for i,controller in enumerate(controllers):
                    if controller_method=='contact_local':
                        # Only a point-tracker measurement + own navigation.
                        # The physical plant, not the control policy, uses
                        # global ship state to synthesize these measurements.
                        local=contact_observation(env,i)
                    elif controller_method=='ship_state_local':
                        local=ship_state_sensors[i].read(env)
                    else:
                        local=common_ship_observation(env,i,
                            measure_environment=(disturbance_mode=='measured'),
                            wind_sensor_std_mps=wind_sensor_std_mps,
                            current_sensor_std_mps=current_sensor_std_mps,
                            rng=sensor_rngs[i] if disturbance_mode=='measured' else None)
                    if controller_method=='ship_state_local' or (export_ship_state_demos and controller_method=='decentralized'):
                        observed_ship_states.append(local)
                    tic=time.perf_counter()
                    action,diag=controller.act(local)
                    diag['optimizer_elapsed_ms']=(time.perf_counter()-tic)*1000.
                    acts.append(action)
                    step_reports.append(diag)
                actions=np.stack(acts)
            else:
                tic=time.perf_counter()
                actions,step_reports=joint.act(env,
                       measure_environment=(disturbance_mode=='measured'))
                elapsed_ms=(time.perf_counter()-tic)*1000.
                for diag in step_reports:
                    diag['optimizer_elapsed_ms']=elapsed_ms/4.
            if controller_method=='ship_state_local' or export_ship_state_demos:
                if controller_method not in ('decentralized','ship_state_local'):
                    raise ValueError('Ship-state NMPC demonstrations require an independent ship-state NMPC controller')
                for j,measurement in enumerate(observed_ship_states):
                    features=ship_state_policy_vector(measurement,env.goal)
                    ship_state_demos.append(dict(episode_time_s=float(measurement.time_s),
                        agent=j+1,**{f'obs{k:02d}':float(val) for k,val in enumerate(features)},
                        action_fraction=float((actions[j,0]+1.)*.5),
                        action_offset_deg=float(actions[j,1]*20.),
                        controller_method=controller_method))
            if export_local_demos or controller_method=='contact_local':
                # Record PRE-DECISION point-only observation and the command
                # chosen by either privileged teacher or local controller.
                # Crucially no full vessel state / peer observations in dataset.
                for j in range(4):
                    sample=contact_observation(env,j)
                    local_demos.append(dict(t_s=float(env.time),agent=j+1,
                        **{f'obs{k:02d}':float(val)
                           for k,val in enumerate(sample.policy_vector())},
                        action_fraction=float((actions[j,0]+1.)*.5),
                        action_offset_deg=float(actions[j,1]*20.),
                        controller_method=controller_method))
            _,_,reward,term,trunc,info=env.step(actions)
            ret+=reward
            max_substep_close_speed=max(max_substep_close_speed,float(info.get('max_approach_speed_mps',0.)))
            max_interaction_force_N=max(max_interaction_force_N,float(info.get('max_interaction_N',0.)))
            interval=env.time-(rows[-1]['time_s'])
            f=np.asarray(env.act.thrust,float)
            integrated_abs_thrust+=float(np.sum(np.abs(f)))*interval
            integrated_thrust_change+=float(np.sum(np.abs(f-last_thrust)))
            last_thrust=f.copy()
            min_gap=min(min_gap,float(env.hull_gap()))
            max_yaw_rate=max(max_yaw_rate,float(abs(np.rad2deg(env.ship[5]))))
            earth_vel=np.array([math.cos(env.ship[2])*env.ship[3]-math.sin(env.ship[2])*env.ship[4],
                                math.sin(env.ship[2])*env.ship[3]+math.cos(env.ship[2])*env.ship[4]])
            if env.hull_gap()<45.:peak_approach_speed=max(peak_approach_speed,float(max(0.,-earth_vel[1])))
            row=record(env.time,actions)
            row.update({f'tug{i+1}_fraction':float((actions[i,0]+1)*.5) for i in range(4)})
            row.update({'push1_N':info['push_normal_N'][0],'push2_N':info['push_normal_N'][1],
                        'tow3_N':info['tow_tension_N'][0],'tow4_N':info['tow_tension_N'][1],
                        'reward':reward,'unsafe':int(info['unsafe']),
                        'success':int(info['success']),
                        'post_success':int(env.monitor.first_success_time_s is not None and
                                           env.time>env.monitor.first_success_time_s)})
            rows.append(row)
            for r in step_reports:
                decisions.append(dict(t_s=float(env.time),**r))
            if verbose and (step%20==0 or term or trunc):
                print(f'[NMPC] step={step:4d} t={env.time:7.1f}s epos={info["position_error_m"]:7.2f} m '
                      f'epsi={info["heading_error_deg"]:6.2f} deg gap={info["hull_gap_m"]:6.2f} m '
                      f'fractions={np.round((actions[:,0]+1)*.5,2).tolist()}',flush=True)
            if save_every_seconds>0 and env.time >= next_save_time-1e-7:
                _save_progress(output,rows,decisions,'RUNNING',env.time)
                next_save_time+=save_every_seconds
            if env.monitor.first_success_time_s is not None and first_success is None:
                first_success=float(env.monitor.first_success_time_s)
            if term or trunc:
                if info['unsafe']:
                    reason='UNSAFE';stop_cause='SAFETY_TERMINATION'
                elif post_success_seconds>0 and info.get('stationkeeping_complete'):
                    passed=env.monitor.post_success_violations==0 and not env.monitor.ever_unsafe
                    reason='SUCCESS_HOLD' if passed else 'STATION_KEEPING_FAILED'
                    stop_cause='POST_BERTHING_HOLD_COMPLETED'
                elif info['success']:
                    reason='SUCCESS';stop_cause='SAFE_BERTHING'
                else:
                    reason='TIMEOUT';stop_cause='MAX_SIM_TIME_GUARD'
                break
            step+=1
    except KeyboardInterrupt:
        reason='INTERRUPTED'
        stop_cause='USER_INTERRUPTED'
        print('[NMPC] Interrupted; writing all recorded telemetry before exit.', flush=True)
    elapsed=time.perf_counter()-start
    if not rows:raise RuntimeError('simulation produced no telemetry')
    _save_progress(output,rows,decisions,reason,env.time)
    if local_demos:
        _atomic_csv(output/'local_point_demonstrations.csv',local_demos)
    if ship_state_demos:
        _atomic_csv(output/'local_ship_state_demonstrations.csv',ship_state_demos)
        (output/'local_ship_state_feature_names.json').write_text(
            json.dumps(list(SHIP_STATE_VECTOR_COLUMNS),indent=2),encoding='utf8')
    # Path metrics are independent of the success condition and include timeouts.
    recorded_xy=np.array([[r['x_m'],r['y_m']] for r in rows])
    path_length=float(np.sum(np.linalg.norm(np.diff(recorded_xy,axis=0),axis=1)))
    direct_distance=float(np.linalg.norm(recorded_xy[0]-env.goal[:2]))
    direction=np.sign(env.goal[0]-recorded_xy[0,0])
    if direction==0: direction=1.
    max_target_overshoot=float(max(0.,np.max(direction*(recorded_xy[:,0]-env.goal[0]))))
    first_success_time=env.monitor.first_success_time_s
    audit=env.monitor
    first_success_state=(env.first_success_ship.tolist() if env.first_success_ship is not None else None)
    stationkeeping_complete=(post_success_seconds>0 and first_success_time is not None and
        audit.post_success_elapsed_s >= post_success_seconds-1e-6)
    stationkeeping_passed=(stationkeeping_complete and audit.post_success_violations==0
                          and not audit.ever_unsafe)
    metrics={'controller_version':CONTROLLER_VERSION,'physics_mode':physics_mode,'controller_method':controller_method,'runtime_information_boundary':('own_contact_point_and_own_tug' if controller_method=='contact_local' else 'shared_ship_state_plus_own_tug_only' if controller_method in ('decentralized','ship_state_local') else 'joint_or_coordinated'),
             'local_demonstration_rows':len(local_demos),
             'ship_state_demonstration_rows':len(ship_state_demos),
             'initial_berthing_success':first_success_time is not None,
             'first_success_time_s':first_success_time,
             'first_success_ship_state':first_success_state,
             'first_success_hull_gap_m':env.first_success_hull_gap_m,
             'post_success_requested_s':float(post_success_seconds),
             'stationkeeping_complete':bool(stationkeeping_complete),
             'stationkeeping_passed':bool(stationkeeping_passed),
             'post_success_observed_s':float(audit.post_success_elapsed_s),
             'post_success_in_limits_s':float(audit.post_success_in_limits_s),
             'post_success_violations':int(audit.post_success_violations),
             'post_success_samples':int(audit.post_success_samples),
             'post_success_fraction_in_limits':(float(audit.post_success_in_limits_s/audit.post_success_elapsed_s)
                 if audit.post_success_elapsed_s>0 else None),
             'post_success_max_position_error_m':(float(audit.post_success_max_position_error_m)
                 if audit.post_success_samples else None),
             'post_success_max_heading_error_deg':(float(audit.post_success_max_heading_error_deg)
                 if audit.post_success_samples else None),
             'post_success_max_speed_mps':(float(audit.post_success_max_speed_mps)
                 if audit.post_success_samples else None),
             'post_success_max_yaw_rate_deg_s':(float(audit.post_success_max_yaw_rate_deg_s)
                 if audit.post_success_samples else None),
             'post_success_min_hull_gap_m':(float(audit.post_success_min_hull_gap_m)
                 if audit.post_success_samples else None),
             'actual_start_ship_xypsi':(rows[0]['x_m'],rows[0]['y_m'],rows[0]['psi_deg']),
             'evaluation_position_jitter_m':float(eval_position_jitter_m),
             'evaluation_heading_jitter_deg':float(eval_heading_jitter_deg),
             'guidance_mode':guidance_mode,
             'start_x_m':float(start_x),'start_y_m':float(start_y),
             'path_length_m':path_length,'direct_distance_m':direct_distance,
             'path_length_ratio':(path_length/direct_distance if direct_distance>1e-6 else None),
             'max_along_target_overshoot_m':max_target_overshoot,
             'coordination_communication_bytes':int(joint.communication_bytes_total) if joint else 0,
             'push_heading_guard_enabled':bool(push_heading_guard and controller_method!='contact_local'),
             'local_heading_helm_enabled':controller_method=='contact_local',
             'push_heading_guard_interventions':[g.n_interventions for g in env.heading_guards],
             'max_push_heading_error_deg':([float(max(abs(r[f'push{i+1}_heading_error_deg']) for r in rows)) for i in range(2)]
                 if controller_method=='contact_local' else [g.max_abs_error_deg for g in env.heading_guards]),
             'mean_admm_primal_residual':(float(np.mean([d['primal_residual'] for d in decisions if d.get('agent')==1]))
                     if controller_method=='admm' and decisions else None),
             'mean_admm_dual_residual':(float(np.mean([d['dual_residual'] for d in decisions if d.get('agent')==1]))
                     if controller_method=='admm' and decisions else None),
             'status':reason,'success':bool(first_success_time is not None),
             'unsafe':bool(info.get('unsafe',False)),'reason':info.get('reason',''),
             'stop_cause':stop_cause,'emergency_time_guard_hit':stop_cause=='MAX_SIM_TIME_GUARD',
             'emergency_max_sim_seconds':float(max_seconds),
             'ended_by_success_or_safety':reason in ('SUCCESS','SUCCESS_HOLD','STATION_KEEPING_FAILED','UNSAFE'),
             'seconds':float(env.time),'cpu_elapsed_seconds':round(elapsed,2),
             'position_error_m':float(info.get('position_error_m',np.nan)),
             'heading_error_deg':float(info.get('heading_error_deg',np.nan)),
             'hull_gap_m':float(info.get('hull_gap_m',np.nan)),
             'total_reward':float(ret),'nm_pc_solvers':(4 if controller_method!='central' else 1),
             'controllers_per_tug':(1 if controller_method in ('decentralized','ship_state_local','contact_local') else 0),
             'wind_mps':float(wind_mps),'wind_to_deg':float(wind_to_deg),
             'current_mps':float(current_mps),'current_to_deg':float(current_to_deg),
             'gust_mps':float(gust_mps),'current_osc_mps':float(current_osc_mps),
             'disturbance_mode':('local_point_residual' if controller_method=='contact_local' else disturbance_mode),
             'position_tolerance_m':float(position_tolerance_m),
             'min_logged_hull_gap_m':float(min_gap),
             'max_logged_yaw_rate_deg_s':float(max_yaw_rate),
             'max_logged_close_speed_mps':float(peak_approach_speed),
             'max_substep_close_speed_mps':float(max_substep_close_speed),
             'max_interaction_force_N':float(max_interaction_force_N),
             'integrated_abs_thrust_Ns':float(integrated_abs_thrust),
             'mean_total_abs_thrust_N':float(integrated_abs_thrust/max(float(env.time),1.)),
             'integrated_abs_thrust_change_N':float(integrated_thrust_change),
             'avg_optimizer_ms':float(np.mean([d['optimizer_elapsed_ms'] for d in decisions])),
             'mean_decision_wall_ms':float(np.mean([
                 sum(d['optimizer_elapsed_ms'] for d in decisions[j:j+4])
                 for j in range(0,len(decisions),4)])),
             'solver_success_rate':float(np.mean([d['success_solver'] for d in decisions])),
             'terminal_counterforce_active_decisions':sum(float(d.get('terminal_counterforce_gate',0.))>0.05 for d in decisions),
             'max_terminal_counterforce_gate':max((float(d.get('terminal_counterforce_gate',0.)) for d in decisions),default=0.),
             'terminal_counterforce_weight':float(local_cfg.terminal_counterforce_weight),
             'model':('Contact-point-only reduced-order local NMPC' if controller_method=='contact_local' else
              'Local nonlinear reduced-order NMPC (shared ship state, private own tug)' if controller_method in ('decentralized','ship_state_local') else
              'Joint nonlinear reduced-order ship MPC / same full physical validation plant'),
             'communication_between_tugs':controller_method=='admm',
             'ship_state_shared_to_all_independent_agents':controller_method in ('decentralized','ship_state_local'),
             'peer_state_access':controller_method in ('central','admm'),
             'centralized_joint_optimizer':controller_method=='central',
             'centralized_force_allocator':False}
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    settings={'controller_version':CONTROLLER_VERSION,'physics_mode':physics_mode,'controller_method':controller_method,
              'guidance_mode':guidance_mode,
              'start_x':float(start_x),'start_y':float(start_y),
              'push_heading_guard':bool(push_heading_guard and controller_method!='contact_local'),
              'admm_rounds':admm_rounds,'admm_rho':admm_rho,
              'admm_inner_iterations':admm_inner_iterations,
              'anti_stall_gain':anti_stall_gain,'anti_stall_window_s':anti_stall_window_s,
              'psi0':psi0,'max_seconds':max_seconds,'save_every_seconds':save_every_seconds,
              'termination_mode':'success_or_unsafe_with_optional_emergency_guard',
              'dt':dt,'decision_seconds':decision_seconds,
              'target_x':target_x,'target_y':target_y,'target_psi_deg':target_psi_deg,
              'horizon_s':horizon,'max_iterations':maxiter,'seed':seed,
              'wind_mps':wind_mps,'wind_to_deg':wind_to_deg,
              'current_mps':current_mps,'current_to_deg':current_to_deg,
              'gust_mps':gust_mps,'gust_period_s':gust_period_s,
              'current_osc_mps':current_osc_mps,
              'wind_sensor_std_mps':wind_sensor_std_mps,
              'current_sensor_std_mps':current_sensor_std_mps,
              'disturbance_mode':('local_point_residual' if controller_method=='contact_local' else disturbance_mode),
              'position_tolerance_m':position_tolerance_m,
              'precision':precision,
              'post_success_seconds':post_success_seconds,
              'eval_position_jitter_m':eval_position_jitter_m,
              'eval_heading_jitter_deg':eval_heading_jitter_deg,
              'max_fraction_step':max_fraction_step,
              'max_offset_step_deg':max_offset_step_deg,
              'slew_weight':slew_weight,'export_local_demos':bool(export_local_demos or controller_method=='contact_local'),
              'export_ship_state_demos':bool(export_ship_state_demos or controller_method=='ship_state_local'),
              'ship_state_vector_columns':list(SHIP_STATE_VECTOR_COLUMNS) if controller_method=='ship_state_local' or export_ship_state_demos else []}
    (output/'configuration.json').write_text(json.dumps(settings,indent=2),encoding='utf8')
    if render:
        from plot_marl import main as plot_main
        import sys
        old=sys.argv
        sys.argv=['plot_marl.py','--input',str(output/'trajectory.csv'),'--out',str(output),
                  '--target-x',str(target_x),'--target-y',str(target_y),
                  '--target-psi-deg',str(target_psi_deg)]
        try:plot_main()
        finally:sys.argv=old
        from nmpc_report import report as create_paper_report
        create_paper_report(output)
    if verbose:print(json.dumps(metrics,indent=2,ensure_ascii=False),flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',action='version',version=CONTROLLER_VERSION+' (four independent ship-state local NMPC controllers)')
    p.add_argument('--out',default='results/nmpc_h20')
    p.add_argument('--psi0',type=float,default=20.)
    p.add_argument('--duration',type=float,default=DEFAULT_EMERGENCY_SIM_SECONDS,
                   help='Emergency simulation-time ceiling, not a target termination time. 0 disables the ceiling; stop at SUCCESS/UNSAFE or Ctrl+C.')
    p.add_argument('--save-every-seconds',type=float,default=300.,
                   help='Persist CSV/progress.json every N simulated seconds (0 disables periodic saving).')
    p.add_argument('--post-success-seconds',type=float,default=0.,
                   help='Continue the same controller/plant after first safe berthing for N seconds; default 0 retains original termination.')
    p.add_argument('--eval-position-jitter-m',type=float,default=0.,
                   help='Seeded uniform +/- position perturbation on each x/y coordinate, for robustness evaluation.')
    p.add_argument('--eval-heading-jitter-deg',type=float,default=0.,
                   help='Seeded uniform +/- initial heading perturbation, for robustness evaluation.')
    p.add_argument('--decision-seconds',type=float,default=10.)
    p.add_argument('--dt',type=float,default=.2)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--horizon',type=float,default=90.)
    p.add_argument('--maxiter',type=int,default=10)
    p.add_argument('--target-x',type=float,default=50.)
    p.add_argument('--target-y',type=float,default=-100.)
    p.add_argument('--target-psi-deg',type=float,default=0.)
    p.add_argument('--start-x',type=float,default=0.)
    p.add_argument('--start-y',type=float,default=0.)
    p.add_argument('--physics-mode',choices=['energy_consistent','legacy'],default='energy_consistent',
                   help='Corrected energy-consistent Coriolis equations by default; legacy reproduces v4.6 cross terms.')
    p.add_argument('--guidance-mode',choices=['mission','legacy'],default='mission',
                   help='mission: reference tracking with along-berth braking/corridor; legacy: exact previous local objective.')
    p.add_argument('--wind-mps',type=float,default=0.)
    p.add_argument('--wind-to-deg',type=float,default=0.,help='Wind TOWARD direction (not meteorological FROM).')
    p.add_argument('--current-mps',type=float,default=0.)
    p.add_argument('--current-to-deg',type=float,default=0.,help='Water current TOWARD direction.')
    p.add_argument('--gust-mps',type=float,default=0.)
    p.add_argument('--gust-period-s',type=float,default=360.)
    p.add_argument('--current-osc-mps',type=float,default=0.)
    p.add_argument('--disturbance-mode',choices=['none','residual','measured'],default='measured')
    p.add_argument('--environmental-reserve',choices=['auto','on','off'],default='auto',
                   help='Raise software push-command cap under adverse measured loading without increasing physical actuator capacity. Auto=on for ship_state_local only.')
    p.add_argument('--wind-sensor-std-mps',type=float,default=0.)
    p.add_argument('--current-sensor-std-mps',type=float,default=0.)
    p.add_argument('--position-tolerance-m',type=float,default=2.5)
    p.add_argument('--precision',dest='precision',action='store_true',default=True,
                   help='Use berth-scale terminal position objective (new default).')
    p.add_argument('--no-precision',dest='precision',action='store_false',
                   help='Ablation: reproduce old near-zero terminal-position penalty.')
    p.add_argument('--max-fraction-step',type=float,default=.38)
    p.add_argument('--max-offset-step-deg',type=float,default=24.)
    p.add_argument('--slew-weight',type=float,default=.10)
    p.add_argument('--anti-stall-gain',type=float,default=1.3,
                   help='Increase position/motion objectives when long-term progress stalls; 0 disables.')
    p.add_argument('--anti-stall-window-s',type=float,default=200.)
    p.add_argument('--disable-push-heading-guard',action='store_true',
                   help='Ablation only: disable the local 0.2 s push heading guard.')
    p.add_argument('--controller-method',choices=['decentralized','ship_state_local','central','admm','contact_local'],
                   default='ship_state_local',help='ship_state_local: independent NMPC with shared ship 3-DOF state and private own tug; decentralized: legacy-equivalent name; contact_local: point only.')
    p.add_argument('--admm-rounds',type=int,default=3)
    p.add_argument('--admm-rho',type=float,default=.5)
    p.add_argument('--admm-inner-iterations',type=int,default=4)
    p.add_argument('--export-local-demos',action='store_true',
                   help='Write point-only observations and actions as IL data (also works with privileged teacher).')
    p.add_argument('--export-ship-state-demos',action='store_true',
                   help='Log 21-feature ship-state-only observation and per-tug expert action (no peer features).')
    p.add_argument('--no-plots',action='store_true')
    a=p.parse_args()
    if any(x<0 for x in (a.wind_mps,a.current_mps,a.gust_mps,a.current_osc_mps,
                         a.wind_sensor_std_mps,a.current_sensor_std_mps)):
        p.error('Environmental magnitudes and sensor std must be nonnegative')
    if a.gust_period_s<=0 or a.position_tolerance_m<=0 or a.anti_stall_window_s<=0:
        p.error('Period/tolerance/window must be positive')
    if a.anti_stall_gain<0:p.error('Anti-stall gain must be nonnegative')
    if a.duration<0 or a.save_every_seconds<0:p.error('duration/save interval must be nonnegative')
    print(f'[NMPC] VERSION={CONTROLLER_VERSION} | SCRIPT={Path(__file__).resolve()}',flush=True)
    print(f'[NMPC] CONFIG: method={a.controller_method}, guidance={a.guidance_mode}, precision={a.precision}, psi0={a.psi0}, wind={a.wind_mps}, current={a.current_mps}, current_to={a.current_to_deg}, seed={a.seed}',flush=True)
    rollout(out=a.out,psi0=a.psi0,max_seconds=a.duration,decision_seconds=a.decision_seconds,
            dt=a.dt,seed=a.seed,render=not a.no_plots,horizon=a.horizon,maxiter=a.maxiter,
            target_x=a.target_x,target_y=a.target_y,target_psi_deg=a.target_psi_deg,
            start_x=a.start_x,start_y=a.start_y,guidance_mode=a.guidance_mode,
            wind_mps=a.wind_mps,wind_to_deg=a.wind_to_deg,
            current_mps=a.current_mps,current_to_deg=a.current_to_deg,
            gust_mps=a.gust_mps,gust_period_s=a.gust_period_s,
            current_osc_mps=a.current_osc_mps,
            disturbance_mode=a.disturbance_mode,environmental_reserve=a.environmental_reserve,
            wind_sensor_std_mps=a.wind_sensor_std_mps,
            current_sensor_std_mps=a.current_sensor_std_mps,
            position_tolerance_m=a.position_tolerance_m,
            precision=a.precision,
            max_fraction_step=a.max_fraction_step,
            max_offset_step_deg=a.max_offset_step_deg,
            slew_weight=a.slew_weight,
            anti_stall_gain=a.anti_stall_gain,
            anti_stall_window_s=a.anti_stall_window_s,
            save_every_seconds=a.save_every_seconds,
            controller_method=a.controller_method,push_heading_guard=not a.disable_push_heading_guard,admm_rounds=a.admm_rounds,
            admm_rho=a.admm_rho,admm_inner_iterations=a.admm_inner_iterations,
            post_success_seconds=a.post_success_seconds,
            eval_position_jitter_m=a.eval_position_jitter_m,
            eval_heading_jitter_deg=a.eval_heading_jitter_deg,physics_mode=a.physics_mode,
            export_local_demos=a.export_local_demos,
            export_ship_state_demos=a.export_ship_state_demos)


if __name__=='__main__':main()
