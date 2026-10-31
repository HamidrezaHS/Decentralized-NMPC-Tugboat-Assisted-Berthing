"""Environmental forcing/sensor boundary and paper reporting tests."""
from pathlib import Path
import csv
import json
import tempfile
import numpy as np
import pytest
from decentralized_nmpc.controller import common_ship_observation,LocalNMPC,NMPCConfig
from marl_game.env import BerthingGameEnv,EnvConfig
from run_nmpc import rollout
from benchmark_environment import SCENARIOS
from environmental_report import generate_report


def _controller(env,i=0,mode='measured'):
    return LocalNMPC(i,env.goal,env.hull.points,env.plant.attach_local[i],
                     env.side,env.plant.max_force,
                     NMPCConfig(disturbance_mode=mode,max_iterations=3))


def test_environment_sensor_has_no_peer_data():
    env=BerthingGameEnv(EnvConfig(train=False,eval_wind_mps=8.,
         eval_wind_direction_deg=-90.,eval_current_mps=.3,
         eval_current_direction_deg=180.))
    o=common_ship_observation(env,0)
    assert o.wind_mps==8.
    assert o.current_mps==.3
    assert np.isclose(o.wind_to_rad,-np.pi/2)
    env.tugs[1:,0:2]+=800
    env.act.thrust[1:]=10000
    o2=common_ship_observation(env,0)
    assert np.array_equal(o.own,o2.own)
    assert np.array_equal(o.ship,o2.ship)
    assert o.wind_mps==o2.wind_mps
    assert np.array_equal(_controller(env).act(o)[0],_controller(env).act(o2)[0])


def test_blind_and_measured_dynamics_are_distinct():
    env=BerthingGameEnv(EnvConfig(train=False,eval_wind_mps=12.,
                        eval_wind_direction_deg=-90.,eval_current_mps=.5,
                        eval_current_direction_deg=-90.))
    ob=common_ship_observation(env,1)
    zero=common_ship_observation(env,1,measure_environment=False)
    measured=_controller(env,1,'measured')
    blind=_controller(env,1,'none')
    assert ob.current_mps>0 and zero.current_mps==0
    # Feed-forward modelling changes the predicted hydrodynamic acceleration;
    # do not require specific control command signs in a nonlinear optimizer.
    sh=ob.ship
    a,c=measured.model.ship_matrices(sh[3],sh[4],sh[5],sh[2],
          Vw=ob.wind_mps,psi_w=ob.wind_to_rad,
          Vc=ob.current_mps,psi_c=ob.current_to_rad)
    a0,c0=blind.model.ship_matrices(sh[3],sh[4],sh[5],sh[2])
    assert not np.allclose(c,c0)


def test_environmental_rollout_records_full_configuration_and_safety():
    with tempfile.TemporaryDirectory() as tmp:
        m=rollout(out=tmp,psi0=20.,max_seconds=20.,decision_seconds=10.,
             wind_mps=7.,wind_to_deg=-90.,current_mps=.2,
             current_to_deg=-90.,gust_mps=1.,gust_period_s=100.,
             maxiter=3,render=False,verbose=False)
        cfg=json.loads((Path(tmp)/'configuration.json').read_text())
        assert cfg['wind_mps']==7.
        assert cfg['current_mps']==.2
        assert m['seconds']>0
        assert m['disturbance_mode']=='measured'
        rows=list(csv.DictReader((Path(tmp)/'trajectory.csv').open()))
        assert 'wind_mps' in rows[0] and 'current_mps' in rows[0]
        assert 'north_speed_mps' in rows[0]
        assert len(rows)>=2
        assert m['min_logged_hull_gap_m']>=0 or m['unsafe']


def test_paper_report_treats_timeouts_as_failures():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        case=root/'wind6_h20_measured_seed7';case.mkdir()
        (case/'configuration.json').write_text(json.dumps({'psi0':20.,'disturbance_mode':'measured',
            'wind_mps':6.,'current_mps':0.}),encoding='utf8')
        (case/'metrics.json').write_text(json.dumps({'status':'TIMEOUT','success':False,'unsafe':False,
            'seconds':10.,'position_error_m':110.,'heading_error_deg':20.,
            'cpu_elapsed_seconds':1.,'integrated_abs_thrust_Ns':100.,'avg_optimizer_ms':1.}),encoding='utf8')
        (case/'trajectory.csv').write_text('time_s,x_m,y_m,position_error_m,heading_error_deg,hull_gap_m\n0,0,0,111,20,70\n10,0,0,110,20,70\n',encoding='utf8')
        generate_report(root)
        ag=list(csv.DictReader((root/'comparison_aggregate.csv').open()))
        assert int(ag[0]['successes'])==0
        assert int(ag[0]['timeouts'])==1
        assert ag[0]['mean_success_time_s']==''
        assert (root/'01_success_matrix.png').exists()


def test_named_scenarios_have_valid_nonnegative_magnitudes():
    for s in SCENARIOS.values():
        assert s.wind_mps>=0 and s.current_mps>=0
        assert s.gust_mps>=0 and s.current_osc_mps>=0
