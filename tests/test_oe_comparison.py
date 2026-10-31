"""Publication comparison regression: mathematical architecture and identical plant."""
from pathlib import Path
import csv,json,tempfile
import numpy as np
import pytest
from marl_game.env import BerthingGameEnv,EnvConfig
from decentralized_nmpc.controller import NMPCConfig
from decentralized_nmpc.coordination_baselines import JointNMPC,CoordinationSettings
from run_nmpc import rollout
from paper_oe_benchmark import report


def _env():
    e=BerthingGameEnv(EnvConfig(train=False,initial_psi_deg=20.,max_seconds=100.,
        eval_wind_mps=6.,eval_wind_direction_deg=-90.,eval_current_mps=.2,
        eval_current_direction_deg=-90.))
    e.reset()
    return e


@pytest.mark.parametrize('method',['central','admm'])
def test_joint_optimizers_respect_same_force_angle_bounds(method):
    env=_env()
    obj=JointNMPC(env,NMPCConfig(max_iterations=2),
         CoordinationSettings(method=method,admm_rounds=2,admm_inner_iterations=2))
    actions,reports=obj.act(env)
    assert actions.shape==(4,2)
    assert np.all(np.isfinite(actions))
    assert np.all(np.abs(actions)<=1.+1e-7)
    fractions=(actions[:,0]+1)/2
    assert np.all(fractions[:2]<=.52+1e-7)
    assert np.all(fractions[2:]<=.65+1e-7)
    assert len(reports)==4
    assert obj.settings.method==method
    assert (obj.communication_bytes_total>0)==(method=='admm')


def test_same_physics_safety_monitor_used_across_methods():
    with tempfile.TemporaryDirectory() as td:
        configs=[]
        for method in ('decentralized','central','admm'):
            folder=Path(td)/method
            m=rollout(out=str(folder),controller_method=method,psi0=20.,
                      max_seconds=10.,maxiter=2,render=False,verbose=False,
                      wind_mps=6.,wind_to_deg=-90.,current_mps=.2,current_to_deg=-90.,
                      admm_rounds=2,admm_inner_iterations=2)
            assert m['status']=='TIMEOUT' and not m['success']
            assert m['emergency_time_guard_hit']
            cfg=json.loads((folder/'configuration.json').read_text())
            data=list(csv.DictReader((folder/'trajectory.csv').open()))
            configs.append((cfg,data))
            assert len(data)==2
            assert data[0]['wind_mps']=='6.0'
        for key in ('psi0','target_x','target_y','target_psi_deg','dt','decision_seconds',
                    'position_tolerance_m','wind_mps','current_mps'):
            assert len(set(c[key] for c,_ in configs))==1
        for _,data in configs:
            assert float(data[0]['x_m'])==0.
            assert float(data[0]['y_m'])==0.


def test_paper_table_keeps_timeout_as_failure():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        out=root/'case_calm_h20_s123_decentralized';out.mkdir()
        m=dict(status='TIMEOUT',success=False,unsafe=False,seconds=10.,
            position_error_m=110.,heading_error_deg=19.,min_logged_hull_gap_m=65.,
            max_substep_close_speed_mps=0.,max_interaction_force_N=0.,
            integrated_abs_thrust_Ns=1.,integrated_abs_thrust_change_N=1.,
            avg_optimizer_ms=1.,cpu_elapsed_seconds=.4)
        c=dict(benchmark_case='calm',psi0=20.,seed=123,controller_method='decentralized')
        (out/'metrics.json').write_text(json.dumps(m))
        (out/'configuration.json').write_text(json.dumps(c))
        (out/'trajectory.csv').write_text('time_s,x_m,y_m,position_error_m,north_speed_mps\n0,0,0,112,0\n10,1,0,110,0\n')
        report(root)
        agg=list(csv.DictReader((root/'Table_2_Method_Aggregate.csv').open()))
        assert agg[0]['successful']=='0'
        assert agg[0]['mean_time_to_success_s']==''
        assert (root/'Figure_2_Tracking.pdf').exists()


def test_no_silent_joint_sensor_noise_advantage():
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError,match='sensor noise'):
            rollout(out=td,controller_method='central',max_seconds=10.,
                    wind_sensor_std_mps=.2,render=False,verbose=False)
