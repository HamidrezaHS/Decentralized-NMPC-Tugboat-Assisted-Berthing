"""Ship-state NMPC information contract and local imitation demonstrations."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from marl_game.env import EnvConfig, BerthingGameEnv
from decentralized_nmpc import LocalNMPC, NMPCConfig, common_ship_observation
from ship_state_local import ShipStateLocalSensor, ship_state_policy_vector, SHIP_STATE_VECTOR_COLUMNS
from run_nmpc import rollout


def _env():
    return BerthingGameEnv(EnvConfig(train=False,max_seconds=60.,initial_psi_deg=20.))


def _controller(env,i):
    return LocalNMPC(i,env.goal,env.hull.points,env.plant.attach_local[i],
                     env.side,env.plant.max_force,NMPCConfig(max_iterations=3))


def test_sensor_only_ship_and_private_own_tug_are_read():
    env=_env();sensor=ShipStateLocalSensor(1)
    before=sensor.read(env)
    env.tugs[[0,2,3],:]+=8000.
    env.act.thrust[[0,2,3]]+=400_000
    env.act.azimuth[[0,2,3]]+=1.
    after=sensor.read(env)
    assert np.array_equal(before.ship,after.ship)
    assert np.array_equal(before.own,after.own)
    assert before.own_thrust==after.own_thrust
    assert before.own_azimuth==after.own_azimuth
    assert np.array_equal(ship_state_policy_vector(before,env.goal),
                          ship_state_policy_vector(after,env.goal))
    assert not before.ship.flags.writeable
    assert not before.own.flags.writeable
    with pytest.raises(ValueError):before.ship[0]=123


def test_sharing_ship_state_is_not_sharing_peer_decisions():
    env=_env()
    obs=ShipStateLocalSensor(0).read(env)
    original=common_ship_observation(env,0)
    a,info=_controller(env,0).act(obs)
    b,_=_controller(env,0).act(original)
    assert np.array_equal(a,b)
    assert np.all(np.isfinite(a))
    assert info['agent']==1


def test_vector_features_exact_names_and_units():
    env=_env();o=ShipStateLocalSensor(3).read(env)
    x=ship_state_policy_vector(o,env.goal)
    assert x.shape==(21,)
    assert len(SHIP_STATE_VECTOR_COLUMNS)==21
    assert np.array_equal(x[:6],env.ship)
    assert np.array_equal(x[6:12],env.tugs[3])
    assert np.array_equal(x[14:17],env.goal)
    assert not any('peer' in f for f in SHIP_STATE_VECTOR_COLUMNS)


def test_bad_input_rejected():
    with pytest.raises(ValueError):ShipStateLocalSensor(4)
    with pytest.raises(ValueError):ShipStateLocalSensor(0,wind_sensor_std_mps=-1)
    with pytest.raises(ValueError):ship_state_policy_vector(ShipStateLocalSensor(0).read(_env()),[1,2])


def test_end_to_end_teacher_dataset_matches_own_actuation(tmp_path):
    out=tmp_path/'ship_state'
    m=rollout(out=str(out),controller_method='ship_state_local',
        max_seconds=20.,decision_seconds=10.,render=False,verbose=False,
        maxiter=3,save_every_seconds=0.)
    assert m['runtime_information_boundary']=='shared_ship_state_plus_own_tug_only'
    assert m['ship_state_demonstration_rows']==8
    assert not m['peer_state_access']
    assert not m['communication_between_tugs']
    assert not m['centralized_joint_optimizer']
    with (out/'local_ship_state_demonstrations.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==8
    assert all(f'obs{i:02d}' in rows[0] for i in range(21))
    assert list(json.loads((out/'local_ship_state_feature_names.json').read_text()))==list(SHIP_STATE_VECTOR_COLUMNS)
    with (out/'local_nmpc_decisions.csv').open() as f:
        decisions=list(csv.DictReader(f))
    assert len(decisions)==8
    for k in range(8):
        assert float(rows[k]['action_fraction'])==pytest.approx(float(decisions[k]['fraction']))
        assert float(rows[k]['action_offset_deg'])==pytest.approx(float(decisions[k]['offset_deg']))
    assert not (out/'local_point_demonstrations.csv').exists()


def test_explicit_mode_matches_preexisting_decentralized_controller(tmp_path):
    common=dict(max_seconds=20.,decision_seconds=10.,render=False,verbose=False,
                maxiter=3,save_every_seconds=0.,seed=123)
    r=rollout(out=str(tmp_path/'old'),controller_method='decentralized',**common)
    s=rollout(out=str(tmp_path/'new'),controller_method='ship_state_local',**common)
    assert r['status']==s['status']
    for key in ('seconds','position_error_m','heading_error_deg'):
        assert r[key]==pytest.approx(s[key],abs=1e-7)
