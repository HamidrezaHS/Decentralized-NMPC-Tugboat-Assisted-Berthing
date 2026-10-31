"""Contact-only execution isolation and deterministic regression tests."""
import csv
from dataclasses import fields
from pathlib import Path
import numpy as np
import pytest
from marl_game.env import BerthingGameEnv,EnvConfig
from contact_local import ContactNMPC,ContactObservation,contact_observation,contact_goal,ContactOnlyHelm
from physical_v4.plant import rot
from run_nmpc import rollout


def environment():
    return BerthingGameEnv(EnvConfig(train=False,contact_only=True,max_seconds=60.,
                                     initial_psi_deg=20.))


def test_exact_local_schema_and_19_policy_values():
    env=environment()
    assert env.observe().shape==(4,19)
    assert {f.name for f in fields(ContactObservation)}=={
        'point_xy','point_velocity_xy','own','tip_xy','tip_velocity_xy',
        'own_thrust_N','own_azimuth_rad','goal_point_xy','time_s','role'}
    assert not hasattr(contact_observation(env,0),'ship')
    for i in range(4):
        obs=contact_observation(env,i)
        expect=env.goal[:2]+rot(env.goal[2])@env.plant.attach_local[i]
        np.testing.assert_allclose(obs.goal_point_xy,expect)
        np.testing.assert_allclose(obs.point_xy,env.plant.attachment(env.ship)[i])
        assert np.isfinite(obs.policy_vector()).all()
        assert not obs.point_xy.flags.writeable


def test_policy_invariance_to_peer_state_and_actuators():
    env=environment()
    initial=contact_observation(env,0)
    ctrl=ContactNMPC(0)
    a1,_=ctrl.act(initial)
    env.tugs[1:,:]+=np.array([200,100,1,1,1,0.1])
    env.act.thrust[1:]=env.plant.max_force
    env.act.azimuth[1:]=.3
    other=contact_observation(env,0)
    np.testing.assert_allclose(initial.policy_vector(),other.policy_vector())
    ctrl2=ContactNMPC(0)
    a2,_=ctrl2.act(other)
    np.testing.assert_allclose(a1,a2,rtol=0,atol=1e-6)
    helm1=ContactOnlyHelm(0,env.plant.max_force)
    helm2=ContactOnlyHelm(0,env.plant.max_force)
    np.testing.assert_allclose(helm1.command(initial,a1),helm2.command(other,a2))


def test_fixed_mission_point_goals_define_vessel_pose():
    env=environment()
    g=env.goal
    stations=env.plant.attach_local
    targets=np.array([contact_goal(g,s) for s in stations])
    # Two distinct, correctly labeled, target positions determine unique
    # orientation (mod 2*pi), without runtime ship measurements.
    separation=stations[1]-stations[0]
    achieved=targets[1]-targets[0]
    psi=np.arctan2(achieved[1],achieved[0])-np.arctan2(separation[1],separation[0])
    center=targets[0]-rot(psi)@stations[0]
    np.testing.assert_allclose(center,g[:2],atol=1e-8)
    np.testing.assert_allclose(rot(psi),rot(g[2]),atol=1e-8)


def test_local_policy_is_bounded_and_fails_closed_on_invalid_inputs():
    env=environment()
    for i in range(4):
        ctrl=ContactNMPC(i)
        o=contact_observation(env,i)
        a,diag=ctrl.act(o)
        assert a.shape==(2,) and np.isfinite(a).all()
        assert np.abs(a).max()<=1.000001
        assert diag['observation_model']=='contact_point_only'
        assert np.isfinite(ContactOnlyHelm(i,env.plant.max_force).command(o,a)).all()
        with pytest.raises(TypeError):ctrl.act(env)
    with pytest.raises(ValueError):
        ContactObservation([np.nan,0],[0,0],np.zeros(6),[0,0],[0,0],0,0,[0,0],0,'push')


def test_local_datasets_do_not_contain_hidden_ship_variables(tmp_path):
    metrics=rollout(out=str(tmp_path),controller_method='contact_local',
                    psi0=20.,max_seconds=20.,decision_seconds=10.,
                    maxiter=5,render=False,verbose=False,save_every_seconds=0.)
    assert metrics['runtime_information_boundary']=='own_contact_point_and_own_tug'
    assert metrics['local_demonstration_rows'] in (4,8)
    with (tmp_path/'local_point_demonstrations.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==metrics['local_demonstration_rows']
    assert set(rows[0])=={'t_s','agent','controller_method','action_fraction','action_offset_deg',
                          *[f'obs{i:02d}' for i in range(19)]}
    assert not any(key in rows[0] for key in ('ship','ship_heading','ship_psi','ship_u','ship_v','peer_action'))


def test_privileged_teacher_can_export_only_local_observations(tmp_path):
    metrics=rollout(out=str(tmp_path),controller_method='decentralized',
                    export_local_demos=True,psi0=20.,max_seconds=10.,
                    maxiter=4,render=False,verbose=False,save_every_seconds=0.)
    assert metrics['local_demonstration_rows']==4
    assert (tmp_path/'local_point_demonstrations.csv').is_file()
