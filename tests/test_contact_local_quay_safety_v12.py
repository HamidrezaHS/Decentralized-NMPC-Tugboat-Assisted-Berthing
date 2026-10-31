"""Local-only quay risk heuristic checks (not a safety proof)."""
from dataclasses import replace
import numpy as np
from contact_local.controller import ContactNMPC
from contact_local.sensors import contact_observation
from contact_local.actuation import ContactOnlyHelm
from marl_game.env import BerthingGameEnv, EnvConfig


def setup():
    env=BerthingGameEnv(EnvConfig(train=False, contact_only=True,
                                  initial_psi_deg=40.))
    ctr=ContactNMPC(0)
    ctr.configure_berth(env.plant.attach_local[0],env.quay_y,
                         env.plant._ship_points)
    return env,ctr,contact_observation(env,0)


def test_risk_proxy_uses_only_own_contact_and_static_geometry():
    env,c,o=setup()
    base=c._local_quay_risk(o)
    env.ship[2]+=0.75  # no new point observation is generated
    env.tugs[1:]+=10.  # other tugs must not matter
    assert base==c._local_quay_risk(o)
    closer=replace(o,point_xy=o.point_xy+[0,-5])
    farther=replace(o,point_xy=o.point_xy+[0,5])
    assert c._local_quay_risk(closer)[0] < base[0] < c._local_quay_risk(farther)[0]


def test_local_heading_yaw_command_is_bounded_and_not_fed_ship_state():
    env,c,o=setup()
    helm=ContactOnlyHelm(0,env.plant.max_force)
    cmd=np.asarray(helm.command(o,np.array([0.6,0.1])))
    assert np.isfinite(cmd).all()
    assert abs(cmd[2]) <= env.plant.c.push_twin_azimuth_max_yaw_moment_Nm
    assert 0 <= cmd[0] <= env.plant.max_force
    shifted=replace(o,own=o.own+np.array([0,0,.15,0,0,0]))
    corrected=np.asarray(helm.command(shifted,np.array([0.6,0.1])))
    assert abs(corrected[2])<=5e6


def test_contact_decisions_report_own_risk_only():
    env,c,o=setup()
    a,d=c.act(o)
    assert a.shape==(2,)
    assert np.isfinite(d['local_quay_proxy_gap_m'])
    assert d['quay_risk_model']=='static_hull_and_own_tug_heading_heuristic'
    assert 'ship_heading' not in d


def test_episodic_helm_calibration_is_reset():
    env=BerthingGameEnv(EnvConfig(train=False,contact_only=True,initial_psi_deg=20.))
    obs=contact_observation(env,0)
    helm=env.local_helms[0]
    helm.command(obs,np.array([0.,0.]))
    assert helm._initial_outboard is not None
    env.reset()
    assert helm._initial_outboard is None
    assert helm._initial_error is None
