"""Prevent regression to the identically-zero tangential-offset bug."""
import numpy as np
from dataclasses import replace
from contact_local import contact_observation, ContactOnlyHelm
from marl_game.env import BerthingGameEnv, EnvConfig


def test_outboard_tangent_is_independent_of_current_rope_vector():
    env = BerthingGameEnv(EnvConfig(train=False, contact_only=True, initial_psi_deg=20.))
    observation = contact_observation(env, 2)
    helm = ContactOnlyHelm(2, env.plant.max_force, env.plant.c.bumper_m, env.plant.c.rope_length)
    # Calibrate from the initial local observation only.
    helm.command(observation, [0., 0.])
    n = helm._initial_outboard
    tangent = np.array([-n[1], n[0]])
    shifted = replace(observation, tip_xy=observation.tip_xy+12.*tangent)
    assert abs(np.dot(shifted.relative_tip_xy, tangent))>11.9
    assert np.isfinite(helm.command(shifted, [0., 0.])).all()


def test_all_four_helms_keep_only_own_initial_calibration():
    env = BerthingGameEnv(EnvConfig(train=False, contact_only=True, initial_psi_deg=20.))
    for i in range(4):
        obs = contact_observation(env,i)
        helm=ContactOnlyHelm(i, env.plant.max_force, env.plant.c.bumper_m, env.plant.c.rope_length)
        helm.command(obs, [0., 0.])
        assert np.linalg.norm(helm._initial_outboard)-1.<1e-10
        assert helm._initial_error>0
