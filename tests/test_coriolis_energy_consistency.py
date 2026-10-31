import numpy as np

from model.ship import Ship


def test_corrected_coriolis_terms_do_no_net_work():
    ship = Ship()
    assert ship.coriolis_mode == "energy_consistent"
    rng = np.random.default_rng(20261022)
    mass = np.array([
        [ship.m0 + ship.mx0, 0.0, 0.0],
        [0.0, ship.m0 + ship.my0, ship.m0 * ship.xG0],
        [0.0, ship.m0 * ship.xG0, ship.IzG0 + ship.Jz0 + ship.m0 * ship.xG0**2],
    ])
    for _ in range(100):
        nu = rng.normal(size=3)
        q = ship.coriolis_terms(*nu)
        assert abs(float(nu @ q)) <= 1e-10 * max(1.0, np.linalg.norm(nu) * np.linalg.norm(q))
        # The inertia matrix is symmetric positive definite for sampled states.
        assert np.linalg.eigvalsh(mass).min() > 0


def test_legacy_mode_retains_original_v46_expression():
    ship = Ship()
    ship.coriolis_mode = "legacy"
    u, v, r = 0.17, -0.08, 0.012
    expected = np.array([
        (ship.m0 + ship.my0) * u * r + ship.m0 * ship.xG0 * r**2,
        -(ship.m0 + ship.mx0) * v * r,
        -ship.m0 * ship.xG0 * u * r,
    ])
    np.testing.assert_array_equal(ship.coriolis_terms(u, v, r), expected)
