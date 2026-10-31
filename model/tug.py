import numpy as np


class Tug:
    def __init__(self):
        # Water / air properties
        self.rho = 1025.0
        self.rho_a = 1.26

        # Tug parameters
        self.Lppi = 22.943
        self.di = 3.084
        self.xGi = 0.0
        self.xTi = -self.Lppi / 2.0
        self.api = self.Lppi / 2.0

        # Towing geometry defaults. main.py may overwrite/add these too.
        self.ati = self.Lppi / 2.0
        self.lci = 40.0

        # Approximate above-water geometry for wind loads.
        # Replace these with measured tug dimensions if available.
        self.Bi = 0.37 * self.Lppi          # approx. harbor tug beam
        self.freeboard_i = max(0.75 * self.di, 2.5)
        self.A_front_wind = self.Bi * self.freeboard_i
        self.A_side_wind = self.Lppi * self.freeboard_i

        # Wind coefficients for a compact bluff tug superstructure/hull.
        # They are intentionally moderate and can be calibrated later.
        self.CX_wind = 0.80
        self.CY_wind = 1.10
        self.CN_wind = 0.08

        # Tug mass, added mass, moment of inertia and added moment inertia
        self.mi = 358.4e3
        self.mxi = 0.05 * self.mi
        self.myi = 0.50 * self.mi
        self.IzGi = self.mi * (0.25 * self.Lppi) ** 2
        self.Jzi = 0.10 * self.IzGi

        # Hydrodynamics coeffs. Original dimensional style kept.
        self.CD0 = 1.10
        # surge
        self.X0 = -60e3
        self.Xvr = 0.0
        # sway
        self.Yv = -180e3
        self.Yr = 0.0
        self.CrY = 0.20
        # yaw
        self.Nv = 0.0
        self.Nr = -80e6
        self.CrN = 0.20

        # propeller thrust deduction
        self.ti = 0.1

    @staticmethod
    def wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _earth_velocity_to_body(self, V, psi_env, psi_body):
        """
        Convert an earth-fixed velocity vector into the tug body frame.

        Convention:
            psi_env is the earth-fixed direction TOWARD which the wind/current
            velocity vector moves. If the value is a meteorological/oceanographic
            'coming from' direction, add 180 deg before passing it here.
        """
        beta = self.wrap_to_pi(psi_env - psi_body)
        return V * np.cos(beta), V * np.sin(beta), beta

    def _crossflow_integrals(self, v_rel_mid, ri, nseg=41):
        xs = np.linspace(-self.Lppi / 2.0, self.Lppi / 2.0, nseg)
        dx = xs[1] - xs[0]
        termY = 0.0
        termN = 0.0
        for x in xs:
            vy = v_rel_mid + self.CrY * ri * x
            vn = v_rel_mid + self.CrN * ri * x
            termY += abs(vy) * vy * dx
            termN += abs(vn) * vn * x * dx
        return termY, termN

    def _hydrodynamic_forces(self, ui, vi, ri, psi_i, Vc=0.0, psi_c=0.0):
        """
        Tug hydrodynamics using velocity relative to water.

        Current is not added as a separate force. Instead, the current velocity
        is transformed to the tug body frame and subtracted from the tug velocity.
        """
        vim = vi - ri * self.xGi
        u_c_body, v_c_body, _ = self._earth_velocity_to_body(Vc, psi_c, psi_i)

        u_rel = ui - u_c_body
        v_rel = vim - v_c_body
        U_rel = np.hypot(u_rel, v_rel)

        termY, termN = self._crossflow_integrals(v_rel, ri)

        XHi = self.X0 * u_rel * U_rel + self.Xvr * v_rel * ri
        YHi = (
            self.Yv * v_rel * abs(u_rel)
            + self.Yr * ri * u_rel
            - 0.5 * self.rho * self.di * self.CD0 * termY
        )
        NHi = (
            self.Nv * v_rel * u_rel
            + self.Nr * ri * abs(u_rel)
            - 0.5 * self.rho * self.di * self.CD0 * termN
        )

        return XHi, YHi, NHi

    def _wind_forces(self, ui, vi, ri, psi_i, Vw=0.0, psi_w=0.0, wind_factor=1.0):
        """
        Apparent-wind loads on the tug in tug-body axes.

        wind_factor is supplied by the coupled ship--tug model. It represents
        geometric shielding by the main ship:
            1.0 = fully exposed to wind
            0.0 = fully shielded
        """
        wind_factor = float(np.clip(wind_factor, 0.0, 1.0))
        if abs(Vw) < 1e-12 or wind_factor <= 1e-12:
            return 0.0, 0.0, 0.0

        vim = vi - ri * self.xGi
        u_air_b, v_air_b, _ = self._earth_velocity_to_body(Vw, psi_w, psi_i)

        # Air velocity relative to the tug.
        u_app = u_air_b - ui
        v_app = v_air_b - vim
        V_app = np.hypot(u_app, v_app)
        if V_app < 1e-12:
            return 0.0, 0.0, 0.0

        beta_app = np.arctan2(v_app, u_app)

        X_w = 0.5 * self.rho_a * self.CX_wind * self.A_front_wind * u_app * abs(u_app)
        Y_w = 0.5 * self.rho_a * self.CY_wind * self.A_side_wind * v_app * abs(v_app)
        N_w = (
            0.5
            * self.rho_a
            * self.CN_wind
            * self.A_side_wind
            * self.Lppi
            * V_app**2
            * np.sin(2.0 * beta_app)
        )

        return wind_factor * X_w, wind_factor * Y_w, wind_factor * N_w

    def push_tug_matrices(
        self,
        lpi,
        psi_pi,
        r0,
        psi0,
        ui,
        vi,
        ri,
        psi_i,
        Ti,
        delta_i,
        Vw=0.0,
        psi_w=0.0,
        Vc=0.0,
        psi_c=0.0,
        wind_factor=1.0,
    ):
        XHi, YHi, NHi = self._hydrodynamic_forces(ui, vi, ri, psi_i, Vc, psi_c)
        X_w, Y_w, N_w = self._wind_forces(ui, vi, ri, psi_i, Vw, psi_w, wind_factor)

        # Matrices
        Api = np.array(
            [
                [self.mi + self.mxi, 0, 0, 0, 0, -1, 0],
                [0, self.mi + self.myi, self.mi * self.xGi, 0, 0, 0, -1],
                [0, self.mi * self.xGi, (self.IzGi + self.Jzi + self.mi * self.xGi**2), 0, 0, 0, -self.api],
                [np.cos(psi_i), -np.sin(psi_i), 0, -1, 0, 0, 0],
                [np.sin(psi_i), np.cos(psi_i), 0, 0, -1, 0, 0],
                [0, 0, self.api * np.sin(psi_i), -1, 0, 0, 0],
                [0, 0, -self.api * np.cos(psi_i), 0, -1, 0, 0],
            ],
            dtype=float,
        )

        Ap0i = np.array(
            [
                [0, 0, 0, 0, 0, np.cos(psi_i - psi0), -np.sin(psi_i - psi0)],
                [0, 0, 0, 0, 0, np.sin(psi_i - psi0), np.cos(psi_i - psi0)],
                [0, 0, 0, 0, 0, lpi * np.sin(psi_i - psi0 - psi_pi), lpi * np.cos(psi_i - psi0 - psi_pi)],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

        Api0 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -lpi * np.sin(psi0 + psi_pi), 1, 0],
                [0, 0, lpi * np.cos(psi0 + psi_pi), 0, 1],
            ],
            dtype=float,
        )

        Cpi = np.array(
            [
                [(self.mi + self.myi) * vi * ri + self.mi * self.xGi * ri**2 + XHi + X_w + (1.0 - self.ti) * Ti * np.cos(delta_i)],
                [-(self.mi + self.mxi) * ui * ri + YHi + Y_w - Ti * np.sin(delta_i)],
                [-self.mi * self.xGi * ui * ri + NHi + N_w - self.xTi * Ti * np.sin(delta_i)],
                [ui * ri * np.sin(psi_i) + vi * ri * np.cos(psi_i)],
                [-ui * ri * np.cos(psi_i) + vi * ri * np.sin(psi_i)],
                [r0**2 * lpi * np.cos(psi0 + psi_pi) - ri**2 * self.api * np.cos(psi_i)],
                [r0**2 * lpi * np.sin(psi0 + psi_pi) - ri**2 * self.api * np.sin(psi_i)],
            ],
            dtype=float,
        )

        return Api, Ap0i, Api0, Cpi

    def tow_tug_matrices(
        self,
        lti,
        psi_ti,
        r0,
        psi0,
        ui,
        vi,
        ri,
        psi_i,
        theta_dot_i,
        theta_i,
        Ti,
        delta_i,
        Vw=0.0,
        psi_w=0.0,
        Vc=0.0,
        psi_c=0.0,
        wind_factor=1.0,
    ):
        XHi, YHi, NHi = self._hydrodynamic_forces(ui, vi, ri, psi_i, Vc, psi_c)
        X_w, Y_w, N_w = self._wind_forces(ui, vi, ri, psi_i, Vw, psi_w, wind_factor)

        Di63 = self.lci * np.sin(psi_i + theta_i) + self.ati * np.sin(psi_i)
        Di66 = self.lci * np.sin(psi_i + theta_i)
        Di73 = -self.lci * np.cos(psi_i + theta_i) - self.ati * np.cos(psi_i)
        Di76 = -self.lci * np.cos(psi_i + theta_i)

        Ati = np.array(
            [
                [self.mi + self.mxi, 0, 0, 0, 0, 0, -np.cos(theta_i)],
                [0, self.mi + self.myi, self.mi * self.xGi, 0, 0, 0, -np.sin(theta_i)],
                [0, self.mi * self.xGi, (self.IzGi + self.Jzi + self.mi * self.xGi**2), 0, 0, 0, -self.ati * np.sin(theta_i)],
                [np.cos(psi_i), -np.sin(psi_i), 0, -1, 0, 0, 0],
                [np.sin(psi_i), np.cos(psi_i), 0, 0, -1, 0, 0],
                [0, 0, Di63, -1, 0, Di66, 0],
                [0, 0, Di73, 0, -1, Di76, 0],
            ],
            dtype=float,
        )

        At0i = np.array(
            [
                [0, 0, 0, 0, 0, 0, np.cos(psi0 - theta_i - psi_i)],
                [0, 0, 0, 0, 0, 0, -np.sin(psi0 - theta_i - psi_i)],
                [0, 0, 0, 0, 0, 0, -lti * np.sin(psi0 - theta_i - psi_i + psi_ti)],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

        Ati0 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -lti * np.sin(psi0 + psi_ti), 1, 0],
                [0, 0, lti * np.cos(psi0 + psi_ti), 0, 1],
            ],
            dtype=float,
        )

        Cti = np.array(
            [
                [(self.mi + self.myi) * vi * ri + self.mi * self.xGi * ri**2 + XHi + X_w + (1.0 - self.ti) * Ti * np.cos(delta_i)],
                [-(self.mi + self.mxi) * ui * ri + YHi + Y_w - Ti * np.sin(delta_i)],
                [-self.mi * self.xGi * ui * ri + NHi + N_w - self.xTi * Ti * np.sin(delta_i)],
                [ui * ri * np.sin(psi_i) + vi * ri * np.cos(psi_i)],
                [-ui * ri * np.cos(psi_i) + vi * ri * np.sin(psi_i)],
                [
                    r0**2 * lti * np.cos(psi0 + psi_ti)
                    - ri**2 * (self.lci * np.cos(psi_i + theta_i) + self.ati * np.cos(psi_i))
                    - (2.0 * ri * self.lci * theta_dot_i + theta_dot_i**2 * self.lci) * np.cos(psi_i + theta_i)
                ],
                [
                    r0**2 * lti * np.sin(psi0 + psi_ti)
                    - ri**2 * (self.lci * np.sin(psi_i + theta_i) + self.ati * np.sin(psi_i))
                    - (2.0 * ri * self.lci * theta_dot_i + theta_dot_i**2 * self.lci) * np.sin(psi_i + theta_i)
                ],
            ],
            dtype=float,
        )

        return Ati, At0i, Ati0, Cti
