import numpy as np


class Ship:
    DEFAULT_CORIOLIS_MODE = 'energy_consistent'

    def __init__(self):
        self.coriolis_mode = self.DEFAULT_CORIOLIS_MODE
        # Water / air properties
        self.rho = 1025.0
        self.rho_a = 1.26

        # Ship parameters
        self.Lpp = 320.0
        self.d = 20.8
        self.B = 58.0
        self.D = 30.0
        self.xG0 = 11.8

        # Ship mass, added mass, moment of inertia and added moment inertia
        self.m0 = 312600.0 * self.rho
        self.mx0 = 0.022 * 0.5 * self.rho * self.Lpp**2 * self.d
        self.my0 = 0.223 * 0.5 * self.rho * self.Lpp**2 * self.d
        self.IzG0 = (0.25 * self.Lpp)**2 * self.m0
        self.Jz0 = 0.011 * 0.5 * self.rho * self.Lpp**4 * self.d

        # Hydrodynamic coefficients, non-dimensional
        self.CD0 = 0.879

        # Surge
        self.X0 = -0.022
        self.Xvr = -0.002

        # Sway
        self.Yv = -0.3777
        self.Yr = 0.083
        self.CrY = 2.058

        # Yaw
        self.Nv = -0.1429
        self.Nr = -0.0562
        self.CrN = 0.543

        # External generalized force applied directly to the main ship in the
        # ship body frame: [X_ext, Y_ext, N_ext]. Used for fender/quay reaction.
        self.external_tau = np.zeros(3, dtype=float)

        # Wind coefficients.
        # These are chosen so that 0.5*C approximately reproduces the old
        # coefficients 0.3, 2.8, 0.09 when the ship is initially fixed.
        self.CX_wind = 0.60
        self.CY_wind = 5.60
        self.CN_wind = 0.18

    def coriolis_terms(self, u0, v0, r0):
        """Return body-frame rigid-body plus added-mass Coriolis loads [X,Y,N].

        The energy-consistent form obeys nu.T @ C(nu) @ nu = 0 for the
        surge/sway/yaw added-mass matrix used by this model. ``legacy`` retains
        the original v4.6 cross terms for exact historical reproduction.
        """
        if self.coriolis_mode == 'legacy':
            return np.array([
                (self.m0 + self.my0) * u0 * r0 + self.m0 * self.xG0 * r0**2,
                -(self.m0 + self.mx0) * v0 * r0,
                -self.m0 * self.xG0 * u0 * r0,
            ], dtype=float)
        if self.coriolis_mode != 'energy_consistent':
            raise ValueError("coriolis_mode must be 'energy_consistent' or 'legacy'")
        return np.array([
            (self.m0 + self.my0) * v0 * r0 + self.m0 * self.xG0 * r0**2,
            -(self.m0 + self.mx0) * u0 * r0,
            (self.mx0 - self.my0) * u0 * v0 - self.m0 * self.xG0 * u0 * r0,
        ], dtype=float)

    @staticmethod
    def wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _earth_velocity_to_body(self, V, psi_env, psi_ship):
        """
        Convert an earth-fixed environmental velocity vector into ship-body components.

        Convention:
            psi_env is the earth-fixed direction TOWARD which the current/wind velocity
            vector moves. For meteorological wind/current "coming from" directions,
            add pi before calling this function.
        """
        beta = self.wrap_to_pi(psi_env - psi_ship)
        return V * np.cos(beta), V * np.sin(beta), beta

    def _crossflow_integrals(self, v_rel_mid, r0, nseg=81):
        """
        Crossflow drag integrals using the local lateral velocity relative to water.
        v_rel_mid is the lateral relative velocity at the reference point corrected
        by xG0, i.e. v0 - r0*xG0 - v_current_body.
        """
        xs = np.linspace(-self.Lpp / 2.0, self.Lpp / 2.0, nseg)
        dx = xs[1] - xs[0]

        termY = 0.0
        termN = 0.0
        for x in xs:
            vy = v_rel_mid + self.CrY * r0 * x
            vn = v_rel_mid + self.CrN * r0 * x
            termY += abs(vy) * vy * dx
            termN += abs(vn) * vn * x * dx

        return termY, termN

    def _wind_forces(self, u0, v0m, psi0, Vw, psi_w):
        """
        Wind loads based on apparent wind velocity in the ship-body frame.

        The wind is NOT included in the water-relative velocity. It is aerodynamic
        forcing and is computed from the air velocity relative to the ship.

        Convention:
            psi_w is the direction TOWARD which the air velocity vector moves in
            earth-fixed coordinates.
        """
        if abs(Vw) < 1e-12:
            return 0.0, 0.0, 0.0

        u_air_b, v_air_b, _ = self._earth_velocity_to_body(Vw, psi_w, psi0)

        # Apparent air velocity seen by the ship in body axes.
        # Positive u_app means air moves from stern to bow in body x direction.
        u_app = u_air_b - u0
        v_app = v_air_b - v0m
        V_app = np.hypot(u_app, v_app)

        if V_app < 1e-12:
            return 0.0, 0.0, 0.0

        A_front = self.B * (self.D - self.d)
        A_side = self.Lpp * (self.D - self.d)
        beta_app = np.arctan2(v_app, u_app)

        # Component-wise quadratic drag model.
        X_w = 0.5 * self.rho_a * self.CX_wind * A_front * u_app * abs(u_app)
        Y_w = 0.5 * self.rho_a * self.CY_wind * A_side * v_app * abs(v_app)

        # Simple yaw wind moment model based on apparent wind angle.
        N_w = (
            0.5
            * self.rho_a
            * self.CN_wind
            * A_side
            * self.Lpp
            * V_app**2
            * np.sin(2.0 * beta_app)
        )

        return X_w, Y_w, N_w

    def ship_matrices(self, u0, v0, r0, psi0, Vw=0.0, psi_w=0.0, Vc=0.0, psi_c=0.0, X_fender=0.0, Y_fender=0.0, N_fender=0.0):
        """
        Build the ship block matrices.

        Important environmental convention:
            psi_c and psi_w are earth-fixed directions TOWARD which the current/wind
            velocity vector moves.

        Main improvement compared with the previous version:
            - Current is NOT added as separate X_c, Y_c, N_c forces.
            - Current enters the hydrodynamic model through water-relative velocities.
            - Wind is computed separately from apparent air velocity relative to ship.
        """

        # Velocity at the hydrodynamic reference point used by the original code.
        v0m = v0 - r0 * self.xG0

        # Current velocity components in ship-body axes.
        u_c_body, v_c_body, beta_c = self._earth_velocity_to_body(Vc, psi_c, psi0)

        # Water-relative velocity components.
        u_rel = u0 - u_c_body
        v_rel = v0m - v_c_body
        U_rel = np.hypot(u_rel, v_rel)

        # Crossflow drag terms must also use water-relative lateral velocity.
        termY, termN = self._crossflow_integrals(v_rel, r0)

        # Convert the non-dimensional coefficients with dimensionally consistent
        # MMG scaling: forces ~ (rho L d) U^2 and moments ~ (rho L^2 d) U^2.
        # The previous code multiplied each coefficient by U_rel**2 and then
        # again by velocity products, which incorrectly yielded quartic speed
        # scaling. This is a model CORRECTION; validate against Sano benchmarks.
        Kf = 0.5 * self.rho * self.Lpp * self.d
        Kh = Kf * self.Lpp
        KH_yaw = Kh * self.Lpp
        XH0 = (Kf * self.X0 * u_rel * U_rel
               + Kh * self.Xvr * v_rel * r0)
        YH0 = (Kf * self.Yv * v_rel * abs(u_rel)
               + Kh * self.Yr * r0 * u_rel
               - 0.5 * self.rho * self.d * self.CD0 * termY)
        NH0 = (Kh * self.Nv * v_rel * u_rel
               + KH_yaw * self.Nr * r0 * abs(u_rel)
               - 0.5 * self.rho * self.d * self.CD0 * termN)

        # Aerodynamic wind forces and moment from apparent wind.
        X_w, Y_w, N_w = self._wind_forces(u0, v0m, psi0, Vw, psi_w)

        # External force/moment in the ship body frame. If no external load is
        # set by the main program, this is zero and the unforced model is recovered.
        try:
            Xext, Yext, Next = np.asarray(self.external_tau, dtype=float).reshape(3)
        except Exception:
            Xext, Yext, Next = 0.0, 0.0, 0.0

        # Ship block matrices.
        A0 = np.array(
            [
                [self.m0 + self.mx0, 0.0, 0.0, 0.0, 0.0],
                [0.0, self.m0 + self.my0, self.m0 * self.xG0, 0.0, 0.0],
                [
                    0.0,
                    self.m0 * self.xG0,
                    self.IzG0 + self.Jz0 + self.m0 * self.xG0**2,
                    0.0,
                    0.0,
                ],
                [np.cos(psi0), -np.sin(psi0), 0.0, -1.0, 0.0],
                [np.sin(psi0), np.cos(psi0), 0.0, 0.0, -1.0],
            ],
            dtype=float,
        )

        # Inertial/Coriolis terms still use the ship velocity over ground.
        # Environmental effects enter only through hydrodynamic relative velocity
        # and aerodynamic apparent wind loads.
        coriolis = self.coriolis_terms(u0, v0, r0)
        C0 = np.array(
            [
                [coriolis[0] + XH0 + X_w + Xext + X_fender],
                [coriolis[1] + YH0 + Y_w + Yext + Y_fender],
                [coriolis[2] + NH0 + N_w + Next + N_fender],
                [u0 * r0 * np.sin(psi0) + v0 * r0 * np.cos(psi0)],
                [-u0 * r0 * np.cos(psi0) + v0 * r0 * np.sin(psi0)],
            ],
            dtype=float,
        )

        return A0, C0
