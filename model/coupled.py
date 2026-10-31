import numpy as np


class Coupled:
    """
    Coupled block-matrix builder for one main ship, any number of pushing tugs,
    and any number of towing tugs.

    Global unknown vector ordering:
        B = [B0, Bp1, ..., Bp_np, Bt1, ..., Bt_nt]^T

    Ship block:
        B0 = [u0_dot, v0_dot, r0_dot, ue0_dot, ve0_dot]^T

    Pushing tug block, returned by Tug.push_tug_matrices:
        Bpi = [ui_dot, vi_dot, ri_dot, uei_dot, vei_dot, Fxi, Fyi]^T

    Towing tug block, returned by Tug.tow_tug_matrices:
        Bti = [ui_dot, vi_dot, ri_dot, uei_dot, vei_dot, theta_ddot_i, Tsi]^T

    The ordering of input lists is always:
        [all pushing tugs] + [all towing tugs]
    for control_inputs and delta_list.
    """

    def __init__(
        self,
        MainShip,
        push_Tugs=None,
        tow_Tugs=None,
        delta_rate=np.inf,
        dt=0.1,
        enable_tug_wind_shadow=False,
        shadow_min_factor=0.20,
        shadow_length_factor=1.20,
        shadow_width_factor=1.20,
    ):
        self.main_ship = MainShip
        self.push_tugboats = list(push_Tugs) if push_Tugs is not None else []
        self.tow_tugboats = list(tow_Tugs) if tow_Tugs is not None else []

        self.n_push_tugs = len(self.push_tugboats)
        self.n_tow_tugs = len(self.tow_tugboats)
        self.n_tugs = self.n_push_tugs + self.n_tow_tugs

        self.dt = float(dt)
        self.max_delta_change = np.inf if np.isinf(delta_rate) else abs(float(delta_rate)) * self.dt

        self.enable_tug_wind_shadow = bool(enable_tug_wind_shadow)
        self.shadow_min_factor = float(np.clip(shadow_min_factor, 0.0, 1.0))
        self.shadow_length_factor = float(max(0.0, shadow_length_factor))
        self.shadow_width_factor = float(max(0.0, shadow_width_factor))
        self.last_tug_wind_factors = [1.0] * self.n_tugs

    def _rate_limit_delta(self, delta_cmd, delta_old):
        """Rate-limit the commanded azimuth/rudder angle without angle-wrap jumps."""
        delta_cmd = float(delta_cmd)
        delta_old = float(delta_old)
        if np.isinf(self.max_delta_change):
            return delta_cmd

        # Use nearest angular difference so +179 -> -179 deg is treated as 2 deg, not 358 deg.
        diff = (delta_cmd - delta_old + np.pi) % (2.0 * np.pi) - np.pi
        diff_limited = np.clip(diff, -self.max_delta_change, self.max_delta_change)
        return delta_old + diff_limited

    @staticmethod
    def _unit(angle):
        return np.array([np.cos(angle), np.sin(angle)], dtype=float)

    def _tug_center_in_ship_body(self, mode, state, tug, psi0):
        """Approximate tug centre relative to the ship centre in ship-body axes."""
        if mode == "push":
            lpi, psi_pi, ui, vi, ri, psi_i = state
            contact = np.array([lpi * np.cos(psi_pi), lpi * np.sin(psi_pi)], dtype=float)
            tug_axis_ship = self._unit(float(psi_i) - float(psi0))
            return contact - float(getattr(tug, "api", 0.0)) * tug_axis_ship

        lti, psi_ti, ui, vi, ri, psi_i, theta_dot_i, theta_i = state
        contact = np.array([lti * np.cos(psi_ti), lti * np.sin(psi_ti)], dtype=float)
        rope_axis_ship = self._unit(float(psi_i) + float(theta_i) - float(psi0))
        tug_axis_ship = self._unit(float(psi_i) - float(psi0))
        return (
            contact
            - float(getattr(tug, "lci", 0.0)) * rope_axis_ship
            - float(getattr(tug, "ati", 0.0)) * tug_axis_ship
        )

    def _wind_factor_for_tug(self, tug_pos_body, psi0, Vw, psi_w):
        """Simple geometric wind-shadow factor caused by the main ship.

        psi_w follows the project convention: earth-fixed direction TOWARD which
        the air moves. A tug downstream of the main ship and laterally close to
        the ship centreline receives the reduced wind factor.
        """
        if (not self.enable_tug_wind_shadow) or abs(float(Vw)) < 1.0e-12:
            return 1.0

        ship_L = float(getattr(self.main_ship, "Lpp", 320.0))
        ship_B = float(getattr(self.main_ship, "B", 0.18 * ship_L))
        wind_dir_body = self._unit(float(psi_w) - float(psi0))
        cross_dir_body = np.array([-wind_dir_body[1], wind_dir_body[0]], dtype=float)

        p = np.asarray(tug_pos_body, dtype=float).reshape(2)
        downstream = float(np.dot(p, wind_dir_body))
        lateral = abs(float(np.dot(p, cross_dir_body)))

        max_downstream = self.shadow_length_factor * ship_L
        max_lateral = self.shadow_width_factor * 0.5 * ship_B
        if downstream >= 0.0 and downstream <= max_downstream and lateral <= max_lateral:
            return self.shadow_min_factor
        return 1.0

    def _check_input_lengths(self, push_tug_states, tow_tug_states, control_inputs, delta_list):
        if len(push_tug_states) != self.n_push_tugs:
            raise ValueError(
                f"Expected {self.n_push_tugs} push_tug_states, got {len(push_tug_states)}."
            )
        if len(tow_tug_states) != self.n_tow_tugs:
            raise ValueError(
                f"Expected {self.n_tow_tugs} tow_tug_states, got {len(tow_tug_states)}."
            )
        if len(control_inputs) != self.n_tugs:
            raise ValueError(
                f"Expected {self.n_tugs} control_inputs, got {len(control_inputs)}."
            )
        if len(delta_list) != self.n_tugs:
            raise ValueError(
                f"Expected {self.n_tugs} previous delta values, got {len(delta_list)}."
            )

    def build_matrices(self, ship_states, push_tug_states=None, tow_tug_states=None,
                       control_inputs=None, delta_list=None):
        """
        Parameters
        ----------
        ship_states : array-like
            (u0, v0, r0, psi0)

        push_tug_states : list[array-like]
            Each pushing tug state:
                (lpi, psi_pi, ui, vi, ri, psi_i)

        tow_tug_states : list[array-like]
            Each towing tug state:
                (lti, psi_ti, ui, vi, ri, psi_i, theta_dot_i, theta_i)

        control_inputs : list[tuple]
            First all pushing tugs, then all towing tugs.
            Each item is (Ti, delta_i).

        delta_list : list[float]
            Previous limited delta_i values with the same ordering as control_inputs.
        """
        push_tug_states = [] if push_tug_states is None else list(push_tug_states)
        tow_tug_states = [] if tow_tug_states is None else list(tow_tug_states)
        control_inputs = [] if control_inputs is None else list(control_inputs)
        delta_list = [0.0] * self.n_tugs if delta_list is None else list(delta_list)

        self._check_input_lengths(push_tug_states, tow_tug_states, control_inputs, delta_list)

        ship_states = np.asarray(ship_states, dtype=float).flatten()
        if ship_states.size >= 8:
            u0, v0, r0, psi0, Vw, psi_w, Vc, psi_c = ship_states[:8]
        elif ship_states.size >= 4:
            u0, v0, r0, psi0 = ship_states[:4]
            Vw = psi_w = Vc = psi_c = 0.0
        else:
            raise ValueError("ship_states must contain at least (u0, v0, r0, psi0).")

        A0, C0 = self.main_ship.ship_matrices(u0, v0, r0, psi0, Vw, psi_w, Vc, psi_c)
        m0 = A0.shape[0]

        tug_mats = []
        tug_sizes = []
        wind_factors = []

        # ------------------------
        # Pushing tugs
        # ------------------------
        for i, tug in enumerate(self.push_tugboats):
            lpi, psi_pi, ui, vi, ri, psi_i = push_tug_states[i]
            Ti, delta_cmd = control_inputs[i]
            delta_i = self._rate_limit_delta(delta_cmd, delta_list[i])
            delta_list[i] = delta_i

            tug_pos_body = self._tug_center_in_ship_body("push", push_tug_states[i], tug, psi0)
            wind_factor = self._wind_factor_for_tug(tug_pos_body, psi0, Vw, psi_w)
            wind_factors.append(wind_factor)

            Api, Ap0i, Api0, Cpi = tug.push_tug_matrices(
                lpi, psi_pi, r0, psi0, ui, vi, ri, psi_i, Ti, delta_i,
                Vw=Vw, psi_w=psi_w, Vc=Vc, psi_c=psi_c, wind_factor=wind_factor,
            )
            tug_mats.append((Api, Ap0i, Api0, Cpi, "push"))
            tug_sizes.append(Api.shape[0])

        # ------------------------
        # Towing tugs
        # ------------------------
        for j, tug in enumerate(self.tow_tugboats):
            idx = self.n_push_tugs + j
            lti, psi_ti, ui, vi, ri, psi_i, theta_dot_i, theta_i = tow_tug_states[j]
            Ti, delta_cmd = control_inputs[idx]
            delta_i = self._rate_limit_delta(delta_cmd, delta_list[idx])
            delta_list[idx] = delta_i

            tug_pos_body = self._tug_center_in_ship_body("tow", tow_tug_states[j], tug, psi0)
            wind_factor = self._wind_factor_for_tug(tug_pos_body, psi0, Vw, psi_w)
            wind_factors.append(wind_factor)

            Ati, At0i, Ati0, Cti = tug.tow_tug_matrices(
                lti, psi_ti, r0, psi0,
                ui, vi, ri, psi_i,
                theta_dot_i, theta_i,
                Ti, delta_i,
                Vw=Vw, psi_w=psi_w, Vc=Vc, psi_c=psi_c, wind_factor=wind_factor,
            )
            tug_mats.append((Ati, At0i, Ati0, Cti, "tow"))
            tug_sizes.append(Ati.shape[0])

        n = self.n_tugs
        A_blocks = [[None] * (n + 1) for _ in range(n + 1)]

        # Ship row: ship equations coupled with each tug block.
        A_blocks[0][0] = A0
        for i in range(n):
            _, A_tug_to_ship, _, _, _ = tug_mats[i]
            A_blocks[0][i + 1] = A_tug_to_ship

        # Tug rows.
        for i in range(n):
            A_self, _, A_ship_to_tug, _, _ = tug_mats[i]
            mi = tug_sizes[i]

            A_blocks[i + 1][0] = A_ship_to_tug
            A_blocks[i + 1][i + 1] = A_self

            for j in range(n):
                if i != j:
                    A_blocks[i + 1][j + 1] = np.zeros((mi, tug_sizes[j]))

        A = np.block(A_blocks)

        C_blocks = [C0]
        for _, _, _, C_tug, _ in tug_mats:
            C_blocks.append(C_tug)
        C = np.concatenate(C_blocks, axis=0)

        block_sizes = [m0] + tug_sizes
        block_modes = ["ship"] + [mode for *_, mode in tug_mats]
        self.last_tug_wind_factors = list(wind_factors)

        return A, C, block_sizes, delta_list, block_modes
