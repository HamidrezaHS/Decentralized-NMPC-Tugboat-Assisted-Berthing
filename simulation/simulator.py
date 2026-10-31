import numpy as np
import matplotlib.pyplot as plt
from model.coupled import Coupled


class Sim:
    def __init__(
        self,
        dt,
        MainShip,
        push_Tugs=None,
        tow_Tugs=None,
        delta_rate=np.inf,
        enable_tug_wind_shadow=False,
        shadow_min_factor=0.20,
        shadow_length_factor=1.20,
        shadow_width_factor=1.20,
    ):
        self.dt = float(dt)
        self.push_Tugs = list(push_Tugs) if push_Tugs is not None else []
        self.tow_Tugs = list(tow_Tugs) if tow_Tugs is not None else []

        self.coupled = Coupled(
            MainShip=MainShip,
            push_Tugs=self.push_Tugs,
            tow_Tugs=self.tow_Tugs,
            delta_rate=delta_rate,
            dt=dt,
            enable_tug_wind_shadow=enable_tug_wind_shadow,
            shadow_min_factor=shadow_min_factor,
            shadow_length_factor=shadow_length_factor,
            shadow_width_factor=shadow_width_factor,
        )
        self.last_tug_wind_factors = [1.0] * (len(self.push_Tugs) + len(self.tow_Tugs))
        self.last_block_sizes = None
        self.last_block_modes = None

    def solve_one_step(self, ship_states, push_tug_states=None, tow_tug_states=None,
                       control_inputs=None, delta_list=None, show_A=False):
        A, C, block_sizes, delta_list, block_modes = self.coupled.build_matrices(
            ship_states=ship_states,
            push_tug_states=push_tug_states,
            tow_tug_states=tow_tug_states,
            control_inputs=control_inputs,
            delta_list=delta_list,
        )

        self.last_block_sizes = block_sizes
        self.last_block_modes = block_modes
        self.last_tug_wind_factors = list(getattr(self.coupled, "last_tug_wind_factors", self.last_tug_wind_factors))

        if show_A:
            self.plot_block_matrix(A, block_sizes, block_modes)

        B = np.linalg.solve(A, C)
        return B, delta_list, block_sizes, block_modes

    @staticmethod
    def split_solution(B, block_sizes, n_push_tugs=0):
        B = np.asarray(B).flatten()
        starts = np.concatenate(([0], np.cumsum(block_sizes)[:-1]))
        blocks = [B[s:s + m] for s, m in zip(starts, block_sizes)]
        ship_B = blocks[0]
        push_B = blocks[1:1 + n_push_tugs]
        tow_B = blocks[1 + n_push_tugs:]
        return ship_B, push_B, tow_B

    def plot_block_matrix(self, A, block_sizes, block_modes=None):
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(A, cmap="coolwarm", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        boundaries = np.cumsum(block_sizes)
        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color="black", linewidth=2)
            ax.axvline(b - 0.5, color="black", linewidth=2)

        if block_modes is None:
            labels = ["Ship"] + [f"Tug {i + 1}" for i in range(len(block_sizes) - 1)]
        else:
            labels = []
            p = 0
            t = 0
            for mode in block_modes:
                if mode == "ship":
                    labels.append("Ship")
                elif mode == "push":
                    p += 1
                    labels.append(f"Push {p}")
                elif mode == "tow":
                    t += 1
                    labels.append(f"Tow {t}")
                else:
                    labels.append(str(mode))

        starts = np.concatenate(([0], boundaries[:-1]))
        centers = starts + np.array(block_sizes) / 2 - 0.5

        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)

        nrows, ncols = A.shape
        if nrows <= 45 and ncols <= 45:
            for i in range(nrows):
                for j in range(ncols):
                    value = A[i, j]
                    text = "0" if abs(value) < 1e-10 else f"{value:.2f}"
                    ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

        ax.set_title("Coupled block matrix A", fontsize=14)
        ax.set_xlabel("Unknown vector blocks")
        ax.set_ylabel("Equation blocks")
        plt.tight_layout()
        plt.show()
