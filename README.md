# Decentralized NMPC for Cooperative Tug-Assisted Ship Berthing

**Research code and numerical results associated with the manuscript:**
*Decentralized nonlinear model predictive control for cooperative tug-assisted ship berthing under varying environmental conditions*.

This repository implements a four-tug decentralized nonlinear model predictive control (NMPC) system for low-speed berthing of a KVLCC2 tanker. Two tugs push through unilateral contact and two tugs tow through tension-only lines. Each local NMPC uses the **measured ship state**, the state of its **own tug**, and environmental inputs. It does **not** read other tugs' states or planned actions. The physics simulation still integrates the five coupled moving vessels.

> **Scope and evidence.** The archived dataset contains 16 cardinal combinations of wind/current direction (wind 8 m/s, current 0.25 m/s, initial heading 20°; one deterministic seed), six regression scenarios, and one combined-disturbance run at an initial heading of 40°. All 23 recorded cases were labelled SUCCESS by the simulation monitor. These are *initial berthing-envelope* outcomes, not measured full-scale operations, a general safety guarantee, a multi-seed success estimate, or proof of sustained station keeping. Code for additional comparisons is not evidence that those comparisons were reported in the article.

## Repository map

| Location | Contents |
|---|---|
| `decentralized_nmpc/`, `ship_state_local/` | Independent predictive controllers and ship-state observation interface |
| `model/`, `physical_v4/`, `geom/` | Ship/tug motion, unilateral forces, and original hull geometry |
| `simulation/`, `visualization/` | Numerical plant and visualizations |
| `tests/` | Automated tests from the recorded project |
| `ENV_DIRECTION_SWEEP.py`, `ENV_DIRECTION_VERIFY.py`, `ENV_DIRECTION_REPORT.py` | Directional experiments, audit and outcome plots |
| `Data/RawRuns/` | Original case-level configuration, trajectory, controller decisions and metrics |
| `Data/RECORDED_CASES.csv`, `Data/SHA256_MANIFEST.csv` | Archived case index and file integrity manifest |
| `Data/` and `Scripts/make_publication_figures.py` | Original figure-generating numerical inputs and rendering script |
| `Figures/` | Reproduced publication-quality PDF figures |
| `docs/` | Dataset dictionary, privacy/licence check, and publication instructions |

## Prerequisites

- Python 3.11 or newer (the package is run through the Python entry points below)
- Scientific Python dependencies listed in `requirements.txt`
- For Windows PowerShell, run commands from the **repository root**.

## Quick start (Windows PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe Scripts/verify_repository.py
```

## Quick start (Linux/macOS)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python Scripts/verify_repository.py
```

### Note on Git line endings and archived data integrity

The original archive SHA-256 manifest records byte-exact CSV and JSON inputs.
Some Git checkouts convert CSV line endings between Windows CRLF and Unix LF.
`Scripts/verify_repository.py` accepts CSVs **only when their contents match the
original manifest after a strictly verified line-ending-only conversion**.
Changes to numeric values, missing files or non-CSV data still fail. The
`.gitattributes` file prevents future conversions for archived CSV files.
See [`docs/CSV_LINE_ENDING_FIX_FA.md`](docs/CSV_LINE_ENDING_FIX_FA.md) for help.

## Inspect the archived results without rerunning the simulator

```bash
python ENV_DIRECTION_VERIFY.py --results Data/RawRuns/environment_4x4_v31
python Scripts/plot_directional_archive.py
python Scripts/make_publication_figures.py
```

The publication-figure command regenerates PDF and PNG files in `Figures/` from the numerical inputs in `Data/`. The directional-report wrapper writes outputs under `results/archived_directional_report/` without altering the raw archive. Neither command reruns the physical model.

## Reproduce a representative single run

```bash
python run_nmpc.py --controller-method ship_state_local --psi0 20 --target-x 50 --target-y -100 --wind-mps 8 --wind-to-deg 90 --current-mps 0.25 --current-to-deg 90 --duration 6500 --maxiter 6 --seed 123 --out results/example_w90_c90
```

The wind/current angles specify the direction **toward which** the disturbance acts, measured counter-clockwise from earth-fixed +X (not meteorological *from* convention).

## Reproduce the 16-direction numerical sweep

```bash
python -u ENV_DIRECTION_SWEEP.py --preset combined --psi0 20 --wind-mps 8 --current-mps 0.25 --wind-angles 0 90 180 270 --current-angles 0 90 180 270 --duration 6500 --maxiter 6 --seed 123 --out results/reproduced_directional_grid
python ENV_DIRECTION_REPORT.py --results results/reproduced_directional_grid
python ENV_DIRECTION_VERIFY.py --results results/reproduced_directional_grid
```

The full sweep is computationally intensive. Its outcome may depend on Python/library versions, numerical tolerances and hardware. Do not replace archived measurements with rerun values without reporting the differences.

To regenerate a **four-stage berthing figure** from a recorded trajectory:

```bash
python plot_marl.py --input Data/RawRuns/environment_4x4_v31/combined_psi20_w8_wd90_c0p25_cd90/trajectory.csv --out results/plots_w90_c90 --target-x 50 --target-y -100 --target-psi-deg 0
```

## Software citation and archived release

Use GitHub's *Cite this repository* facility (metadata in [`CITATION.cff`](CITATION.cff)). After author approval, create a tagged GitHub Release and archive that exact release on Zenodo. Add the resulting DOI here and to the manuscript. **No DOI has been assigned by this package.**

## License and use

No redistribution or reuse license is granted by default until the copyright holders confirm the terms; see [`docs/LICENSE_DECISION_REQUIRED.md`](docs/LICENSE_DECISION_REQUIRED.md). The example MIT text under `docs/templates/` is *not* the project license. The authors must confirm the terms before releasing publicly. Check all geometry and any third-party components before publishing.

## Contact

Please open a GitHub Issue for questions after the repository is made public. Do not commit personal contact details unless their owners have approved.
