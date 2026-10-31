# Data dictionary and experimental provenance

## Campaigns

- `Data/RawRuns/environment_4x4_v31/`: 16 combinations of wind (8 m/s) and current (0.25 m/s), toward-directions independently selected from {0°, 90°, 180°, 270°}; initial heading 20°.
- `Data/RawRuns/v31_regression_suite/`: six baseline/regression cases.
- `Data/RawRuns/v31_heading40_combo/`: a supplementary 40° initial-heading combined environment case.

The directory names preserve internal simulation labels for traceability. These identifiers are not part of the scientific article text.

## Per-case artifacts

- `configuration.json`: parameters passed to the simulator.
- `metrics.json`: terminal status and numerical statistics; `SUCCESS` means that the simulator's strict initial-berthing acceptance monitor was satisfied.
- `trajectory.csv`: time histories of ship and tug pose/velocity, actuator states and interactions.
- `local_nmpc_decisions.csv`: solver-level decisions per tug, including commanded thrust fraction and azimuth settings.
- `progress.json`: simulator's last recorded progress metadata, when available.

## Dataset indices

- `RECORDED_CASES.csv`: one row per archived run.
- `SHA256_MANIFEST.csv`: SHA-256 checksum of every archived input file. Excludes the manifest itself and `RECORDED_CASES.csv` to avoid circular checksums.
- `directional_matrix.csv`: compact 16-case matrix as used in the manuscript graphics.

## Statistical limitations

The primary sweep uses one recorded seed and fixed wind/current magnitudes. The 16/16 result refers to these 16 **numerical cases only**, not to a probabilistic success estimate or performance with wave loading, noisy sensors, uncertain model coefficients or tug faults. Numerical results are from the simulator, not experimental measurements.

## Units and angles

Ship and tug positions: meters. Speed: m/s. Heading and wind/current directions: degrees. Forces: newtons. Recorded time: seconds. Wind and current angles indicate *toward* directions, counter-clockwise from positive earth-fixed X.
