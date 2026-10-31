# Tested software environment

The package was validated locally with the following versions (the CI workflow also tests Python 3.11):

- Python 3.13.5
- NumPy 2.3.5
- SciPy 1.17.0
- Matplotlib 3.10.8
- ezdxf 1.4.4
- Shapely 2.1.2
- pytest 9.0.2

The 76 tests passed in this local environment. The test workflow for Python 3.11 is included but its hosted run cannot be verified until the repository is published to GitHub. These are software/consistency tests, not 76 independent berthing demonstrations.

Full 16-case physics recomputation is *not* part of the fast CI test; it requires substantially more compute time and should be run separately using the documented command.
