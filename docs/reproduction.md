# Reproduction Guide

[Project overview](../README.md) · [Methods](methods.md) · [Results](results.md)

## Environment

Use Python 3.12 in a virtual environment and install [`requirements.txt`](../requirements.txt). It specifies compatible major-version ranges for NumPy, SciPy and Matplotlib. Publication checks used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and Matplotlib 3.11.2 on macOS arm64. The archive's original NumPy version is recorded separately in [`protocol.json`](../protocol.json). Small floating-point differences across BLAS implementations are expected.

All commands below run from the repository root with the virtual environment active. Windows activation is `.venv\Scripts\activate`; shell line-continuation syntax can be replaced by entering a command on one line.

## Inspect the included archive

```bash
python physics.py
python check_inference.py
python examples/math_examples.py
python make_figures.py
```

`physics.py` prints mass balance, water residual, positivity and elapsed time for its default demonstration. It does not run the full main experiment. `check_inference.py` recomputes prior normalization, release-timing independence and likelihood composition from the main response library and case observations. It writes `results/inference_checks.json`. The mathematical example script checks analytical fixtures and rewrites its small JSON report.

`make_figures.py` reads the archived data and regenerates all eight PNG figures under `figures/`, the aggregate figure summary and three English CSV sampling reports under `results/`. It performs no new PDE solves or design runs. PNG figures are sized for readable GitHub display.

The release flattens original development and held-out main cases into one directory. Seeds are unique, so `results/case_1102.npz` and `results/case_3001.npz` are unambiguous. Main aggregate tables are `development_results.json` and `heldout_results.json`. Continuous-stress results and enumeration use the `stress_` prefix. Fresh reproductions retain separate development/held-out directories to keep their outputs independent.

## Fresh main experiment

```bash
python reproduce_main.py --out results/reproduction
```

The destination must not exist. The wrapper creates it, limits BLAS threading to one, builds a new response library, runs three development cases, freezes the protocol, runs twelve held-out cases and checks structural invariance. Main outputs appear in:

```text
results/reproduction/
  library.npz
  protocol.json
  development/case_1101.json ... case_1103.npz
  heldout/case_3001.json ... case_3012.npz
  inference_checks.json
```

A case JSON contains truth index, initial data, current target summary, four sets of predictive metrics and detailed greedy candidate scores. A case NPZ contains observations, posterior weights, illustrative updates and evaluation arrays. The response library contains prior weights, parameter states, observation predictions, future target values, basis fields and configuration metadata.

The public packaging changes descriptive status metadata and normalizes file paths. Mathematical settings, observations and archived numerical arrays remain the original values. `protocol.json` retains the historical execution configuration/source hashes separately from the public descriptive configuration hash. The library's configuration hash is synchronized with the public configuration so fresh entry points can validate it. Historical source hashes identify the original execution files, rather than claiming byte identity with scripts whose paths and output options were adjusted for this release.

## Exploratory stress tests

```bash
python reproduce_main.py --out results/reproduction_with_stress --stress
```

This adds the small-pool enumeration, 64 continuous same-model cases and 16 finer-grid/time-step truth cases in the new directory. These tests perform fresh PDE solves and monitoring design for each case; they take longer than rebuilding the figures. They do not tune or replace the main protocol.

To reuse the included main library while writing new stress results elsewhere:

```bash
python stress_tests.py --library results/library.npz --out results/reproduction/stress --development-case results/case_1102.npz
```

Use a fresh `--out` for this lower-level command: it writes named files and does not refuse an existing directory. The release includes aggregate per-case stress metrics, the summary and the enumeration arrays. It omits the larger collection of individual stress posterior arrays and selection traces; the command regenerates those.

## Numerical checks without replacing evidence

```bash
python validate_physics.py --out physics_results/reproduction
python check_fine_grid.py --out physics_results/reproduction
```

The first command runs conservation, analytical diffusion, linear head, positivity, superposition and controlled refinement checks. The second extends the fixed mesh probe to 6.25 m cells and is more expensive. Both accept `--out`; omitting it rewrites their named archived files in `physics_results/`.

The fine-grid result is numerical sensitivity evidence, not proof of convergence of every quantity of interest. The source and observation footprints are held fixed under refinement.

## Optional MODFLOW 6 comparison

The core solver does not require MODFLOW. For the external reference check, install FloPy separately:

```bash
python -m pip install -r requirements-modflow.txt
python benchmark_mf6.py --exe /path/to/mf6 --out physics_results/reproduction_modflow
```

Obtain the executable for your platform from the [official MODFLOW 6 releases](https://github.com/MODFLOW-ORG/modflow6/releases). No executable or vendored third-party library is bundled here. The original saved audit used MODFLOW 6 version `6.8.0+8680167.dirty` and FloPy 3.11.0; the executable hash and reference configuration are in [`mf6_audit.json`](../physics_results/mf6_audit.json). Consult the official distribution for its software notices and licensing.

The script writes two reference workspaces: the public P09-derived configuration and a matched upstream/zero-dispersion configuration. It asserts agreement only for the latter, saving the comparison fields and JSON audit in the chosen output directory. It records the executable's actual version and hash on every run. The source URL identifies the fixed official example commit; it is not downloaded or executed by this script. The separate MT3DMS executable is not invoked.

## Local validation performed for this release

The structural checks passed for all 15 archived main cases. Analytical examples passed; the physical validation suite passed in a separate fresh directory; all eight figures rebuilt from saved arrays. One complete development case (seed 1102) was rerun from the included library, preserving every selected set and reproducing key predictive metrics within floating-point tolerance. Python source files were syntax checked, relative Markdown links were checked and figure files were decoded for inspection.

The complete main experiment, 80 stress cases and external MODFLOW comparison were not rerun during packaging. Their published tables are the existing archived results; this distinction matters when interpreting environment changes and runtime figures.
