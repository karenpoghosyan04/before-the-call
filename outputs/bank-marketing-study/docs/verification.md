# Verification record

Executed locally on 2026-10-14. This describes completed work, not planned work.

## Environment

- Apple M3, 8 GiB RAM, eight logical CPUs, macOS 26.2, arm64.
- Hardware model and memory were checked with read-only `sysctl` commands outside
  the restrictive execution sandbox. The runner itself records those fields as
  unavailable inside the sandbox; this distinction is preserved in its JSON.
- CPython 3.13.7; CPU training with two threads. PyTorch's MPS backend was built
  but was not exposed in the experiment process. No MPS experiment was run.
- Two newly created virtual environments, without system site packages. The second
  installed `requirements-lock.txt`, then the editable project with
  `--no-build-isolation --no-deps`. `pip check` found no broken requirements.
- The exact transitive dependency snapshot is in `requirements-lock.txt`; principal
  package versions are also saved under `results/full/environment.json`.

## Commands exercised

| Check | Observed outcome |
|---|---|
| Official archive download and nested CSV verification | Archive and CSV SHA-256 matched |
| `bank-study prepare` from an empty separate data directory | Download, extraction, audit and splits succeeded |
| `bank-study prepare` with cached data | Same split indices and source checksums |
| `bank-study run --config configs/smoke.json` | Three classical fits and two small MLP fits completed |
| Smoke report and metric verification | Completed successfully |
| `bank-study run --config configs/full.json` | Three classical fits and four MLP fits completed |
| Full report and metric verification | Saved scores and locked thresholds reproduced from predictions |
| Full run with `--name reproduce` in second clean environment | Completed with identical validation and test predictions |
| `python scripts/check_reproduction.py results/full results/reproduce` | 15 prediction/label arrays bitwise identical; 7 checkpoint prediction checks passed |
| `python -m pytest -q` | 13 passed, including saved full-run consistency |
| Visual review | PR curve, loss/AP curves and confusion matrices inspected; labels and counts readable |
| Git exclusions | Environment, raw ZIP, checkpoints and smoke outputs correctly ignored |

The first test run, before full experiments existed, had 12 passing tests and one
intentional skip. After the reference results existed, all 13 passed in both environments.
An intermediate report regeneration required refreshing the audit after adding
descriptive drift fields; preparation and report generation were rerun successfully.
This affected reporting only, not preprocessing or model predictions.

The original full run's measured computation time is 2.5918735419982113 seconds.
It includes data loading, preprocessing, fits, validation and final evaluation
inside `run`, excluding imports, installation, download, plotting and testing.
All MLP fits ran nine epochs and selected epoch one; the configured maximum was 60.
No model-size or search-space reduction was necessary. Installation took much longer
than training because package downloads were relatively slow.

`results/reproduction-check.json` stores the successful comparison record. The
second run and both environments remain local but are excluded from the distribution.
Exact equality is demonstrated on this machine with these versions; it is not a
cross-platform determinism guarantee.

## Selection and audit boundaries

The dataset was inspected for schema, duplicate patterns and partition class rates.
Model/weighting selection, early stopping and all decision thresholds used validation
data only. `selection.json` was written before test inference; `runtime.json` hashes
that manifest. The final test was evaluated for the fixed baselines and three selected
MLP seeds. The second execution repeated the identical experiment solely to verify
reproducibility. No configurations or thresholds were changed in response to test
scores. Subsequent additions described drift and clarified reporting.

No models were published and no remote Git repository was configured. The local
repository is initialized on `codex/bank-marketing-study`; files are left uncommitted
so the owner can review the initial project before creating its first commit.
