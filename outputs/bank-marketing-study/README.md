# Before the Call

A small, reproducible study of term-deposit subscription prediction using UCI Bank
Marketing: classical baselines, a PyTorch MLP, and an honest look at temporal drift.

**Research question:** Does positive-class weighting improve MLP average precision
on imbalanced tabular data when predicting a later period?

The completed study uses `bank-additional-full.csv` (41,188 records), a conservative
pre-call feature policy, and chronological 60/20/20 boundaries. It excludes call
duration, cumulative campaign contacts and economic indicators with unverified
publication timing. Later identical predictor patterns are removed without consulting
labels, leaving 34,513 observations. Every learned preprocessing step fits training
data only. This is an educational experiment, not a production financial system.

## Findings from the completed run

| Model | Test average precision | Test ROC-AUC | Test F1 |
|---|---:|---:|---:|
| Prior dummy | 0.3203 | 0.5000 | 0.4852 |
| Logistic regression | 0.4659 | 0.6469 | 0.4381 |
| Histogram gradient boosting | 0.4115 | 0.6242 | 0.5012 |
| Weighted MLP, predeclared seed 42 | 0.5006 | 0.6699 | 0.5088 |

Class weighting increased seed-42 **validation AP from 0.1400 to 0.1903**. The chosen
configuration was repeated with seeds 42, 43 and 44. Test AP was **0.4716 ± 0.0267**
(mean ± sample standard deviation), ranging from 0.4482 to 0.5006. Logistic regression
beat two of the three MLP seeds. The primary MLP won this comparison, but the seed
variability does not support a strong claim of neural-network superiority.

Subscription prevalence shifted from **4.79% → 10.77% → 32.03%** across the retained
train, validation and test sets. Only six training records had an observed previous-
contact distance; later periods had many more. Standardizing such sparse history
can amplify extrapolation. All four MLP fits selected epoch 1 and stopped after
epoch 9. These limitations are part of the finding, not hidden behind the best score.

The complete computation took **2.59 seconds** inside the experiment runner on an
Apple M3 CPU with two threads. This excludes dependency installation, download,
Python imports, tests and report rendering. A second clean environment reproduced
every saved validation and test prediction exactly on the same machine. Hardware,
versions, configurations, seed histories, thresholds and runtime are recorded.

![Final precision–recall comparison](results/full/plots/precision_recall.png)

Read the [generated results report](results/full/report.md) for precision, recall,
confusion matrices, loss curves, subgroup diagnostics and all seed results.

## Reproduce

Run from this repository's root. Tested with Python 3.13.7 on macOS arm64.
The exact dependency lock includes transitive packages; it is a tested environment
snapshot, not a promise of identical wheels on every OS. The project metadata
supports Python 3.11–3.13. A first installation needs internet access to PyPI and UCI.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check

# Verify/download the source, audit it, and persist the fixed splits.
bank-study prepare

# Small end-to-end execution check (not a research result).
bank-study run --config configs/smoke.json
bank-study report --run-dir results/smoke

# Full experiment; preserve the bundled reference run in results/full.
bank-study run --config configs/full.json --name reproduce
bank-study report --run-dir results/reproduce
bank-study verify --run-dir results/reproduce

# Tests and an exact same-machine reproduction check.
python -m pytest -q
python scripts/check_reproduction.py results/full results/reproduce

# Regenerate and verify the bundled report without training or raw data.
bank-study report --run-dir results/full
bank-study verify --run-dir results/full
```

On Windows, activate with `.venv\Scripts\activate`. Runs refuse to overwrite an
existing results directory. Use a new `--name` for another run; no deletion is needed.
The delivered local workspace already contains the tested `smoke` and `reproduce`
runs; choose fresh names there. They are absent from a fresh Git checkout or ZIP.
The full config permits 60 epochs with patience 8; early stopping determined the
actual count. No search-space reduction was needed. CPU is the tested default;
MPS is optional via `device` in a copied config if the runtime exposes it.

## Protocol and evidence

1. Verify archive and CSV checksums, retain the official row order, and persist
   label-blind split indices and the data audit.
2. Fit median imputation, numeric standardization and one-hot encoding on training
   rows. All four model families share the same processed inputs.
3. Compare unweighted versus balanced positive-class BCE at seed 42. Keep network,
   regularization, optimizer and early-stopping policy fixed; select by validation AP.
4. Repeat the selected weighting with two more seeds. Select checkpoints and
   maximum-F1 thresholds on validation only; write `selection.json` before test inference.
5. Evaluate the frozen models on the final holdout. Report all seeds, never select
   the best test seed. Recompute saved metrics from saved predictions when reporting.

Average precision is the recall-weighted stepwise precision summary from scikit-learn,
not trapezoidal PR-AUC. Threshold-specific metrics use `score >= threshold`. The
dummy's F1-optimal validation threshold predicts everyone positive, which explains
its high recall and surprisingly competitive F1 under the later prevalence shift.
No cost model is implied by this threshold rule.

```text
configs/                 Small smoke and full experiment configurations
src/bankstudy/           Data, training, evaluation, experiment, report and CLI
tests/                   Leakage, split, checkpoint, determinism and metric tests
data/                    Provenance, audit and original-row split indices
results/full/            Reference predictions, metrics, histories, report and plots
scripts/                 Exact reproduction check
docs/                    Data card, technical walkthrough and application notes
artifacts/               Local preprocessors/models/checkpoints (Git-ignored)
```

Raw data, environments, caches, smoke/reproduction outputs and checkpoints are
excluded from Git. Local model files remain available after running. Saved
predictions make the reference metrics auditable without publishing checkpoints.
Only load joblib model files you trust.

## Limitations and reading guide

One date-ordered split, missing client identifiers and exact timestamps, possible
outcome maturation across boundaries, profile exclusions, retrospective feature
availability assumptions, sparse-history scaling and large prevalence drift all
limit interpretation. AP also depends on prevalence. The weighting comparison uses
one selection seed; only its winner is repeated. Baseline hyperparameters are fixed,
so this is not an exhaustive tuning benchmark. No random-split performance claim,
calibration, fairness audit, causal effect estimate or production validation is made.

- [Data card](docs/data-card.md): verified variants, feature definitions, license and leakage risks.
- [Technical walkthrough and interview questions](docs/technical-walkthrough.md): readable explanations tied to the code.
- [Application notes and two factual CV bullets](docs/application-notes.md).
- [Verification record](docs/verification.md): what was actually executed.

Developed with AI assistance, including implementation, experiment execution,
testing and documentation. It should not be represented as independently authored
without assistance. Review the code, rerun it and explain the choices in your own words.

Dataset: Moro, S., Rita, P., & Cortez, P. (2014),
[Bank Marketing, UCI](https://doi.org/10.24432/C5K306), CC BY 4.0. Code: MIT.
