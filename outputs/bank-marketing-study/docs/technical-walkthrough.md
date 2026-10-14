# Technical walkthrough

Read the source in this order: `data.py`, `model.py`, `evaluate.py`,
`experiment.py`, then `report.py`. There are no external services or hidden notebooks.

## 1. What is being estimated?

The model ranks recorded term-deposit subscription outcomes using information we
assume is available immediately before a scheduled call. The target is `y == "yes"`.
This is an observational association problem. It does not estimate the incremental
effect of calling somebody, because there is no comparable uncontacted control group.
The [data card](data-card.md) explains feature availability and temporal assumptions.

## 2. Preprocessing and leakage

Splitting precedes fitting the imputer, scaler and categorical encoder. Training
medians replace numeric missing values; training means and standard deviations set
numeric scales; training categories define the one-hot vocabulary. Validation and
test pass through the same fitted transformer. A category seen only in validation
does not become a learned feature. Tests inject a huge held-out value and an unseen
category to demonstrate this isolation.

An allowlist selects predictors, so neither `duration`, `y`, row IDs nor unexpected
new columns can enter the model. This is stronger than casually dropping one column.
The value `pdays=999` means no prior contact, not “999 actual days ago.” An indicator
preserves that information while median imputation supplies a numeric distance.

Chronological partitions retain distribution shift. Deduplication retains the first
identical predictor pattern and ignores labels. It can discard legitimate repeated
profiles, so its effect is documented rather than presented as unquestionable cleaning.

## 3. Classical comparisons

The prior dummy outputs the training positive fraction for every row. It checks that
rank-aware metrics behave sensibly under imbalance. Logistic regression uses L2
regularization with `C=1`. Histogram gradient boosting uses shallow-ish trees
(`max_leaf_nodes=15`), 150 boosting iterations, learning rate 0.08 and L2=1.
Its internal early stopping is disabled, avoiding an extra random holdout inside the
chronological training block. These baseline settings are fixed before evaluation.

All models receive identical processed rows and use the same validation threshold
rule. This is a practical comparison, not an equal hyperparameter-search-budget
benchmark: the MLP has two weighting candidates plus checkpoint selection, while
baseline hyperparameters are fixed. No test-result-driven baseline tuning occurred.

## 4. PyTorch training loop

The MLP has two hidden layers (64 and 32 units), ReLU activations and dropout 0.1.
A final linear unit outputs one **logit** per example. The forward method preserves
the batch dimension even for a singleton batch. `BCEWithLogitsLoss` combines sigmoid
and binary cross-entropy stably, so the training forward pass must not add a sigmoid.
Only inference converts logits to probabilities.

For each epoch, a seeded generator shuffles training indices. Each minibatch performs
`zero_grad`, forward pass, loss calculation, `backward`, then `optimizer.step`.
AdamW uses learning rate 0.001 and weight decay 0.0001. Dropout is active in training
and disabled in `eval()`. Validation uses `no_grad()`, does not update parameters,
and evaluates average precision on the whole validation partition.

Early stopping tracks **validation average precision**, with minimum improvement
0.0001 and patience 8, up to 60 epochs. An improved checkpoint is saved immediately.
After training, the best saved weights are reloaded; the last epoch is not implicitly
treated as the best. The checkpoint contains architecture, state dictionary, seed,
configuration, positive weight, selected epoch and validation score. Loading uses
`weights_only=True`, maps to the requested device and restores evaluation mode.
Tests compare predictions before/after loading and repeat training with the same seed.

We use direct tensor minibatches rather than worker processes because the data fits
comfortably in memory. CPU with two threads avoids overhead for these small matrices
and simplifies repeatability. MPS is an explicit option when available, but the
completed CPU study does not establish MPS speed or bitwise equivalence.

## 5. The controlled question

Does weighting the minority positive class improve MLP **ranking**?

The unweighted loss for a logit `z` is binary cross-entropy. The weighted condition
multiplies its positive-label term by `n_negative / n_positive`, computed from the
training labels only. It does not duplicate data. Both conditions use the same seed,
architecture, regularization, optimizer and stopping policy. Thus the explicit
intervention is positive-class weight. Different selected epochs are an outcome of
the shared stopping rule, not a second manually tuned variable.

Weighting changes gradient contributions and typically changes the score scale.
It need not improve AP, and weighted outputs are not automatically calibrated
subscription probabilities. Threshold optimization is separate from ranking
evaluation. To compare loss curves fairly, the plots use unweighted BCE for both
models in evaluation mode; the actual weighted minibatch objective is also saved.

The seed-42 validation winner determines the weighting. Seeds 43 and 44 repeat that
configuration, each with validation-selected checkpoints and thresholds. Seed 42
remains the primary model; we never choose the best test seed. The sample standard
deviation across three runs measures optimization randomness on the same rows, not
uncertainty about performance on new customers or future time periods.

## 6. Metrics and threshold

Here a positive is a subscriber. Precision = TP/(TP+FP); recall = TP/(TP+FN);
F1 = 2·precision·recall/(precision+recall). Undefined precision/F1 are reported as
zero. The confusion matrix has rows = actual [no, yes], columns = predicted [no, yes],
so its flattened order is TN, FP, FN, TP.

ROC-AUC is the probability that a randomly selected positive receives a higher
score than a randomly selected negative, with ties receiving half credit. A constant
score has ROC-AUC 0.5. ROC-AUC can look favorable when many negatives make the
false-positive rate small, even when precision is operationally poor.

Average precision (AP) is the sum of precision at each recall increase, weighted by
the size of that increase: `sum((R_n - R_(n-1)) * P_n)`. We use scikit-learn's
`average_precision_score`. It is a stepwise, non-interpolated summary, **not** the
trapezoidal area under the precision–recall curve. Linear interpolation between PR
points can give a different result. A dedicated test demonstrates the distinction.
Constant-score AP equals the evaluated set's positive prevalence. AP is
prevalence-dependent, so comparing AP across validation and test is not a pure
measure of model degradation when their class rates differ.

Threshold selection enumerates validation score cutoffs and maximizes validation F1.
Among equal maxima it selects the higher cutoff. The rule uses `score >= threshold`
and avoids the synthetic terminal PR point that has no threshold. Each final model's
threshold is persisted before any test predictions. For the constant dummy, predicting
everyone positive maximizes F1; the resulting confusion matrix is expected.

This teaching objective assumes no particular commercial cost ratio. A deployment
would need contact costs, capacity, value estimates and calibration; those would be
chosen using appropriate validation data and business constraints, not test tuning.

## 7. Evidence trail and limitations

`selection.json` freezes candidates, validation metrics, selected weighting,
thresholds and checkpoint hashes before test transformation or prediction. The
runtime record hashes that manifest. Held-out predictions and labels allow `verify`
to recompute every saved metric. Report generation runs this verification first.
The source code enforces the order; hashes detect later changes but are not an
external timestamp or proof against a malicious operator.

The run reads the dataset and validates split integrity before training, including
checking that each partition has both classes. The data audit summarizes test
prevalence and schema. This descriptive audit is not model selection: test outcomes
are not passed to training, early stopping, weighting selection or threshold tuning.

Even clean preprocessing cannot repair missing client IDs, absent exact dates,
unknown target-maturation timing, retrospective contact variables, sampled customers,
or unmeasured confounding. One fixed temporal split is a limited experiment.
If the classical model wins, the appropriate conclusion is that it worked better
under this protocol and small search budget—not that neural networks never work
on tabular data. Wider claims would require repeated temporal evaluations.

## Interview practice

1. **Why exclude duration?** It is learned during/after the call, so it is unavailable
   at the prediction time. High accuracy with it would answer a different question.
2. **Why exclude campaign and macro indicators too?** Campaign totals may include
   later attempts; macro release/vintage timing cannot be verified. Explain that
   exclusion trades some predictive signal for a more conservative availability policy.
3. **How would preprocessing leakage occur?** Fitting a scaler or vocabulary before
   splitting lets held-out data affect the representation. Point to the test that
   checks training statistics remain unchanged after transforming held-out data.
4. **Why AP rather than accuracy?** A majority-only classifier can achieve high
   accuracy under imbalance. AP evaluates positive-class ranking over thresholds,
   while precision/recall/F1 characterize the chosen operating point.
5. **Why can weighted training reduce BCE comparability?** Weighted BCE has a different
   objective scale. The plots separately compute common unweighted BCE in eval mode.
6. **Does class weighting necessarily increase useful recall?** It can shift scores;
   with thresholds independently optimized, ranking may improve or worsen. Refer to
   the actual validation matrix rather than assuming improvement.
7. **Why save the best checkpoint?** The final epoch can overfit or have lower AP.
   Explain validation-only stopping, checkpoint metadata, and the round-trip test.
8. **What do three seeds establish?** Limited sensitivity to initialization and
   minibatch order on this split. They do not establish a confidence interval.
9. **What is wrong with a random split here?** It mixes periods and possibly clients,
   making it a different generalization question. It is not universally invalid,
   but does not represent the desired future-period deployment setting.
10. **What would you do next?** Obtain customer IDs, exact timestamps and outcome
    maturation dates; use rolling-origin, grouped, gapped evaluation; then test
    calibration and cost/capacity-based policies on validation data.
11. **Could the model decide whom to call?** Predicting subscription is not predicting
    uplift caused by calling. An intervention study or appropriate causal design
    would be needed to answer that question.
12. **What did AI assistance cover?** Repository implementation, experiment execution,
    checks and draft explanations. Describe only the parts you have personally
    reviewed, rerun and can explain; do not imply unaided authorship.
