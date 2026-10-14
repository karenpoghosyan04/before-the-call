# Application notes

## Two factual CV bullets

- Completed an AI-assisted tabular ML study using 41,188 UCI Bank Marketing records,
  comparing dummy, logistic regression, gradient boosting and PyTorch MLP models
  with chronological splits, training-only preprocessing and validation-selected thresholds.
- Evaluated class-weighted MLP training across three seeds, reporting test average
  precision of 0.4716 ± 0.0267; packaged reproducible runs, saved checkpoints,
  13 passing tests and an analysis of temporal drift and feature-leakage risks.

Use these after personally reviewing and rerunning the project. They describe the
completed artifact and explicitly acknowledge assistance; they do not claim unaided
implementation, production deployment, business impact, or a general improvement
over classical models. Replace or remove claims you cannot explain comfortably.

## A brief interview narrative

“I wanted a project that complements retrieval work with model training and evaluation.
This AI-assisted experiment compares classical models with a small PyTorch network.
The main engineering decision was to define the prediction time first: call duration
cannot be known beforehand, and I used a conservative policy for campaign totals and
economic indicators too. I used a chronological split, which exposed a strong change
in subscription rates and prior-contact history.

“Positive-class weighting improved validation AP for the controlled seed. The primary
MLP also had higher test AP than logistic regression, but two other seeds were worse
than logistic regression. So I would emphasize reproducibility and understanding
the limitations, rather than claiming a reliable model improvement. The next study
would strengthen temporal validation and test a less fragile history representation.”

## Study before presenting

1. Run the smoke experiment under a new name and locate its checkpoint and thresholds.
2. Trace one raw row through the feature allowlist, sentinel handling, preprocessing
   and MLP. Explain why logits enter BCE and probabilities enter AP.
3. Recompute precision and recall from the primary MLP's confusion matrix:
   TN=3,647, FP=1,338, FN=1,091, TP=1,258.
4. Explain why the prior dummy can have F1=0.4852 and still be a poor calling policy.
5. Inspect the first-epoch checkpoint and the falling validation-AP curve. Discuss
   distribution shift and sparse-history scaling as hypotheses, not proved causes.
6. Explain why the three-seed AP standard deviation is not a confidence interval.
7. Be explicit about assistance and describe your own review, experiments and learning.

The [technical walkthrough](technical-walkthrough.md#interview-practice) includes
twelve questions with explanations covering leakage, loss functions, seeds, metrics,
chronology, causal interpretation and next steps.
