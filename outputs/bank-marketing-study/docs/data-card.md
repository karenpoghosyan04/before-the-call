# Data card and prediction-time assumptions

## Official provenance and variant

The source is [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing),
DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306), attributed to Sérgio Moro,
Paulo Rita and Paulo Cortez (2014). The UCI license is
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the dataset when
reusing data or derived results. The code has a separate MIT license.

The official ZIP contains two nested ZIPs. We inspected the included names file,
not just the landing page: the latter combines details from different versions.

| Variant | Rows | Predictor columns | Ordering |
|---|---:|---:|---|
| bank-additional-full.csv (chosen) | 41,188 | 20 | Date order, May 2008–November 2010 |
| bank-additional.csv | 4,119 | 20 | Random approximately 10% sample |
| bank-full.csv | 45,211 | 16 | Date order, older feature set |
| bank.csv | 4,521 | 16 | Random approximately 10% sample |

The page describes the old version as having “17 inputs”; its file has 17 columns
**including the target**, hence 16 predictors. The downloaded headers and names
files are the authority for schema. The chosen version has 21 columns including `y`.
It retains the full ordered sample and richer prior-campaign history, enabling a
temporal evaluation at laptop scale. We do not mix variants.

`data/provenance.json` records the source, exact nested member, retrieval/verification
time, and SHA-256 checksums. `bank-study prepare` checks both the downloaded archive
and CSV; it fails if either changes. Only the specific CSV and names file are extracted.
Raw data is excluded from Git. Derived split indices, audit summaries and predictions
are included for reproducibility and remain subject to dataset attribution.

## Feature dictionary and policy

Definitions below paraphrase `bank-additional-names.txt` in the official archive.

| Feature | Meaning | Policy |
|---|---|---|
| age | Client age | Numeric; retain |
| job | Occupation category | Retain |
| marital | Marital category; divorced also includes widowed | Retain |
| education | Education category | Retain |
| default | Credit in default | Retain |
| housing | Housing loan status | Retain |
| loan | Personal loan status | Retain |
| contact | Contact channel: telephone or cellular | Assume planned channel is known |
| month | Month of last contact | Assume the planned call month is known |
| day_of_week | Weekday of last contact | Assume the planned weekday is known |
| duration | Last call length in seconds | Exclude: directly unavailable before the call |
| campaign | Number of contacts in current campaign, including last contact | Exclude: cumulative history can encode later contacts; no attempt timestamps |
| pdays | Days since prior campaign contact; 999 means no previous contact | Replace sentinel with missing; add indicator |
| previous | Number of contacts before the current campaign | Retain, assuming historical snapshot |
| poutcome | Previous campaign outcome | Retain, assuming historical snapshot |
| emp.var.rate | Quarterly employment variation | Exclude; publication/vintage timing not verified |
| cons.price.idx | Monthly consumer price index | Exclude; publication/vintage timing not verified |
| cons.conf.idx | Monthly consumer confidence index | Exclude; publication/vintage timing not verified |
| euribor3m | Daily three-month Euribor rate | Exclude with the economic feature group |
| nr.employed | Quarterly number of employees | Exclude; publication/vintage timing not verified |
| y | Whether the client subscribed to a term deposit | Binary target; never an input |

This leaves 13 original predictors and one engineered indicator. For `pdays`, the
training median imputes the missing sentinel; `previously_contacted` preserves its
meaning. If all training values are missing, the imputer keeps the column and uses
zero. Other numeric values are median-imputed then standardized. Categorical
`unknown` is retained as a meaningful missingness level; unseen held-out categories
produce an all-zero one-hot block. All learned transformations fit only the training
partition. The numeric scaling is also used for the tree baseline for a common input
representation; tree splits do not require it.

The names file explicitly describes categorical missingness as `unknown`, despite
the landing page's general “no missing values” summary. `data/audit.json` reports
actual nulls and unknown counts, observed category levels, duplicates and class rates.

## Split and duplicate decisions

The first 60% of original row positions are training, the next 20% validation,
and the final 20% test. Boundaries are fixed before duplicate exclusions.
There is no random shuffle or target-based stratification. Original zero-based row
positions are saved in `data/splits.json`; they are never features.

Later occurrences of identical **retained raw predictor patterns** are excluded,
keeping the first occurrence, regardless of `y`. This also removes exact duplicate
records and prevents a repeated feature vector from appearing in multiple partitions.
Conflicting-label patterns are counted. Identical vectors can represent different
people, so this conservative exclusion changes the population and is not a claim
that all excluded records are erroneous. It does not solve repeated-client leakage:
no client identifiers are available. The audit reports counts before and after removal.

Date order is documented by UCI, but exact dates are absent. Contiguous row splitting
is an approximation to future prediction: a boundary can cut through a single day,
and we cannot enforce a label-availability gap or group by customer. Campaign outcomes
may mature after a split boundary. Planned contact fields and prior history require
availability assumptions that the anonymized data cannot fully verify.

A random split would mix economic periods and potentially repeated customers across
partitions. It usually assesses same-population interpolation rather than future-period
performance; it could overstate usefulness in a changing campaign. Conversely, our
single temporal split exposes drift and can be unusually difficult or unrepresentative.
We do not claim a measured random-versus-temporal performance gap: no random-split
experiment was run. Rolling-origin evaluation with outcome-availability gaps and
client grouping is the appropriate extension if richer metadata becomes available.

## Scope

This is an educational retrospective prediction study of contacted customers at one
Portuguese institution. It is not a financial decision system, causal uplift model,
credit-risk model, or recommendation to use personal attributes in real campaigns.
Age and other demographic proxies can introduce fairness concerns. There is no
fairness, privacy, calibration or deployment validation. Historical economic context,
selection into the contacted sample, campaign policy and shifts in subscription rates
limit generalization. Predictions describe recorded outcomes, not necessarily whether
the imminent call causes subscription.
