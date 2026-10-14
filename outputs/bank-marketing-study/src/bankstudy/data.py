"""Verified data acquisition, predictor policy, and persistent temporal splits."""

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SOURCE = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
CSV_HASH = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
ARCHIVE_HASH = "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"
VARIANT = "bank-additional-full.csv"
DROP = ["duration", "campaign", "emp.var.rate", "cons.price.idx",
        "cons.conf.idx", "euribor3m", "nr.employed"]
RAW_FEATURES = ["age", "job", "marital", "education", "default", "housing",
                "loan", "contact", "month", "day_of_week", "pdays", "previous", "poutcome"]


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def acquire(root):
    raw = Path(root) / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "bank-marketing.zip"
    if not archive.exists():
        with urllib.request.urlopen(SOURCE, timeout=90) as response:
            blob = response.read()
        if digest(blob) != ARCHIVE_HASH:
            raise ValueError("Source archive changed; review provenance before proceeding")
        archive.write_bytes(blob)
    blob = archive.read_bytes()
    if digest(blob) != ARCHIVE_HASH:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(blob)) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip"))) as inner:
            csv = inner.read("bank-additional/" + VARIANT)
            names = inner.read("bank-additional/bank-additional-names.txt")
    if digest(csv) != CSV_HASH:
        raise ValueError("Dataset checksum mismatch")
    (raw / VARIANT).write_bytes(csv)
    (raw / "bank-additional-names.txt").write_bytes(names)
    provenance = {
        "source_url": SOURCE, "landing_page": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "doi": "10.24432/C5K306", "license": "CC BY 4.0",
        "citation": "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository.",
        "variant": VARIANT, "csv_sha256": CSV_HASH, "archive_sha256": ARCHIVE_HASH,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "nested_member": "bank-additional.zip/bank-additional/" + VARIANT,
        "names_sha256": digest(names),
    }
    write_json(Path(root) / "data/provenance.json", provenance)
    return read_data(root)


def read_data(root):
    path = Path(root) / "data/raw" / VARIANT
    if digest(path.read_bytes()) != CSV_HASH:
        raise ValueError("Dataset checksum mismatch; run prepare")
    frame = pd.read_csv(path, sep=";")
    if frame.shape != (41188, 21) or set(frame.y.unique()) != {"yes", "no"}:
        raise ValueError("Unexpected dataset schema")
    return frame


def features(frame):
    # Explicit allowlist prevents duration, target, row IDs, or new columns sneaking in.
    x = frame[RAW_FEATURES].copy()
    x["previously_contacted"] = (x.pdays != 999).astype(int)
    x["pdays"] = x.pdays.mask(x.pdays == 999, np.nan)
    return x


def make_preprocessor(x_train):
    cat = x_train.select_dtypes(include=["object", "string"]).columns.tolist()
    num = [c for c in x_train if c not in cat]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
    ], remainder="drop", sparse_threshold=0)


def temporal_splits(frame, fractions=(0.6, 0.2, 0.2)):
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError("Need three positive split fractions summing to one")
    n = len(frame)
    a, b = int(n * fractions[0]), int(n * sum(fractions[:2]))
    # Boundaries use ORIGINAL row positions. Purge later identical predictor patterns
    # without consulting the labels. This conservative policy changes the population.
    keep = ~frame[RAW_FEATURES].duplicated(keep="first").to_numpy()
    positions = np.arange(n)
    splits = {name: positions[lo:hi][keep[lo:hi]].tolist()
              for name, lo, hi in [("train", 0, a), ("validation", a, b), ("test", b, n)]}
    return splits


def validate_splits(frame, splits):
    sets = [set(splits[k]) for k in ("train", "validation", "test")]
    if any(not s for s in sets) or any(len(s) != len(splits[k]) for s, k in zip(sets, ("train", "validation", "test"))):
        raise ValueError("Empty split or repeated row index")
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError("Overlapping splits")
    if min(sets[0]) < 0 or max(sets[2]) >= len(frame):
        raise ValueError("Row index out of range")
    if not max(sets[0]) < min(sets[1]) <= max(sets[1]) < min(sets[2]):
        raise ValueError("Temporal order violated")
    combined = [i for k in ("train", "validation", "test") for i in splits[k]]
    if frame.iloc[combined][RAW_FEATURES].duplicated().any():
        raise ValueError("Duplicate predictor patterns remain")
    for k in splits:
        if frame.iloc[splits[k]].y.nunique() != 2:
            raise ValueError(f"Both classes required in {k}")


def prepare(root):
    frame = acquire(root)
    splits = temporal_splits(frame)
    validate_splits(frame, splits)
    # This audit describes the data; test metrics are never used for selection.
    audit = {
        "rows": len(frame), "columns": len(frame.columns),
        "null_counts": frame.isna().sum().to_dict(),
        "unknown_counts": (frame == "unknown").sum().to_dict(),
        "categorical_levels": {c: sorted(frame[c].unique().tolist()) for c in frame.select_dtypes("object")},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_predictor_patterns": int(frame[RAW_FEATURES].duplicated().sum()),
        "conflicting_label_patterns": int((frame.groupby(RAW_FEATURES, dropna=False).y.nunique() > 1).sum()),
        "excluded_features": DROP,
        "split_summary": {k: {"n": len(ids), "positives": int((frame.iloc[ids].y == "yes").sum()),
                              "prevalence": float((frame.iloc[ids].y == "yes").mean()),
                              "first_row": min(ids), "last_row": max(ids)} for k, ids in splits.items()},
        "original_boundaries": [int(len(frame) * 0.6), int(len(frame) * 0.8)],
        "raw_prevalence": float((frame.y == "yes").mean()),
    }
    train_x = features(frame.iloc[splits["train"]])
    numeric = train_x.select_dtypes("number").columns
    categorical = train_x.select_dtypes("object").columns
    audit["feature_drift"] = {}
    for split, indices in splits.items():
        x = features(frame.iloc[indices])
        audit["feature_drift"][split] = {
            "numeric": {c: {"observed_count": int(x[c].notna().sum()),
                            "min": float(x[c].min()), "max": float(x[c].max()),
                            "outside_train_range": int(((x[c] < train_x[c].min()) | (x[c] > train_x[c].max())).sum())}
                        for c in numeric},
            "unseen_categories": {c: sorted(set(x[c]) - set(train_x[c])) for c in categorical},
        }
    write_json(Path(root) / "data/splits.json", {"csv_sha256": CSV_HASH, "fractions": [0.6, 0.2, 0.2], "indices": splits})
    write_json(Path(root) / "data/audit.json", audit)
    return audit
