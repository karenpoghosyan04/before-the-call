import numpy as np
import pandas as pd
import pytest

from bankstudy.data import RAW_FEATURES, features, make_preprocessor, temporal_splits, validate_splits


def example_frame(n=100):
    df = pd.DataFrame({c: ["unknown"] * n for c in RAW_FEATURES})
    df["age"] = np.arange(n) + 18
    df["pdays"] = 999
    df["previous"] = 0
    df["y"] = ["yes" if i % 2 else "no" for i in range(n)]
    df["duration"] = np.arange(n) * 100
    return df


def test_preprocessing_never_fits_validation_categories_or_statistics():
    train = pd.DataFrame({"age": [20., 40., np.nan], "job": ["a", "b", "a"]})
    heldout = pd.DataFrame({"age": [1e9], "job": ["never_seen"]})
    pre = make_preprocessor(train)
    pre.fit(train)
    scaler = pre.named_transformers_["num"].named_steps["standardscaler"]
    imputer = pre.named_transformers_["num"].named_steps["simpleimputer"]
    encoder = pre.named_transformers_["cat"].named_steps["onehotencoder"]
    np.testing.assert_allclose(scaler.mean_, [30.])
    np.testing.assert_allclose(imputer.statistics_, [30.])
    result = pre.transform(heldout)
    assert result.shape == (1, 3)
    assert result[0, 1:].sum() == 0
    assert "never_seen" not in encoder.categories_[0]
    np.testing.assert_allclose(scaler.mean_, [30.])


def test_duration_target_and_row_identifiers_cannot_enter_features():
    df = example_frame()
    df["row_id"] = range(len(df))
    a = features(df)
    df["duration"] = -1000
    df["y"] = "yes"
    pd.testing.assert_frame_equal(a, features(df))
    assert not {"duration", "y", "campaign", "row_id"} & set(a.columns)
    assert a.pdays.isna().all()
    assert (a.previously_contacted == 0).all()


def test_split_order_integrity_and_label_blind_duplicate_policy():
    df = example_frame()
    df.loc[80, RAW_FEATURES] = df.loc[0, RAW_FEATURES]
    df.loc[80, "y"] = "yes"  # contradictory target must not influence duplicate decision
    splits = temporal_splits(df)
    validate_splits(df, splits)
    assert 0 in splits["train"] and 80 not in splits["test"]
    assert max(splits["train"]) < min(splits["validation"])
    assert max(splits["validation"]) < min(splits["test"])
    before = splits
    df["y"] = "no"
    assert temporal_splits(df) == before


def test_split_validator_rejects_overlap_and_duplicate_patterns():
    df = example_frame()
    splits = temporal_splits(df)
    splits["test"].append(0)
    with pytest.raises(ValueError, match="Overlapping"):
        validate_splits(df, splits)
    splits = temporal_splits(df)
    df.loc[80, RAW_FEATURES] = df.loc[0, RAW_FEATURES]
    with pytest.raises(ValueError, match="Duplicate predictor"):
        validate_splits(df, splits)


def test_invalid_split_fractions_are_rejected():
    with pytest.raises(ValueError):
        temporal_splits(example_frame(), [0.8, 0.2, 0.2])


def test_all_missing_training_numeric_column_is_retained():
    train = pd.DataFrame({"pdays": [np.nan, np.nan], "job": ["a", "a"]})
    pre = make_preprocessor(train).fit(train)
    result = pre.transform(pd.DataFrame({"pdays": [7.], "job": ["new"]}))
    assert result.shape == (1, 2) and np.isfinite(result).all()
