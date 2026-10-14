"""Check two completed runs, including every prediction and locally saved checkpoint."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from bankstudy.data import features, read_data, write_json
from bankstudy.model import load_checkpoint, predict
from bankstudy.report import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("reproduction", type=Path)
    args = parser.parse_args()
    for path in (args.reference, args.reproduction):
        verify(path)
    arrays_checked = 0
    for split in ("validation", "test"):
        with np.load(args.reference / f"{split}_predictions.npz") as a, np.load(args.reproduction / f"{split}_predictions.npz") as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
                arrays_checked += 1
    root = args.reference.resolve().parents[1]
    config = json.loads((args.reference / "config.json").read_text())
    artifacts = root / "artifacts" / config["name"]
    preprocessor = joblib.load(artifacts / "preprocessor.joblib")
    frame = read_data(root)
    indices = json.loads((args.reference / "run_indices.json").read_text())
    checkpoint_count = 0
    for split in ("validation", "test"):
        x = preprocessor.transform(features(frame.iloc[indices[split]])).astype(np.float32)
        with np.load(args.reference / f"{split}_predictions.npz") as saved:
            for key in saved.files:
                if key.startswith("mlp_"):
                    model, _ = load_checkpoint(artifacts / f"{key}.pt")
                    # BLAS thread choices can change last bits; predictions must agree numerically.
                    np.testing.assert_allclose(predict(model, x), saved[key], rtol=1e-6, atol=1e-7)
                    checkpoint_count += 1
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "reference": args.reference.name, "reproduction": args.reproduction.name,
              "prediction_arrays_bitwise_equal": arrays_checked,
              "checkpoint_prediction_checks_passed": checkpoint_count,
              "scope": "Same-machine CPU reproducibility; not a cross-platform guarantee"}
    write_json(root / "results/reproduction-check.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
