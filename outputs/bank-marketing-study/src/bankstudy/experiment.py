"""One controlled validation experiment followed by one final test evaluation."""

import importlib.metadata
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .data import CSV_HASH, digest, features, make_preprocessor, read_data, validate_splits, write_json
from .evaluate import choose_threshold, metrics
from .model import predict, resolve_device, train_mlp


def environment():
    packages = ["torch", "scikit-learn", "pandas", "numpy", "matplotlib", "joblib", "pytest", "threadpoolctl"]
    hardware = {}
    for key in ["machdep.cpu.brand_string", "hw.memsize", "hw.logicalcpu"]:
        try:
            hardware[key] = subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            hardware[key] = "unavailable in sandbox"
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "logical_cpus": os.cpu_count(), "hardware": hardware,
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "packages": {p: importlib.metadata.version(p) for p in packages}}


def run(root, config_path, run_name=None):
    root = Path(root)
    config = json.loads(Path(config_path).read_text())
    if run_name is not None:
        config["name"] = run_name
    name = config["name"]
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid run name")
    out, artifacts = root / "results" / name, root / "artifacts" / name
    if out.exists():
        raise FileExistsError(f"Run directory already exists: {out}. Use a new config name to preserve evidence.")
    saved_split = json.loads((root / "data/splits.json").read_text())
    if saved_split["csv_sha256"] != CSV_HASH or saved_split["fractions"] != config["split_fractions"]:
        raise ValueError("Prepared data and experiment split config disagree")
    if config["seed"] not in config["repeat_seeds"] or len(set(config["repeat_seeds"])) != len(config["repeat_seeds"]):
        raise ValueError("Repeat seeds must be unique and include the selection seed")
    if not config["class_weight_options"] or set(config["class_weight_options"]) - {"none", "balanced"}:
        raise ValueError("Unknown weighting configuration")
    started = time.perf_counter()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(config["threads"]))
    device = resolve_device(config["device"])
    torch.set_num_threads(config["threads"])
    frame = read_data(root)
    ids = saved_split["indices"]
    validate_splits(frame, ids)
    if config["max_rows_per_split"]:
        rng = np.random.default_rng(config["seed"])
        ids = {k: sorted(rng.choice(v, min(len(v), config["max_rows_per_split"]), replace=False).tolist())
               for k, v in ids.items()}
        validate_splits(frame, ids)
    out.mkdir(parents=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    write_json(out / "environment.json", environment())
    write_json(out / "run_indices.json", ids)
    # No test feature transformation or prediction until selection.json is persisted.
    xtr_raw, xv_raw = [features(frame.iloc[ids[k]]) for k in ("train", "validation")]
    ytr, yv = [(frame.iloc[ids[k]].y == "yes").to_numpy(dtype=np.int64) for k in ("train", "validation")]
    preprocessor = make_preprocessor(xtr_raw)
    xtr = preprocessor.fit_transform(xtr_raw).astype(np.float32)
    xv = preprocessor.transform(xv_raw).astype(np.float32)
    joblib.dump(preprocessor, artifacts / "preprocessor.joblib")
    write_json(out / "feature_names.json", preprocessor.get_feature_names_out().tolist())
    models, val_metrics, histories, runtimes, val_predictions = {}, {}, {}, {}, {}
    with threadpool_limits(limits=config["threads"]):
        baselines = {
            "dummy": DummyClassifier(strategy="prior"),
            "logistic": LogisticRegression(C=config["logistic_C"], max_iter=2000, solver="lbfgs", random_state=config["seed"]),
            "hist_gradient_boosting": HistGradientBoostingClassifier(**config["tree"], early_stopping=False,
                                                                     random_state=config["seed"]),
        }
        for label, estimator in baselines.items():
            tick = time.perf_counter()
            estimator.fit(xtr, ytr)
            probabilities = estimator.predict_proba(xv)[:, 1]
            threshold = choose_threshold(yv, probabilities)
            val_metrics[label] = metrics(yv, probabilities, threshold)
            runtimes[label] = time.perf_counter() - tick
            models[label], val_predictions[label] = estimator, probabilities
            joblib.dump(estimator, artifacts / f"{label}.joblib")
            print(f"{label}: validation AP={val_metrics[label]['average_precision']:.4f}", flush=True)
        candidates = {}
        for weight in config["class_weight_options"]:
            label = f"mlp_{weight}_seed{config['seed']}"
            model, history = train_mlp(xtr, ytr, xv, yv, config["mlp"], weight, config["seed"],
                                       artifacts / f"{label}.pt", device)
            probabilities = predict(model, xv, device)
            val_metrics[label] = metrics(yv, probabilities, choose_threshold(yv, probabilities))
            histories[label], runtimes[label] = history, history["seconds"]
            models[label], val_predictions[label] = model, probabilities
            candidates[weight] = label
            print(f"{label}: validation AP={val_metrics[label]['average_precision']:.4f}; {history['epochs_run']} epochs", flush=True)
        # Ties follow the predeclared config order, never test results.
        selected_weight = max(candidates, key=lambda w: val_metrics[candidates[w]]["average_precision"])
        selected_label = candidates[selected_weight]
        final_labels = list(baselines) + [selected_label]
        for seed in config["repeat_seeds"]:
            if seed == config["seed"]:
                continue
            label = f"mlp_{selected_weight}_seed{seed}"
            model, history = train_mlp(xtr, ytr, xv, yv, config["mlp"], selected_weight, seed,
                                       artifacts / f"{label}.pt", device)
            probabilities = predict(model, xv, device)
            val_metrics[label] = metrics(yv, probabilities, choose_threshold(yv, probabilities))
            histories[label], runtimes[label] = history, history["seconds"]
            models[label], val_predictions[label] = model, probabilities
            final_labels.append(label)
            print(f"{label}: validation AP={val_metrics[label]['average_precision']:.4f}; {history['epochs_run']} epochs", flush=True)
        primary = list(baselines) + [selected_label]
        selection = {
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "criterion": "maximum validation average precision; config-order tie break",
            "threshold_rule": "maximum validation F1; highest threshold breaks ties",
            "selected_weight": selected_weight, "primary_mlp": selected_label,
            "validation_winner": max(primary, key=lambda m: val_metrics[m]["average_precision"]),
            "final_labels": final_labels, "thresholds": {k: val_metrics[k]["threshold"] for k in final_labels},
            "validation_metrics": val_metrics,
            "config_sha256": digest((out / "config.json").read_bytes()),
            "splits_sha256": digest((root / "data/splits.json").read_bytes()),
            "checkpoint_sha256": {p.name: digest(p.read_bytes()) for p in artifacts.glob("*.pt")},
            "test_used_for_selection": False,
        }
        write_json(out / "selection.json", selection)
        write_json(out / "histories.json", histories)
        np.savez_compressed(out / "validation_predictions.npz", y=yv, **val_predictions)
        # FINAL EVALUATION: do not use the following values to change model choices.
        xt = preprocessor.transform(features(frame.iloc[ids["test"]])).astype(np.float32)
        yt = (frame.iloc[ids["test"]].y == "yes").to_numpy(dtype=np.int64)
        test_metrics, test_predictions, groups = {}, {}, []
        for label in final_labels:
            estimator = models[label]
            probabilities = (predict(estimator, xt, device) if label.startswith("mlp")
                             else estimator.predict_proba(xt)[:, 1])
            threshold = selection["thresholds"][label]
            test_predictions[label] = probabilities
            test_metrics[label] = metrics(yt, probabilities, threshold)
            # Descriptive diagnostics only; never used to modify the selected model.
            if label in [selected_label, selection["validation_winner"]]:
                for column in ["contact", "poutcome"]:
                    values = frame.iloc[ids["test"]][column].to_numpy()
                    for value in sorted(set(values)):
                        mask = values == value
                        if len(set(yt[mask])) == 2:
                            groups.append({"model": label, "column": column, "value": value,
                                           **metrics(yt[mask], probabilities[mask], threshold)})
        np.savez_compressed(out / "test_predictions.npz", y=yt, **test_predictions)
        write_json(out / "subgroups.json", groups)
        write_json(out / "metrics.json", {"test": test_metrics, "validation": val_metrics})
    write_json(out / "runtime.json", {"total_seconds": time.perf_counter() - started,
                                      "fit_and_validation_seconds": runtimes, "device": str(device),
                                      "threads": config["threads"],
                                      "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                      "selection_sha256": digest((out / "selection.json").read_bytes())})
    return out
