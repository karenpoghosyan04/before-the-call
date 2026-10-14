"""Generate figures and a factual report, verifying saved metrics against predictions."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from .data import digest
from .evaluate import metrics

DISPLAY = {"dummy": "Dummy prior", "logistic": "Logistic regression", "hist_gradient_boosting": "Gradient boosting"}


def label(name):
    return DISPLAY.get(name, name.replace("mlp_", "MLP ").replace("_", " "))


def verify(run_dir):
    run_dir = Path(run_dir)
    saved = json.loads((run_dir / "metrics.json").read_text())
    selection = json.loads((run_dir / "selection.json").read_text())
    runtime = json.loads((run_dir / "runtime.json").read_text())
    if digest((run_dir / "selection.json").read_bytes()) != runtime["selection_sha256"]:
        raise AssertionError("Selection manifest changed after evaluation")
    if digest((run_dir / "config.json").read_bytes()) != selection["config_sha256"]:
        raise AssertionError("Configuration changed")
    for split in ["validation", "test"]:
        with np.load(run_dir / f"{split}_predictions.npz") as arrays:
            if set(arrays.files) - {"y"} != set(saved[split]):
                raise AssertionError("Prediction/metric model mismatch")
            for model, expected in saved[split].items():
                if split == "test" and expected["threshold"] != selection["thresholds"][model]:
                    raise AssertionError("Test threshold differs from locked validation choice")
                actual = metrics(arrays["y"], arrays[model], expected["threshold"])
                for key, value in actual.items():
                    np.testing.assert_allclose(value, expected[key], rtol=1e-10, atol=1e-12)
    return saved


def generate(run_dir):
    run_dir = Path(run_dir)
    saved = verify(run_dir)
    selection = json.loads((run_dir / "selection.json").read_text())
    histories = json.loads((run_dir / "histories.json").read_text())
    runtime = json.loads((run_dir / "runtime.json").read_text())
    env = json.loads((run_dir / "environment.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    audit = json.loads((run_dir.parents[1] / "data/audit.json").read_text())
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.dpi": 170})
    test = saved["test"]
    primary = ["dummy", "logistic", "hist_gradient_boosting", selection["primary_mlp"]]
    with np.load(run_dir / "test_predictions.npz") as arrays:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for model in primary[1:]:
            p, r, _ = precision_recall_curve(arrays["y"], arrays[model])
            ax.step(r, p, where="post", linewidth=1.5,
                    label=f"{label(model)} · AP {test[model]['average_precision']:.3f}")
        ax.axhline(arrays["y"].mean(), color="#6b7280", linestyle="--", label="Test prevalence (constant-score AP)")
        ax.set(xlabel="Recall", ylabel="Precision", title="Final chronological holdout · precision–recall", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.15)
        fig.savefig(plots / "precision_recall.png")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for weight in config["class_weight_options"]:
        key = f"mlp_{weight}_seed{config['seed']}"
        history = histories[key]
        epochs = [x["epoch"] for x in history["history"]]
        line = axes[0].plot(epochs, [x["validation_bce"] for x in history["history"]], label=f"{weight}: validation")[0]
        axes[0].plot(epochs, [x["train_bce"] for x in history["history"]], "--", color=line.get_color(), label=f"{weight}: train")
        axes[1].plot(epochs, [x["validation_ap"] for x in history["history"]], label=f"{weight} (best epoch {history['best_epoch']})")
        best = history["history"][history["best_epoch"] - 1]
        axes[1].scatter([history["best_epoch"]], [best["validation_ap"]], s=28)
    axes[0].set(title="Comparable unweighted BCE · eval mode", xlabel="Epoch", ylabel="Binary cross-entropy")
    axes[1].set(title="Early stopping uses validation AP", xlabel="Epoch", ylabel="Average precision")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    fig.savefig(plots / "training_curves.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    for ax, model in zip(axes.flat, primary):
        cm = np.array(test[model]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
               xlabel="Predicted", ylabel="Actual", title=label(model))
    fig.savefig(plots / "confusion_matrices.png")
    plt.close(fig)
    mlps = [v for k, v in test.items() if k.startswith("mlp")]
    ap = np.array([v["average_precision"] for v in mlps])
    winner = selection["validation_winner"]
    best_classical = max(primary[1:3], key=lambda k: test[k]["average_precision"])
    comparison = "exceeds" if test[best_classical]["average_precision"] > test[selection["primary_mlp"]]["average_precision"] else "does not exceed"
    lines = ["# Results from the executed experiment", "",
             f"Run: `{config['name']}`. Completed: {runtime['completed_at_utc']}. "
             f"Runtime: {runtime['total_seconds']:.2f} seconds ({runtime['device']}, {runtime['threads']} threads). "
             f"Python {env['python']}; PyTorch {env['packages']['torch']}; scikit-learn {env['packages']['scikit-learn']}.", "",
             "This report is generated from saved predictions and checks every metric before rendering. "
             "Smoke runs verify execution only; their scores are not research findings.", "",
             "## Data and drift", "",
             f"Original observations: {audit['rows']:,}. Exact duplicate rows: {audit['exact_duplicate_rows']}. "
             f"Later identical predictor patterns excluded: {audit['duplicate_predictor_patterns']:,}; "
             f"patterns with conflicting outcomes: {audit['conflicting_label_patterns']:,}. "
             "Duplicates are defined on the retained original predictors, without consulting outcomes. "
             "Exclusions can remove distinct customers with identical profiles.", "",
             "| Full-data partition | Retained n | Positive prevalence | Observed pdays (not sentinel) |",
             "|---|---:|---:|---:|"]
    for split, summary in audit["split_summary"].items():
        observed = audit["feature_drift"][split]["numeric"]["pdays"]["observed_count"]
        lines.append(f"| {split} | {summary['n']:,} | {summary['prevalence']:.2%} | {observed:,} |")
    lines += ["", "These are full-data audit counts; smoke runs subsample them. "
              "Previous-contact distance has very few observed training values. Median imputation makes its training variance small, "
              "so standardization can amplify later values outside that range. This is a concrete extrapolation risk and may contribute "
              "to unstable validation curves; this diagnostic does not establish causality. "
              f"Selected MLP checkpoint epochs in this run: {', '.join(str(h['best_epoch']) for h in histories.values())}. "
              "Future validation studies could compare robust scaling or a different history representation. "
              "No preprocessing changes were made in response to test scores.", "",
             "## Validation experiment", "",
             "Question: does positive-class weighting improve MLP ranking (average precision)? "
             "Only the BCE positive weight changes; architecture, regularization, data, seed and stopping rule are fixed.", "",
             "| Candidate | Validation AP | ROC-AUC | Best epoch | Epochs run | Fit + validation seconds |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, hist in histories.items():
        m = saved["validation"][name]
        lines.append(f"| {label(name)} | {m['average_precision']:.4f} | {m['roc_auc']:.4f} | {hist['best_epoch']} | {hist['epochs_run']} | {hist['seconds']:.2f} |")
    lines += ["", f"Selected weighting: **{selection['selected_weight']}**. Model family chosen by validation AP: **{label(winner)}**. "
              "The primary MLP is the predeclared selection seed, not the best test seed. "
              "Additional seeds repeat the selected configuration; each has its own validation-selected checkpoint and threshold.", "",
              "Fixed-baseline validation AP: " + "; ".join(
                  f"{label(k)} {saved['validation'][k]['average_precision']:.4f}" for k in primary[:3]) + ".", "",
              "## Final test metrics", "",
              "Thresholds maximize validation F1; ties choose the higher threshold. Predictions use score ≥ threshold. "
              "They are not retuned on the final holdout. F1 is a teaching objective, not a business cost model.", "",
              "| Model | AP | ROC-AUC | Precision | Recall | F1 | Threshold | TN / FP / FN / TP |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name, m in test.items():
        cm = np.array(m["confusion_matrix"]).flatten()
        lines.append(f"| {label(name)} | {m['average_precision']:.4f} | {m['roc_auc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['threshold']:.6f} | {' / '.join(map(str, cm))} |")
    lines += ["", f"Test size: {test['dummy']['n']:,}; positive prevalence: {test['dummy']['prevalence']:.2%}. "
              "The constant-score dummy predicts everybody positive under validation-F1 selection; its recall is 1 and precision is prevalence. "
              "This is expected, not a useful calling policy.", "",
              f"Across {len(mlps)} MLP seeds: test AP mean {ap.mean():.4f}, "
              f"sample standard deviation {ap.std(ddof=1) if len(ap)>1 else 0:.4f}, range {ap.min():.4f}–{ap.max():.4f}. "
              "This measures training randomness on one fixed split, not sampling uncertainty or a confidence interval.", "",
              f"The highest test AP among the two classical baselines ({label(best_classical)}) {comparison} the primary MLP. "
              f"It exceeds {sum(test[best_classical]['average_precision'] > m['average_precision'] for m in mlps)} "
              f"of the {len(mlps)} MLP seed results. "
              "This is a descriptive comparison after freezing choices; the validation-selected model is unchanged. "
              "A neural network is not automatically better on tabular data: inductive bias, sample size and temporal drift matter.", "",
              "![Precision–recall curves](plots/precision_recall.png)", "",
              "![Training curves](plots/training_curves.png)", "",
              "Loss plots use unweighted BCE in evaluation mode for both configurations, allowing like-for-like comparison. "
              "Weighted optimization loss is also saved in histories.json; it is on a different scale. "
              "The best AP checkpoint can precede the minimum validation BCE.", "",
              "![Confusion matrices](plots/confusion_matrices.png)", "", "## Error patterns", ""]
    for name in [winner] if winner == selection["primary_mlp"] else [winner, selection["primary_mlp"]]:
        m = test[name]
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        lines.append(f"- {label(name)}: {fp:,} false positives (unnecessary calls in this framing), {fn:,} false negatives "
                     f"(missed subscribers); {fp/(fp+tn):.1%} false-positive rate and {fn/(fn+tp):.1%} false-negative rate.")
    lines += ["", "Descriptive subgroup checks (small groups are noisy; these are not a fairness audit):", "",
              "| Model | Group | n | Prevalence | Precision | Recall |", "|---|---|---:|---:|---:|---:|"]
    for g in json.loads((run_dir / "subgroups.json").read_text()):
        lines.append(f"| {label(g['model'])} | {g['column']}={g['value']} | {g['n']} | {g['prevalence']:.3f} | {g['precision']:.3f} | {g['recall']:.3f} |")
    lines += ["", "## Interpretation limits", "",
              "One chronological partition cannot establish a general ranking. Row order approximates time; exact dates and client IDs are absent. "
              "Temporal prevalence shifts, unknown categories, repeat customers, and the duplicate-pattern exclusion policy affect evaluation. "
              "The dataset contains contacted customers and retrospective outcomes, so it cannot establish the causal value of a call. "
              "No random-split experiment, calibration study, fairness audit or production validation was executed. "
              "The weighting search uses one seed; repeated seeds for both candidates and rolling-origin validation would strengthen the research.", ""]
    (run_dir / "report.md").write_text("\n".join(lines))
    return run_dir / "report.md"
