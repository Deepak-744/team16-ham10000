"""Verify metrics against saved HAM10000 predictions and optionally regenerate plots."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np


RESULTS = Path(__file__).resolve().parent
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
SPLIT_ROWS = {"val": 1008, "test": 1016}
INK, TEAL, GOLD, GRAY = "#203337", "#157872", "#a95827", "#667a7e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": "#adc1bc",
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
    "grid.color": "#e3ece8", "axes.axisbelow": True,
})


def save(fig, path):
    fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def evaluate(scores, labels):
    if scores.shape != (len(labels), len(CLASSES)):
        raise ValueError("Unexpected score array dimensions")
    if not np.all((scores >= 0) & (scores <= 1)) or not np.allclose(scores.sum(axis=1), 1):
        raise ValueError("The rows must contain finite softmax scores")
    if labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() >= len(CLASSES):
        raise ValueError("Unexpected class indices")
    predictions = scores.argmax(axis=1)
    confidence = scores.max(axis=1).astype(float)
    correct = predictions == labels
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    recall = divide(diagonal, support)
    precision = divide(diagonal, predicted)
    f1 = divide(2 * diagonal, support + predicted)
    bins = []
    for left, right in zip(np.arange(10) / 10, np.arange(1, 11) / 10):
        mask = (confidence >= left) & ((confidence < right) if right < 1 else (confidence <= right))
        count = int(mask.sum())
        bins.append({"left": left, "right": right, "n": count, "correct": int(correct[mask].sum()),
                     "accuracy": float(correct[mask].mean()) if count else None,
                     "mean_score": float(confidence[mask].mean()) if count else None})
    ece = sum(b["n"] * abs(b["accuracy"] - b["mean_score"]) for b in bins if b["n"]) / len(labels)

    # Sort by descending score. Advance across ties together so every point
    # corresponds to retaining all examples at or above an actual threshold.
    order = np.argsort(-confidence, kind="stable")
    sorted_scores = confidence[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(order) - 1]
    retained = ends + 1
    hits = np.cumsum(correct[order])[ends]
    curve = [{"threshold": score, "n": int(n), "correct": int(hit),
              "coverage": n / len(labels), "accuracy": hit / n}
             for score, n, hit in zip(sorted_scores[ends], retained, hits)]
    thresholds = []
    for threshold in (0.0, 0.7, 0.8, 0.9):
        mask = confidence >= threshold
        n = int(mask.sum())
        thresholds.append({"threshold": threshold, "n": n, "correct": int(correct[mask].sum()),
                           "coverage": n / len(labels), "accuracy": float(correct[mask].mean()) if n else None})
    return {"n": len(labels), "correct": int(correct.sum()), "accuracy": float(correct.mean()),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "macro_precision": float(precision.mean()), "errors": int((~correct).sum()),
            "errors_at_least_0_8": int(((~correct) & (confidence >= 0.8)).sum()),
            "class_metrics": {name: {"support": int(support[i]), "recall": recall[i],
                                     "precision": precision[i], "f1": f1[i]}
                              for i, name in enumerate(CLASSES)},
            "confusion": matrix.tolist(), "normalized_confusion": divide(matrix, support[:, None]).tolist(),
            "ece": ece, "reliability_bins": bins, "coverage_curve": curve, "threshold_examples": thresholds}


def load_run(results_name, scores_name):
    stored = json.loads((RESULTS / results_name).read_text(encoding="utf-8"))
    metrics = {}
    with np.load(RESULTS / scores_name, allow_pickle=False) as archive:
        for split, expected_rows in SPLIT_ROWS.items():
            labels = archive[f"{split}_y"]
            if labels.shape != (expected_rows,):
                raise ValueError(f"{scores_name}: expected {expected_rows} {split} labels")
            metrics[split] = evaluate(archive[f"{split}_probs"], labels)
            recorded = stored.get(split, {})
            for key in ("accuracy", "macro_f1", "balanced_accuracy", "macro_precision"):
                if key in recorded:
                    np.testing.assert_allclose(metrics[split][key], recorded[key], rtol=0, atol=1e-10,
                                               err_msg=f"{results_name}: {split} {key}")
            matrix = recorded.get("confusion_matrix", recorded.get("confusion"))
            if matrix is not None:
                np.testing.assert_array_equal(metrics[split]["confusion"], matrix,
                                              err_msg=f"{results_name}: {split} confusion matrix")
    metrics["sources"] = [results_name, scores_name]
    return metrics


def confusion_plot(test, output_dir):
    values = np.asarray(test["normalized_confusion"])
    fig, ax = plt.subplots(figsize=(6.25, 5.1), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("teal", ["#f0f5f1", "#6daba1", "#0a5352"])
    img = ax.imshow(values, cmap=cmap, vmin=0, vmax=1)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{100*value:.0f}" if value else "·", ha="center", va="center",
                    color="white" if value >= .58 else INK, fontsize=9)
    ax.set(xticks=np.arange(len(CLASSES)), xticklabels=CLASSES, yticks=np.arange(len(CLASSES)),
           yticklabels=CLASSES, xlabel="Predicted label", ylabel="True label")
    legend = fig.colorbar(img, ax=ax, shrink=.75)
    legend.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    save(fig, output_dir / "normalized_confusion.png")


def reliability_plot(test, output_dir):
    bins = [b for b in test["reliability_bins"] if b["n"]]
    x, y, counts = ([b[key] for b in bins] for key in ("mean_score", "accuracy", "n"))
    fig, (ax, hist) = plt.subplots(2, 1, figsize=(6.25, 5.5), sharex=True, layout="constrained",
                                   gridspec_kw={"height_ratios": [3.5, 1]})
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Perfect reliability")
    ax.vlines(x, np.minimum(x, y), np.maximum(x, y), color="#9cc4b9", linewidth=1.8)
    ax.scatter(x, y, color=TEAL, s=33, zorder=3, label="Observed test bins")
    ax.set(xlim=(0, 1), ylim=(-.02, 1.02), ylabel="Observed accuracy")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid()
    hist.bar([b["left"]+.05 for b in bins], counts, width=.087, color="#92b9b1")
    hist.set(xlabel="Top-class model score", ylabel="Images")
    hist.grid(axis="y")
    save(fig, output_dir / "reliability.png")


def coverage_plot(test, output_dir):
    curve = test["coverage_curve"]
    cutoffs = test["threshold_examples"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
    ax.step([p["coverage"] for p in curve], [p["accuracy"] for p in curve], where="pre",
            linewidth=1.55, color=TEAL)
    ax.axhline(test["accuracy"], linestyle="--", color=GRAY, linewidth=1, label="All test images")
    text_locations = ((.91, .812), (.78, .856), (.61, .978), (.18, .932))
    for point, location in zip(cutoffs, text_locations):
        x, y = point["coverage"], point["accuracy"]
        ax.scatter(x, y, c=GOLD, s=25, zorder=4)
        label = "All images" if point["threshold"] == 0 else f"Score ≥ {point['threshold']:.1f}"
        ax.annotate(label, (x, y), xytext=location, fontsize=8.8, ha="center",
                    arrowprops=dict(arrowstyle="-", color=GOLD, linewidth=.75))
    ax.set(xlim=(0, 1.02), ylim=(.8, 1.02), xlabel="Coverage (fraction retained)", ylabel="Accuracy among retained images")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid()
    save(fig, output_dir / "coverage_accuracy.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Write metrics JSON and regenerated ConvNeXt plots to this directory")
    args = parser.parse_args()

    conv = load_run("convnext_tiny_v5_results.json", "convnext_tiny_v5_probs.npz")
    transnet = load_run("results_transnet_ep42_seed42.json", "probs_transnet_ep42_seed42.npz")
    older = json.loads((RESULTS / "results_FULL.json").read_text(encoding="utf-8"))["test_metrics"]

    for name, run in (("ConvNeXt-Tiny v5", conv), ("TransNet paper recipe, seed 42", transnet)):
        for split in SPLIT_ROWS:
            scores = run[split]
            print(f"{name} {split} (n={scores['n']}), accuracy={scores['accuracy']:.4f}, "
                  f"macro-F1={scores['macro_f1']:.4f}, balanced accuracy={scores['balanced_accuracy']:.4f}")
    print(f"ConvNeXt-Tiny v5 test ECE (10 bins)={conv['test']['ece']:.4f}")
    print(f"Older TransNet paper-recipe test macro-F1 (recorded)={older['macro_f1']:.4f}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"class_order": CLASSES, "convnext_tiny_v5": conv, "transnet_paper_seed42": transnet,
                   "transnet_paper_older": {"source": "results_FULL.json", "recorded_test": older}}
        (args.output_dir / "recomputed_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                                                 encoding="utf-8")
        confusion_plot(conv["test"], args.output_dir)
        reliability_plot(conv["test"], args.output_dir)
        coverage_plot(conv["test"], args.output_dir)


if __name__ == "__main__":
    main()
