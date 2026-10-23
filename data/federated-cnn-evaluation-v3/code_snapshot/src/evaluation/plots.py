"""Standalone research figures; retain scope, denominator, units and missingness."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np

PARTITIONS = ("iid", "label_skew", "category_skew")
METHODS = ("undersampling", "oversampling", "smote", "ensemble_undersampling")
METHOD_NAMES = ("Under", "Over", "SMOTE", "Ensemble")
COLORS = ("#315E87", "#AF821A", "#C16338")
PARTITION_NAMES = ("IID", "Label skew", "Category skew")


def _style(axis, *, rate=True):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.set_axisbelow(True)
    if rate:
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylim(bottom=0)


def export_plots(report, output, formats=("png", "svg")):
    output = Path(output)
    files = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "svg.fonttype": "none"})

    def save(figure, name, note):
        figure.text(0.02, 0.018, note, fontsize=9, color="#444444", va="bottom")
        for suffix in formats:
            filename = f"{name}.{suffix}"
            figure.savefig(output / filename, dpi=160, facecolor="white")
            files.append(filename)
        plt.close(figure)

    def compare_cohort(numerical, prefix, title):
        # Never silently overwrite duplicate experiment cells (e.g. multiple seeds).
        cells = {}
        for row in numerical:
            key = row["partition"], row["method"], row["split"]
            if key in cells:
                raise ValueError("separate report inputs by numerical seed/configuration; duplicate plot cell")
            cells[key] = row
        if numerical:
            figure, axes = plt.subplots(2, 2, figsize=(12, 7.5))
            for metric_index, metric in enumerate(("adult_recall", "adult_f1")):
                max_value = max(row["metrics"][metric] for row in numerical)
                for col, split in enumerate(("val", "test")):
                    axis = axes[metric_index, col]
                    x = np.arange(4)
                    for position, partition in enumerate(PARTITIONS):
                        values = [cells.get((partition, method, split), {}).get("metrics", {}).get(metric, np.nan) for method in METHODS]
                        axis.bar(x + (position-1)*0.24, values, 0.23, color=COLORS[position],
                                 hatch=("", "//", "..")[position], label=PARTITION_NAMES[position])
                    axis.set_xticks(x, METHOD_NAMES)
                    axis.set_ylim(0, min(1, max(0.1, max_value*1.22)))
                    axis.set_title(f"{'Validation' if split == 'val' else 'Final test'} · {metric.replace('_', ' ').title()}")
                    _style(axis)
            figure.suptitle("IID versus separate non-IID scenarios · numerical models", fontsize=14)
            axes[0, 0].legend(ncol=3, fontsize=9)
            figure.tight_layout(rect=(0, 0.075, 1, 0.94))
            save(figure, f"01_iid_vs_non_iid_{prefix}", "Separate model/scope/seed cohort. Gaps mean not measured or not applicable. No best method selected.")

            def balancing_figure(partitions, name, title):
                figure, axes = plt.subplots(len(partitions), 2, figsize=(12, 3.4*len(partitions)+1), squeeze=False)
                for row_index, partition in enumerate(partitions):
                    for col, split in enumerate(("val", "test")):
                        axis = axes[row_index, col]
                        for index, metric in enumerate(("adult_precision", "adult_recall", "adult_f1")):
                            values = [cells.get((partition, method, split), {}).get("metrics", {}).get(metric, np.nan) for method in METHODS]
                            axis.bar(np.arange(4)+(index-1)*0.24, values, 0.23,
                                     color=COLORS[index], hatch=("", "//", "..")[index],
                                     label=metric.replace("adult_", "").title())
                        axis.set_xticks(np.arange(4), METHOD_NAMES)
                        axis.set_title(f"{PARTITION_NAMES[PARTITIONS.index(partition)]} · {'Validation' if split == 'val' else 'Final test'}")
                        axis.set_ylim(0, 1)
                        _style(axis)
                axes[0, 0].legend(ncol=3, fontsize=9)
                figure.suptitle(title, fontsize=14)
                figure.tight_layout(rect=(0, 0.09 if len(partitions) == 1 else 0.065, 1, 0.93))
                save(figure, f"{name}_{prefix}", "Original holdouts; same model/scope/seed cohort. No uncertainty intervals from a single seed.")

            balancing_figure(("iid",), "02_balancing_iid", "Balancing methods under IID · numerical models")
            balancing_figure(("label_skew", "category_skew"), "03_balancing_non_iid", "Balancing methods under non-IID · numerical models")

            figure, axes = plt.subplots(1, 2, figsize=(12, 5.4))
            markers = ("o", "s", "^", "D")
            for col, split in enumerate(("val", "test")):
                axis = axes[col]
                for p, partition in enumerate(PARTITIONS):
                    for m, method in enumerate(METHODS):
                        row = cells.get((partition, method, split))
                        if row is not None:
                            axis.scatter(row["metrics"]["adult_recall"], row["metrics"]["adult_f1"],
                                         c=COLORS[p], marker=markers[m], s=65,
                                         label=f"{PARTITION_NAMES[p]} / {METHOD_NAMES[m]}")
                axis.set_title("Validation" if split == "val" else "Final test")
                axis.set_xlabel("Adult recall")
                axis.set_ylabel("Adult F1")
                axis.xaxis.set_major_formatter(PercentFormatter(1))
                _style(axis)
                axis.set_xlim(left=0)
            figure.suptitle("Adult recall and F1 across methods · numerical models", fontsize=14)
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=4, fontsize=8)
            figure.tight_layout(rect=(0, 0.27, 1, 0.94))
            save(figure, f"04_adult_recall_f1_{prefix}", "Each point is a method/scenario result, not a confidence interval; exact values and supports are in report.json.")

    cohorts = {}
    for row in report["rows"]:
        if row["split"] not in ("val", "test") or row["partition"] not in PARTITIONS:
            continue
        if row["round"] is not None:
            final = report["experiments"][row["experiment_id"]]["configuration"]["rounds"]
            if row["round"] != final:
                continue
        key = row["family"], row["data_kind"], row["scope"], row["seed"], row["classifier"], row["ratio"]
        cohorts.setdefault(key, []).append(row)
    for index, (key, cohort) in enumerate(sorted(cohorts.items())):
        family, kind, scope, seed, classifier, ratio = key
        title = f"{family} / {kind} / seed {seed}"
        compare_cohort(cohort, f"{family}_{kind}_seed{seed}_{index}", title)

    experiments = sorted({row["experiment_id"] for row in report["rows"] if row["scope"] == "federated global model"})
    for experiment in experiments:
        rows = [row for row in report["rows"] if row["experiment_id"] == experiment]
        first = rows[0]
        scope = f"{first['family']} · {first['data_kind']} · {first['partition']} · {first['method']}"
        figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for axis, metric in zip(axes, ("loss", "adult_recall", "adult_f1")):
            for index, split in enumerate(("train", "val")):
                selected = sorted([row for row in rows if row["split"] == split and metric in row["metrics"]], key=lambda row: row["round"])
                if selected:
                    axis.plot([row["round"] for row in selected], [row["metrics"][metric] for row in selected],
                              color=COLORS[index], marker=("o", "s")[index], linestyle=("-", "--")[index],
                              markerfacecolor=(COLORS[index], "white")[index], label=("Original train", "Validation")[index])
            axis.set_title(metric.replace("_", " ").title())
            axis.set_xlabel("Communication round")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set_ylim(bottom=0)
            _style(axis, rate=metric != "loss")
            axis.legend(fontsize=9)
        figure.suptitle(f"Training convergence · {scope}", fontsize=12)
        figure.tight_layout(rect=(0, 0.13, 1, 0.92))
        save(figure, f"05_convergence_{experiment}", "Updated global model evaluated locally on original train/validation data. Synthetic runs are implementation checks only.")

        rounds = sorted([row for row in rows if row["split"] == "val"], key=lambda row: row["round"])
        if any(row.get("timing") and row.get("communication") for row in rounds):
            figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            x = [row["round"] for row in rounds]
            axes[0].bar(x, [row.get("timing", {}).get("client_training_seconds_sum", np.nan) for row in rounds], color=COLORS[0])
            axes[0].set_ylabel("Sum of client training wall time, seconds")
            axes[1].bar(x, [row.get("communication", {}).get("total_bytes", np.nan)/1024**2 for row in rounds], color=COLORS[1], hatch="//")
            axes[1].set_ylabel("Flower protobuf payload, MiB")
            for axis in axes:
                axis.set_xlabel("Communication round")
                axis.set_xticks(x)
                _style(axis, rate=False)
                axis.set_ylim(bottom=0)
            figure.suptitle(f"Measured training and communication · {scope}", fontsize=11)
            figure.tight_layout(rect=(0, 0.15, 1, 0.92))
            save(figure, f"06_costs_{experiment}", "Client time sum is not parallel elapsed time. Payload counts exclude gRPC/TLS/control/retry overhead; they are not measured network traffic.")
    return files
