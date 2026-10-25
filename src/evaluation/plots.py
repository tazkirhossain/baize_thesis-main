"""Standalone research figures; retain scope, denominator, units and missingness."""
from pathlib import Path
import textwrap

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
FAMILY_NAMES = {"character_cnn": "Character CNN", "deberta_v3_base": "DeBERTa-v3-base",
                "engineered_numeric": "Engineered numerical features"}
METHOD_LABELS = {"undersampling": "Random undersampling", "oversampling": "Random oversampling",
                 "smote": "SMOTE", "ensemble_undersampling": "Undersampling ensemble"}
INPUT_LABELS = {"url_only": "URL only", "url_plus_metadata": "URL + website context",
                "numeric": "Numerical features"}
KIND_LABELS = {"real": "Real data", "synthetic": "Synthetic smoke test", "unknown": "Unspecified data"}
PARTITION_LABELS = {"iid": "IID", "label_skew": "Non-IID label skew",
                    "category_skew": "Non-IID category skew", "unknown": "Unspecified partition"}


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
                         "axes.labelsize": 10, "axes.titleweight": "semibold",
                         "figure.titlesize": 14, "figure.titleweight": "semibold",
                         "legend.frameon": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    def save(figure, name, note):
        figure.text(0.02, 0.018, textwrap.fill(note, width=135), fontsize=9, color="#444444", va="bottom")
        for suffix in formats:
            filename = f"{name}.{suffix}"
            figure.savefig(output / filename, dpi=300, facecolor="white",
                           bbox_inches="tight", pad_inches=0.12)
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
            figure.suptitle(f"IID versus separate non-IID scenarios · {title}", fontsize=13)
            axes[0, 0].legend(ncol=3, fontsize=9)
            figure.tight_layout(rect=(0, 0.075, 1, 0.94))
            save(figure, f"01_iid_vs_non_iid_{prefix}", "Separate model/scope/seed cohort. Gaps mean not measured or not applicable. No best method selected.")

            def balancing_figure(partitions, name, title):
                if not any(row["partition"] in partitions for row in numerical):
                    return
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

            balancing_figure(("iid",), "02_balancing_iid", f"Balancing methods under IID · {title}")
            balancing_figure(("label_skew", "category_skew"), "03_balancing_non_iid", f"Balancing methods under non-IID · {title}")

            figure, axes = plt.subplots(1, 2, figsize=(12, 5.4))
            markers = ("o", "s", "^", "D")
            for col, split in enumerate(("val", "test")):
                axis = axes[col]
                for p, partition in enumerate(PARTITIONS):
                    for m, method in enumerate(METHODS):
                        row = cells.get((partition, method, split))
                        if row is not None:
                            axis.scatter(row["metrics"]["adult_recall"], row["metrics"]["adult_f1"],
                                         facecolors=COLORS[p] if p != 1 else "white",
                                         edgecolors="black" if p == 2 else COLORS[p], linewidths=1.3,
                                         marker=markers[m], s=65,
                                         label=f"{PARTITION_NAMES[p]} / {METHOD_NAMES[m]}")
                axis.set_title("Validation" if split == "val" else "Final test")
                axis.set_xlabel("Adult recall")
                axis.set_ylabel("Adult F1")
                axis.xaxis.set_major_formatter(PercentFormatter(1))
                _style(axis)
                x_values = [row["metrics"]["adult_recall"] for row in numerical]
                y_values = [row["metrics"]["adult_f1"] for row in numerical]
                axis.set_xlim(max(0, min(x_values)-0.025), min(1, max(x_values)+0.025))
                axis.set_ylim(max(0, min(y_values)-0.008), min(1, max(y_values)+0.008))
            figure.suptitle(f"Adult recall and F1 across methods · {title}", fontsize=13)
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=4, fontsize=8)
            figure.tight_layout(rect=(0, 0.27, 1, 0.94))
            save(figure, f"04_adult_recall_f1_{prefix}", "Focused scatter axes, shared across panels. Each point is a method/scenario result, not a confidence interval; exact values are in report.json.")

    cohorts = {}
    for row in report["rows"]:
        if row["split"] not in ("val", "test") or row["partition"] not in PARTITIONS:
            continue
        if row["round"] is not None:
            final = report["experiments"][row["experiment_id"]]["configuration"]["rounds"]
            if row["round"] != final:
                continue
        key = row["family"], row["data_kind"], row["scope"], row["seed"], row["classifier"], row["ratio"], row.get("training_protocol", "historical_numeric" if row["family"] == "engineered_numeric" else "end_to_end"), row.get('input_mode', 'numeric' if row['family']=='engineered_numeric' else 'url_only')
        cohorts.setdefault(key, []).append(row)
    for index, (key, cohort) in enumerate(sorted(cohorts.items())):
        family, kind, scope, seed, classifier, ratio, protocol, input_mode = key
        model_scope = "local numerical models" if family == "engineered_numeric" else f"{family} FedAvg"
        title = f"{model_scope} / {protocol} / {input_mode} / {kind} / seed {seed}"
        compare_cohort(cohort, f"{family}_{kind}_seed{seed}_{index}", title)

    experiments = sorted({row["experiment_id"] for row in report["rows"] if row["scope"] == "federated global model"})
    for experiment in experiments:
        rows = [row for row in report["rows"] if row["experiment_id"] == experiment]
        first = rows[0]
        selection_enabled = bool(report['experiments'][experiment]['configuration'].get('model_selection'))
        scope = " · ".join((
            FAMILY_NAMES.get(first['family'], first['family'].replace('_', ' ').title()),
            INPUT_LABELS.get(first.get('input_mode', 'url_only'), first.get('input_mode', 'url_only').replace('_', ' ').title()),
            KIND_LABELS.get(first['data_kind'], first['data_kind'].replace('_', ' ').title()),
            PARTITION_LABELS.get(first['partition'], first['partition'].replace('_', ' ').title()),
            METHOD_LABELS.get(first['method'], first['method'].replace('_', ' ').title()),
        ))
        figure, axes = plt.subplots(2, 4, figsize=(16, 8))
        curve_metrics = ("loss", "accuracy", "balanced_accuracy", "macro_f1",
                         "adult_precision", "adult_recall", "adult_f1", "non_adult_f1")
        for panel_index, (axis, metric) in enumerate(zip(axes.flat, curve_metrics)):
            for index, split in enumerate(("train", "val")):
                selected = sorted([row for row in rows if row["split"] == split and metric in row["metrics"]], key=lambda row: row["round"])
                if selected:
                    axis.plot([row["round"] for row in selected], [row["metrics"][metric] for row in selected],
                              color=COLORS[index], marker=("o", "s")[index], linestyle=("-", "--")[index],
                              markerfacecolor=(COLORS[index], "white")[index], label=(("Optimization pool" if selection_enabled else "Original train"), "Validation")[index])
            axis.set_title(f"({chr(97 + panel_index)})  {metric.replace('_', ' ').title()}", loc="left")
            axis.set_xlabel("Communication round")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set_ylim(bottom=0)
            _style(axis, rate=metric != "loss")
            axis.legend(fontsize=9)
        figure.suptitle(f"Federated training convergence\n{scope}", fontsize=13)
        figure.tight_layout(rect=(0, 0.09, 1, 0.94))
        save(figure, f"05_convergence_{experiment}", "Updated global model evaluated locally on unresampled training/validation data. Training-selection records, when configured, are excluded from the optimization pool. Synthetic runs are implementation checks only.")

        # Use the fixed final round, never choose a round using holdout scores.
        final_round = report["experiments"][experiment]["configuration"]["rounds"]
        final_rows = sorted((row for row in rows if row["round"] == final_round
                             and row["split"] in ("train", "val", "test")),
                            key=lambda row: ("train", "val", "test").index(row["split"]))
        if final_rows:
            figure, axes = plt.subplots(1, len(final_rows), figsize=(5*len(final_rows), 5), squeeze=False)
            for panel_index, (axis, row) in enumerate(zip(axes.flat, final_rows)):
                matrix = np.asarray(row["metrics"]["confusion_matrix"])
                axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
                support = matrix.sum(axis=1)
                for actual in range(2):
                    for predicted in range(2):
                        fraction = matrix[actual, predicted]/support[actual] if support[actual] else 0
                        axis.text(predicted, actual, f"{matrix[actual, predicted]:,}\n({fraction:.1%})",
                                  ha="center", va="center", fontsize=10,
                                  color="white" if matrix[actual, predicted] > matrix.max()/2 else "black")
                axis.set_xticks((0, 1), ("Non-adult", "Adult"))
                axis.set_yticks((0, 1), ("Non-adult", "Adult"))
                axis.set_xlabel("Predicted class")
                axis.set_ylabel("True class")
                label = {'train': 'Optimization pool' if selection_enabled else 'Original train', 'val':'Validation', 'test':'Final test'}[row['split']]
                checkpoint = row.get('checkpoint_round', row['round'])
                axis.set_title(f"({chr(97 + panel_index)})  {label} · N={matrix.sum():,}\ncheckpoint round {checkpoint}", loc="left")
                axis.grid(False)
            figure.suptitle(f"Confusion matrices at communication round {final_round}\n{scope}", fontsize=13)
            figure.tight_layout(rect=(0, 0.12, 1, 0.91))
            save(figure, f"07_confusion_{experiment}", "Each cell shows count and percentage within its true-class row. Each panel uses its own count color scale; compare printed values. No resampling of evaluation data.")

        rounds = sorted([row for row in rows if row["split"] == "val"], key=lambda row: row["round"])
        if any(row.get("timing") and row.get("communication") for row in rounds):
            figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            x = [row["round"] for row in rounds]
            training_values = [row.get("timing", {}).get("client_training_seconds_sum", np.nan) for row in rounds]
            payload_values = [row.get("communication", {}).get("total_bytes", np.nan)/1024**2 for row in rounds]
            training_bars = axes[0].bar(x, training_values, color=COLORS[0])
            axes[0].set_ylabel("Sum of client training wall time, seconds")
            payload_bars = axes[1].bar(x, payload_values, color=COLORS[1], hatch="//")
            axes[1].set_ylabel("Flower protobuf payload, MiB")
            panel_titles = ("Client training time", "Communication payload")
            for panel_index, (axis, bars, values) in enumerate(zip(axes, (training_bars, payload_bars),
                                                                   (training_values, payload_values))):
                axis.set_xlabel("Communication round")
                axis.set_xticks(x)
                _style(axis, rate=False)
                axis.set_ylim(bottom=0)
                axis.set_title(f"({chr(97 + panel_index)})  {panel_titles[panel_index]}", loc="left")
                labels = [f"{value:.1f}" if np.isfinite(value) else "" for value in values]
                axis.bar_label(bars, labels=labels, padding=3, fontsize=9)
            figure.suptitle(f"Measured training and communication costs\n{scope}", fontsize=13)
            figure.tight_layout(rect=(0, 0.15, 1, 0.92))
            save(figure, f"06_costs_{experiment}", "Client time sum is not parallel elapsed time. Payload excludes gRPC/TLS/control/retries; final round includes optional final-test exchange.")
    return files
