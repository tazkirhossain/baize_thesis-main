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
        metadata = report['experiments'][experiment]
        selection_enabled = bool(metadata['configuration'].get('model_selection'))
        final_round = metadata['configuration']['rounds']
        selection = metadata.get('selection')
        if selection_enabled and (not isinstance(selection, dict)
                                  or selection.get('source') != 'training_selection'):
            raise ValueError('Best-round figures require recorded training-selection provenance')
        # Read the recorded training-only decision; never rank validation/test scores.
        best_round = selection['round'] if selection_enabled and isinstance(selection, dict) else None
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
            if best_round is not None:
                axis.axvline(best_round, color=COLORS[2], linestyle=":", linewidth=1.5)
                if best_round != final_round:
                    axis.axvline(final_round, color="#666666", linestyle="-.", linewidth=1)
            axis.legend(fontsize=9)
        figure.suptitle(f"Federated training convergence\n{scope}", fontsize=13)
        figure.tight_layout(rect=(0, 0.09, 1, 0.94))
        threshold_note = (" Per-round decision thresholds use training selection only; loss is threshold-independent."
                          if report['experiments'][experiment]['configuration'].get('threshold_selection') else "")
        marker_note = (f" Dotted line: selected best round {best_round}; dash-dot line: last round {final_round}."
                       if best_round is not None and best_round != final_round else
                       f" Dotted line: selected best = last round {final_round}." if best_round is not None else "")
        save(figure, f"05_convergence_{experiment}", "Updated global model evaluated locally on unresampled training/validation data. Training-selection records are excluded from optimization. Synthetic runs are implementation checks only." + threshold_note + marker_note)

        if best_round is not None:
            # Test rows are logged at completion, but may evaluate an earlier checkpoint.
            def checkpoint_row(split, checkpoint):
                matches = [row for row in rows if row['split'] == split
                           and row.get('checkpoint_round', row['round']) == checkpoint]
                if len(matches) > 1:
                    raise ValueError(f"Ambiguous {split} results for checkpoint round {checkpoint}")
                return matches[0] if matches else None

            labels = (f"Selected best · round {best_round}", f"Last · round {final_round}")
            note = ("Best checkpoint uses training-selection Adult F1 only (earliest round wins ties). "
                    "Validation/test remain unresampled. Values are recorded measurements; missing evaluations are not zero.")
            figure, axes = plt.subplots(2, 2, figsize=(13, 9))
            groups = (("Adult precision", "Adult recall", "Adult F1"),
                      ("Non-adult precision", "Non-adult recall", "Non-adult F1"),
                      ("Accuracy", "Balanced accuracy", "Macro F1"), ("Loss",))
            fields = (("adult_precision", "adult_recall", "adult_f1"),
                      ("non_adult_precision", "non_adult_recall", "non_adult_f1"),
                      ("accuracy", "balanced_accuracy", "macro_f1"), ("loss",))
            for panel, (axis, titles, metrics) in enumerate(zip(axes.flat, groups, fields)):
                for index, checkpoint in enumerate((best_round, final_round)):
                    row = checkpoint_row('val', checkpoint)
                    values = [row['metrics'].get(field, np.nan) if row else np.nan for field in metrics]
                    bars = axis.bar(np.arange(len(metrics)) + (index - .5)*.36, values, .35,
                                    color=COLORS[index], hatch=("", "//")[index], label=labels[index])
                    axis.bar_label(bars, labels=[(f"{v:.3f}" if panel == 3 else f"{v:.1%}")
                                                if np.isfinite(v) else "Not evaluated" for v in values],
                                   padding=3, fontsize=9)
                axis.set_xticks(np.arange(len(metrics)), titles)
                axis.set_title(f"({chr(97+panel)})  Validation", loc='left')
                _style(axis, rate=panel != 3)
                if panel != 3:
                    axis.set_ylim(0, 1.12)
                else:
                    axis.margins(y=.2)
            axes[0, 0].legend(fontsize=9)
            thresholds = []
            for label, checkpoint in zip(labels, (best_round, final_round)):
                row = checkpoint_row('val', checkpoint)
                if row and 'threshold' in row['metrics']:
                    thresholds.append(f"{label}: threshold {row['metrics']['threshold']:.3f}")
            figure.suptitle(f"Selected best versus last checkpoint\n{scope}", fontsize=13)
            figure.tight_layout(rect=(0, .13, 1, .92))
            save(figure, f"10_best_vs_last_metrics_{experiment}", note + " " + "; ".join(thresholds))

            figure, axes = plt.subplots(2, 2, figsize=(11, 10))
            for split_index, split in enumerate(('val', 'test')):
                candidates = [checkpoint_row(split, checkpoint) for checkpoint in (best_round, final_round)]
                maximum = max([max(max(line) for line in row['metrics']['confusion_matrix'])
                               for row in candidates if row] + [1])
                for index, (axis, row) in enumerate(zip(axes[split_index], candidates)):
                    title = f"{labels[index]}\n{'Validation' if split == 'val' else 'Final test'}"
                    if row is None:
                        axis.set_facecolor('#F2F2F2')
                        axis.text(.5, .5, "Not evaluated", ha='center', va='center', transform=axis.transAxes)
                        axis.set_xticks([])
                        axis.set_yticks([])
                    else:
                        matrix = np.asarray(row['metrics']['confusion_matrix'])
                        axis.imshow(matrix, cmap='Blues', vmin=0, vmax=maximum)
                        for actual in range(2):
                            support = matrix[actual].sum()
                            for predicted in range(2):
                                value = matrix[actual, predicted]
                                fraction = value/support if support else 0
                                axis.text(predicted, actual, f"{value:,}\n({fraction:.1%})", ha='center', va='center',
                                          color='white' if value > maximum/2 else 'black')
                        axis.set_xticks((0, 1), ('Non-adult', 'Adult'))
                        axis.set_yticks((0, 1), ('Non-adult', 'Adult'))
                        axis.set_xlabel('Predicted class')
                        axis.set_ylabel('True class')
                        title += f" · N={matrix.sum():,}"
                        if 'threshold' in row['metrics']:
                            title += f" · threshold {row['metrics']['threshold']:.3f}"
                    axis.set_title(title, fontsize=10, loc='left')
                    axis.grid(False)
            figure.suptitle(f"Best and last checkpoint confusion matrices\n{scope}", fontsize=13)
            figure.tight_layout(rect=(0, .12, 1, .92))
            save(figure, f"11_best_vs_last_confusion_{experiment}", note +
                 " Cells show counts and true-class percentages; color scale is shared within each split. "
                 "Final test is associated with its actual checkpoint, not its logging round.")

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
            thresholds = (" Training-selected thresholds: " + "; ".join(
                f"{row['split']}={row['metrics']['threshold']:.3f}" for row in final_rows) + "."
                if threshold_note else "")
            save(figure, f"07_confusion_{experiment}", "Each cell shows count and percentage within its true-class row. Each panel uses its own count color scale; compare printed values. No resampling of evaluation data." + thresholds)

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

        client_rounds = report["experiments"][experiment].get("client_training_rounds", [])
        if client_rounds:
            final_clients = client_rounds[-1]["clients"]
            labels = [f"Client {row['client_id'] + 1}" for row in final_clients]
            x = np.arange(len(final_clients))
            original_share = [100*row["original_adult"]/row["original_train"] if row["original_train"] else 0
                              for row in final_clients]
            balanced_share = [100*row["used_adult"]/row["resampled_train"] if row["resampled_train"] else 0
                              for row in final_clients]
            weights = [row["aggregation_weight"] for row in final_clients]
            figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
            bars = axes[0].bar(x, original_share, color=COLORS[0])
            axes[0].set_title("(a)  Original Adult prevalence", loc="left")
            axes[0].set_ylabel("Adult share of local optimization pool, %")
            axes[0].bar_label(bars, labels=[f"{value:.2f}%" for value in original_share], padding=3, fontsize=8)
            bars = axes[1].bar(x, balanced_share, color=COLORS[1], hatch="//")
            axes[1].axhline(50, color="#444444", linewidth=1, linestyle=":", label="1:1 reference")
            axes[1].set_title("(b)  Adult share after balancing", loc="left")
            axes[1].set_ylabel("Adult share of local training view, %")
            axes[1].bar_label(bars, labels=[f"{value:.1f}%" for value in balanced_share], padding=3, fontsize=8)
            axes[1].legend(fontsize=8)
            bars = axes[2].bar(x, weights, color=COLORS[2])
            axes[2].set_title("(c)  FedAvg aggregation weight", loc="left")
            axes[2].set_ylabel("Training examples used for weighting")
            axes[2].bar_label(bars, labels=[f"{value:,}" for value in weights], padding=3, fontsize=8)
            for axis in axes:
                axis.set_xticks(x, labels, rotation=25, ha="right")
                axis.set_ylim(bottom=0)
                _style(axis, rate=False)
            figure.suptitle(f"Client data balance and aggregation · round {client_rounds[-1]['round']}\n{scope}", fontsize=13)
            figure.tight_layout(rect=(0, 0.13, 1, 0.90))
            save(figure, f"08_client_balance_{experiment}",
                 "Non-adult share is the complement of Adult share. Balancing changes only the local training view; FedAvg uses the declared aggregation policy.")

            round_numbers = [item["round"] for item in client_rounds]
            client_ids = list(range(report["experiments"][experiment]["configuration"]["n_clients"]))
            matrices = []
            for field in ("train_loss", "train_seconds"):
                matrix = np.full((len(client_ids), len(round_numbers)), np.nan)
                for column, item in enumerate(client_rounds):
                    for row in item["clients"]:
                        matrix[row["client_id"], column] = row[field]
                matrices.append(matrix)
            figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
            heat_titles = ("Local optimization loss", "Local training time, seconds")
            for panel_index, (axis, matrix, title) in enumerate(zip(axes, matrices, heat_titles)):
                palette = plt.colormaps["Blues"].copy()
                palette.set_bad("#EEEEEE")
                image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=palette)
                axis.set_title(f"({chr(97 + panel_index)})  {title}", loc="left")
                axis.set_xticks(np.arange(len(round_numbers)), round_numbers)
                axis.set_yticks(np.arange(len(client_ids)), [f"Client {client_id + 1}" for client_id in client_ids])
                axis.set_xlabel("Communication round")
                for row_index in range(matrix.shape[0]):
                    for column in range(matrix.shape[1]):
                        value = matrix[row_index, column]
                        if np.isfinite(value):
                            axis.text(column, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8,
                                      color="white" if value > np.nanmax(matrix)*0.58 else "black")
                figure.colorbar(image, ax=axis, shrink=0.78)
                axis.grid(False)
            figure.suptitle(f"Client optimization heterogeneity across rounds\n{scope}", fontsize=13)
            figure.tight_layout(rect=(0, 0.12, 1, 0.90))
            save(figure, f"09_client_heterogeneity_{experiment}",
                 "Grey cells indicate clients not sampled in that round. Loss is measured on each balanced local training view and is not a validation metric.")
    return files
