"""Readable summaries from already measured global evaluation, never refitting."""


def print_round_summary(report, *, split, rounds):
    rnd = report['round']
    print(f'\n=== GLOBAL EVALUATION: round {rnd}/{rounds} | {split} ===', flush=True)
    print('Updated global model; original local data; summed confusion counts across five clients.', flush=True)
    rows = [('Original train', report['train'])] if split == 'val' and 'train' in report else []
    rows.append(('Validation' if split == 'val' else 'Final test', report[split]))
    print(f'{"Split":<16} {"N":>10} {"Loss":>9} {"Accuracy":>9} {"Balanced":>9} {"Macro-F1":>9}', flush=True)
    for name, values in rows:
        print(f'{name:<16} {values["n_records"]:>10,} {values["loss"]:>9.4f} '
              f'{values["accuracy"]:>9.4f} {values["balanced_accuracy"]:>9.4f} {values["macro_f1"]:>9.4f}', flush=True)
        print(f'  Adult:     precision={values["adult_precision"]:.4f}  recall={values["adult_recall"]:.4f}  F1={values["adult_f1"]:.4f}', flush=True)
        print(f'  Non-adult: precision={values["non_adult_precision"]:.4f}  recall={values["non_adult_recall"]:.4f}  F1={values["non_adult_f1"]:.4f}', flush=True)
        print(f'  Confusion [[TN, FP], [FN, TP]]: {values["confusion_matrix"]}', flush=True)
    timing, communication = report.get('timing', {}), report.get('communication', {})
    if 'round_wall_seconds' in timing:
        print(f'  Round wall time: {timing["round_wall_seconds"]:.1f}s; '
              f'summed client training: {timing["client_training_seconds_sum"]:.1f}s', flush=True)
    if 'total_bytes' in communication:
        print(f'  Serialized application payload: {communication["total_bytes"]/1024**2:.2f} MiB', flush=True)
    print('Rates are fractions (0 to 1). No parameters or thresholds are changed from these scores.\n', flush=True)
