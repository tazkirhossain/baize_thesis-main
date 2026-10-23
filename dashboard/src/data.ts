import type { Method, Metric, Partition, Round, Run, Snapshot, Split } from './types';

export const MODEL_NAMES = { character_cnn: 'Character CNN', deberta_v3_base: 'DeBERTa-v3-base' };
export const PARTITION_NAMES: Record<Partition, string> = { iid: 'IID', label_skew: 'Non-IID · label skew', category_skew: 'Non-IID · category skew', unknown: 'Unspecified / smoke fixture' };
export const METHOD_NAMES: Record<Method, string> = { none: 'No balancing', undersampling: 'Random undersampling', oversampling: 'Random oversampling', smote: 'SMOTE', ensemble_undersampling: 'Ensemble undersampling' };
export const METRIC_NAMES: Record<Metric, string> = { loss: 'Loss', accuracy: 'Accuracy', adult_precision: 'Adult precision', adult_recall: 'Adult recall', adult_f1: 'Adult F1', macro_f1: 'Macro-F1', balanced_accuracy: 'Balanced accuracy' };
export const SPLIT_NAMES: Record<Split, string> = { train: 'Original training', val: 'Validation', test: 'Final test' };
export const COLORS = ['#68dcca', '#9e93ff', '#65a9f6', '#efbe75', '#ec91b6', '#b8d67b', '#e88b79', '#82c8e9'];
export const ADULT_COLOR = '#9e93ff';
export const NON_ADULT_COLOR = '#59cbbb';
export const count = (value: number | null | undefined) => value == null ? '—' : value.toLocaleString('en-US');
export const pct = (value: number | null | undefined) => value == null ? 'Not measured' : `${(value*100).toFixed(1)}%`;
export function duration(value: number | null | undefined) {
  if (value == null) return 'Not recorded';
  if (value < 60) return `${value.toFixed(1)} s`;
  if (value < 3600) return `${(value/60).toFixed(1)} min`;
  return `${(value/3600).toFixed(1)} h`;
}
export const runColor = (id: string) => COLORS[Array.from(id).reduce((v, c) => v+c.charCodeAt(0), 0)%COLORS.length];
export const roundsIn = (run: Run, range: [number, number]) => run.rounds.filter(r => r.round_number >= range[0] && r.round_number <= range[1]);
export const latestRound = (run: Run | undefined, range: [number, number]): Round | undefined => run && roundsIn(run, range).at(-1);
export function compatible(runs: Run[]): boolean {
  if (runs.length <= 1) return true;
  if (!runs[0].comparison_group || !runs.every(r => r.comparison_group === runs[0].comparison_group)) return false;
  return runs.every(run => runs.filter(r => r.partition_type === run.partition_type)
    .every(r => Boolean(run.partition_data_signature) && r.partition_data_signature === run.partition_data_signature));
}
export async function fetchSnapshot(signal?: AbortSignal): Promise<Snapshot> {
  const response = await fetch('/api/experiments', { signal, cache: 'no-store' });
  if (!response.ok) throw new Error(`Results API returned HTTP ${response.status}`);
  const value = await response.json();
  if (value.schema_version !== 1 || !Array.isArray(value.runs)) throw new Error('Unsupported results schema');
  return value;
}
const csvCell = (value: unknown) => {
  let text = value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);
  if (/^[=+@\-\t\r]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
};
export function resultsCsv(runs: Run[], range: [number, number], split: Split): string {
  const headers = ['experiment_id', 'experiment_name', 'model_name', 'partition_type', 'imbalance_method', 'data_kind',
    'training_protocol', 'evaluation_split', 'round_number', 'batch_size', 'local_epochs', 'federated_rounds', 'seed',
    'adult_precision', 'adult_recall', 'adult_f1', 'non_adult_precision', 'non_adult_recall', 'non_adult_f1',
    'macro_f1', 'balanced_accuracy', 'accuracy', 'loss', 'n_records', 'confusion_matrix',
    'client_training_duration_seconds', 'round_duration_seconds', 'communication_payload_bytes', 'status'];
  const rows: unknown[][] = [headers];
  for (const run of runs) for (const round of roundsIn(run, range)) {
    const m = round.metrics[split];
    if (!m) continue;
    rows.push([run.experiment_id, run.name, run.model_name, run.partition_type, run.imbalance_method, run.data_kind,
      run.training_protocol, split, round.round_number, run.configuration.batch_size, run.configuration.local_epochs,
      run.configuration.rounds, run.configuration.seed, m.adult_precision, m.adult_recall, m.adult_f1,
      m.non_adult_precision, m.non_adult_recall, m.non_adult_f1, m.macro_f1, m.balanced_accuracy,
      m.accuracy, m.loss, m.n_records, m.confusion_matrix, round.timing.client_training_seconds_sum,
      round.timing.round_wall_seconds, round.communication.total_bytes, run.status]);
  }
  return rows.map(row => row.map(csvCell).join(',')).join('\r\n');
}
export function downloadCsv(runs: Run[], range: [number, number], split: Split) {
  const url = URL.createObjectURL(new Blob([resultsCsv(runs, range, split)], {type: 'text/csv;charset=utf-8'}));
  const a = document.createElement('a'); a.href = url; a.download = `federated-results-${split}.csv`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
