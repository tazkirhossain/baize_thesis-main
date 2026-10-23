export type Split = 'train' | 'val' | 'test';
export type Model = 'character_cnn' | 'deberta_v3_base';
export type Partition = 'iid' | 'label_skew' | 'category_skew' | 'unknown';
export type Method = 'none' | 'undersampling' | 'oversampling' | 'smote' | 'ensemble_undersampling';
export type Metrics = {
  adult_precision: number; adult_recall: number; adult_f1: number;
  non_adult_precision: number; non_adult_recall: number; non_adult_f1: number;
  macro_f1: number; balanced_accuracy: number; accuracy: number; loss?: number;
  n_records: number; adult_count: number; non_adult_count: number;
  confusion_matrix: [[number, number], [number, number]];
};
export type Metric = keyof Pick<Metrics, 'loss' | 'accuracy' | 'adult_precision' | 'adult_recall' | 'adult_f1' | 'macro_f1' | 'balanced_accuracy'>;
export interface Client {
  client_id: number; status: string; original_train: number; resampled_train: number;
  original_adult: number; original_non_adult: number; used_adult: number; used_non_adult: number;
  duplicated: number; synthetic: number | null; examples_processed: number;
  ensemble_members: number; warmup_examples_processed: number | null;
  train_loss: number; train_seconds: number; fit_seconds: number;
  aggregation_weight: number; smote_effective_k: number | null;
}
export interface Round {
  round_number: number; metrics: Partial<Record<Split, Metrics>>; clients: Client[];
  optimization_loss: number | null;
  timing: Record<string, number | null>; communication: Record<string, number | null>;
}
export interface Run {
  experiment_id: string; name: string; model_name: Model; partition_type: Partition;
  imbalance_method: Method; data_kind: 'real' | 'synthetic' | 'unknown';
  training_protocol: 'end_to_end' | 'encoder_embeddings'; status: 'completed' | 'incomplete';
  configuration: { batch_size: number; local_epochs: number; rounds: number; n_clients: number;
    clients_per_round: number; learning_rate: number; seed: number; threshold: number;
    aggregation_weight: string; evaluate_training: boolean; evaluate_final_test: boolean;
    sequence_length: number; adult_to_non_adult: string; warmup_epochs: number };
  rounds: Round[]; comparison_group: string | null; evaluation_population_id: string | null;
  partition_data_signature: string | null;
  evaluation_protocol: string; duration_seconds: number | null; updated_at: string;
  sources: { file: string; sha256: string }[]; monitoring: string;
  prediction_probabilities_available: boolean; example_predictions_available: boolean;
}
export interface Snapshot {
  schema_version: 1; generated_at: string; runs: Run[];
  dataset: { scope: string; n_records: number; class_counts: { adult: number; non_adult: number };
    category_counts: Record<string, number>; clients: {client_id: number; total: number; adult: number; non_adult: number}[];
    source: string; source_sha256: string } | null;
  issues: {run: string; reason: string}[];
  training_templates: Record<string, Record<string, unknown>>;
  capabilities: { read_only: boolean; live_telemetry: boolean; raw_records: boolean };
}
