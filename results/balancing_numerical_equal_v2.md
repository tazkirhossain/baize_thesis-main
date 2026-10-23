# Four-method numerical comparison

pooled local numerical-model predictions; not a global FL model or neural results

Single predeclared-seed comparison. No method was automatically selected. All methods use original holdouts; warnings and per-client metrics are retained in the source reports.

Metrics below pool the confusion counts of independent client models on their own holdouts. They do not describe a model trained by federated aggregation.

| Partition | Method | Val Adult F1 | Test Adult precision | Test Adult recall | Test Adult F1 | Test macro-F1 | Test balanced accuracy | Test accuracy | Fit warnings |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| client-category-skew-v1 | ensemble_undersampling | 0.0915 | 0.0419 | 0.5607 | 0.0780 | 0.4506 | 0.6336 | 0.7032 | 22 |
| client-category-skew-v1 | oversampling | 0.0834 | 0.0388 | 0.6359 | 0.0732 | 0.4247 | 0.6377 | 0.6394 | 0 |
| client-category-skew-v1 | smote | 0.0898 | 0.0412 | 0.5902 | 0.0771 | 0.4430 | 0.6378 | 0.6834 | 0 |
| client-category-skew-v1 | undersampling | 0.0766 | 0.0369 | 0.6449 | 0.0698 | 0.4136 | 0.6297 | 0.6151 | 4 |
| client-iid-v1 | ensemble_undersampling | 0.0842 | 0.0385 | 0.5453 | 0.0720 | 0.4412 | 0.6168 | 0.6851 | 20 |
| client-iid-v1 | oversampling | 0.0869 | 0.0392 | 0.5214 | 0.0730 | 0.4481 | 0.6144 | 0.7032 | 0 |
| client-iid-v1 | smote | 0.0857 | 0.0385 | 0.5406 | 0.0718 | 0.4419 | 0.6155 | 0.6872 | 0 |
| client-iid-v1 | undersampling | 0.0775 | 0.0363 | 0.5640 | 0.0683 | 0.4283 | 0.6106 | 0.6551 | 1 |
| client-label-skew-v1 | ensemble_undersampling | 0.0837 | 0.0376 | 0.5325 | 0.0702 | 0.4398 | 0.6099 | 0.6838 | 22 |
| client-label-skew-v1 | oversampling | 0.0755 | 0.0349 | 0.6081 | 0.0661 | 0.4118 | 0.6117 | 0.6151 | 0 |
| client-label-skew-v1 | smote | 0.0817 | 0.0369 | 0.5329 | 0.0690 | 0.4372 | 0.6072 | 0.6781 | 0 |
| client-label-skew-v1 | undersampling | 0.0820 | 0.0376 | 0.5932 | 0.0707 | 0.4279 | 0.6227 | 0.6509 | 5 |

Confusion-matrix order is [Non-adult, Adult]. Detailed matrices and client-level results are in the JSON reports. No DeBERTa or character CNN results are included.
