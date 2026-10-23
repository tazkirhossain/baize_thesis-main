# Four-method numerical comparison

pooled local numerical-model predictions; not a global FL model or neural results

Single predeclared-seed comparison. No method was automatically selected. All methods use original holdouts; warnings and per-client metrics are retained in the source reports.

Metrics below pool the confusion counts of independent client models on their own holdouts. They do not describe a model trained by federated aggregation.

| Partition | Method | Val Adult F1 | Test Adult precision | Test Adult recall | Test Adult F1 | Test macro-F1 | Test balanced accuracy | Test accuracy | Fit warnings |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| client-category-skew-v1 | ensemble_undersampling | 0.1747 | 0.1861 | 0.1931 | 0.1896 | 0.5853 | 0.5869 | 0.9630 | 4 |
| client-category-skew-v1 | oversampling | 0.1717 | 0.1990 | 0.1766 | 0.1872 | 0.5848 | 0.5802 | 0.9656 | 0 |
| client-category-skew-v1 | smote | 0.1751 | 0.1867 | 0.1947 | 0.1906 | 0.5858 | 0.5876 | 0.9630 | 0 |
| client-category-skew-v1 | undersampling | 0.1536 | 0.1877 | 0.1731 | 0.1801 | 0.5810 | 0.5780 | 0.9647 | 0 |
| client-iid-v1 | ensemble_undersampling | 0.1568 | 0.1804 | 0.1764 | 0.1784 | 0.5799 | 0.5790 | 0.9636 | 2 |
| client-iid-v1 | oversampling | 0.1605 | 0.1781 | 0.1789 | 0.1785 | 0.5798 | 0.5800 | 0.9631 | 0 |
| client-iid-v1 | smote | 0.1595 | 0.1925 | 0.1766 | 0.1842 | 0.5831 | 0.5798 | 0.9650 | 0 |
| client-iid-v1 | undersampling | 0.1583 | 0.1652 | 0.1789 | 0.1718 | 0.5760 | 0.5791 | 0.9614 | 0 |
| client-label-skew-v1 | ensemble_undersampling | 0.1599 | 0.1830 | 0.1781 | 0.1805 | 0.5810 | 0.5799 | 0.9638 | 4 |
| client-label-skew-v1 | oversampling | 0.1522 | 0.1955 | 0.1676 | 0.1805 | 0.5815 | 0.5759 | 0.9659 | 0 |
| client-label-skew-v1 | smote | 0.1465 | 0.1912 | 0.1587 | 0.1734 | 0.5781 | 0.5717 | 0.9661 | 0 |
| client-label-skew-v1 | undersampling | 0.1569 | 0.1624 | 0.1787 | 0.1701 | 0.5751 | 0.5788 | 0.9610 | 1 |

Confusion-matrix order is [Non-adult, Adult]. Detailed matrices and client-level results are in the JSON reports. No DeBERTa or character CNN results are included.
