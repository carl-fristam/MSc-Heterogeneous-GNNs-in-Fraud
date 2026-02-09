Loading data...
Train size: 7,603,881 | Test size: 1,900,971
Train class distribution: {0: 7595983, 1: 7898}
Resampled with smote (ratio=0.1)
  Before: {0: 7595983, 1: 7898}
  After:  {0: 7595983, 1: 759598}

############################################################
# Training LogisticRegression
############################################################
Threshold for 80% recall: 0.0376
  -> Precision: 0.0013, Recall: 0.8000

============================================================
LogisticRegression (default) (threshold=0.5)
============================================================
              precision    recall  f1-score   support

           0     0.9990    0.9998    0.9994   1898996
           1     0.2635    0.0668    0.1066      1975

    accuracy                         0.9988   1900971
   macro avg     0.6313    0.5333    0.5530   1900971
weighted avg     0.9983    0.9988    0.9985   1900971

ROC-AUC: 0.7013
PR-AUC:  0.0496

============================================================
LogisticRegression (tuned) (threshold=0.03763414875315494)
============================================================
              precision    recall  f1-score   support

           0     0.9994    0.3509    0.5194   1898996
           1     0.0013    0.8000    0.0026      1975

    accuracy                         0.3513   1900971
   macro avg     0.5003    0.5754    0.2610   1900971
weighted avg     0.9984    0.3513    0.5189   1900971

ROC-AUC: 0.7013
PR-AUC:  0.0496

############################################################
# Training RandomForest
############################################################
Threshold for 80% recall: 0.0000
  -> Precision: 0.0010, Recall: 1.0000

============================================================
RandomForest (default) (threshold=0.5)
============================================================
              precision    recall  f1-score   support

           0     0.9991    0.9567    0.9775   1898996
           1     0.0042    0.1762    0.0082      1975

    accuracy                         0.9559   1900971
   macro avg     0.5017    0.5665    0.4929   1900971
weighted avg     0.9981    0.9559    0.9765   1900971

ROC-AUC: 0.6510
PR-AUC:  0.0095

============================================================
RandomForest (tuned) (threshold=0.0)
============================================================
/Users/cvdf/Developer/MSc-GNNs-in-AML/.venv/lib/python3.14/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/Users/cvdf/Developer/MSc-GNNs-in-AML/.venv/lib/python3.14/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/Users/cvdf/Developer/MSc-GNNs-in-AML/.venv/lib/python3.14/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
              precision    recall  f1-score   support

           0     0.0000    0.0000    0.0000   1898996
           1     0.0010    1.0000    0.0021      1975

    accuracy                         0.0010   1900971
   macro avg     0.0005    0.5000    0.0010   1900971
weighted avg     0.0000    0.0010    0.0000   1900971

ROC-AUC: 0.6510
PR-AUC:  0.0095

############################################################
# Training XGBoost
############################################################
Threshold for 80% recall: 0.0567
  -> Precision: 0.0018, Recall: 0.8010

============================================================
XGBoost (default) (threshold=0.5)
============================================================
              precision    recall  f1-score   support

           0     0.9991    0.9988    0.9989   1898996
           1     0.0848    0.1043    0.0936      1975

    accuracy                         0.9979   1900971
   macro avg     0.5419    0.5516    0.5463   1900971
weighted avg     0.9981    0.9979    0.9980   1900971

ROC-AUC: 0.7832
PR-AUC:  0.0943

============================================================
XGBoost (tuned) (threshold=0.05673850327730179)
============================================================
              precision    recall  f1-score   support

           0     0.9996    0.5328    0.6951   1898996
           1     0.0018    0.8010    0.0036      1975

    accuracy                         0.5331   1900971
   macro avg     0.5007    0.6669    0.3493   1900971
weighted avg     0.9986    0.5331    0.6944   1900971

ROC-AUC: 0.7832
PR-AUC:  0.0943

============================================================
SUMMARY
============================================================
Model                                  ROC-AUC     PR-AUC         F1
------------------------------------------------------------
LogisticRegression (default)            0.7013     0.0496     0.1066
LogisticRegression (tuned)              0.7013     0.0496     0.0026
RandomForest (default)                  0.6510     0.0095     0.0082
RandomForest (tuned)                    0.6510     0.0095     0.0021
XGBoost (default)                       0.7832     0.0943     0.0936
XGBoost (tuned)                         0.7832     0.0943     0.0036