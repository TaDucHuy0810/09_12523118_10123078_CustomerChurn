# Model evaluation and explanation

## Why these metrics

The target `Churn Label` is imbalanced: churn is about 26.5% of the cleaned data. Accuracy alone can look acceptable while missing many churn customers, so the primary selection metric is F1-score. F1 balances Precision and Recall. Recall measures how many real churn customers are found; Precision measures how many flagged customers are truly churn. ROC-AUC evaluates ranking quality across thresholds. Accuracy is retained for context, not as the sole selection rule.

The split is stratified 80/20 with random state 42. Model selection uses 5-fold stratified cross-validation on the training set, so the holdout test set is not used to choose the model.

## Model-by-model interpretation

### Logistic Regression

- Strength: coefficients and direction of influence are comparatively easy to explain; it gives the best ROC-AUC (0.843) and the best cross-validation F1 (0.645) in this experiment.
- Limitation: it assumes a mostly linear decision boundary after preprocessing, so complex interactions may be underrepresented.
- Dataset result: it is the default because its ranking quality and cross-validation F1 are strongest and its performance is stable across folds.

### KNN

- Strength: simple non-parametric baseline; it achieved the highest holdout Accuracy (0.758).
- Limitation: prediction is slower because it compares a new point with training points, and the result is sensitive to scaling and the local neighborhood.
- Dataset result: Accuracy is highest, but F1 is only 0.548 and Recall is 0.553. It correctly classifies many non-churn customers, while missing more churn customers than Logistic Regression. This is why Accuracy alone would select the wrong model for the business goal.

### Decision Tree

- Strength: rules are easy to present to a reviewer and it has the highest Recall (0.813), so it finds the most churn customers.
- Limitation: a tree can overfit local splits; its Precision is only 0.478, so it creates more false alarms.
- Dataset result: it is useful when missing a churn customer is more costly than contacting a false positive, but its lower Accuracy and Precision make it less balanced than Logistic Regression.

### Naive Bayes

- Strength: fast to train and predict, with high Recall (0.797); it catches many churn customers.
- Limitation: it assumes feature independence, which is not fully true here because Contract, Internet Service, payment and charges are related.
- Dataset result: it produces many positive churn predictions, so it catches churn but also creates false alarms. Its Precision (0.467), F1 (0.589) and ROC-AUC (0.811) are below Logistic Regression.

## Why Logistic Regression is the default

Logistic Regression is not the highest on every individual holdout metric. KNN has higher Accuracy and Decision Tree has higher Recall. However, this project prioritizes balanced churn detection and generalization. Logistic Regression has the best ROC-AUC and cross-validation F1, while keeping Recall high (0.789). It is therefore the most defensible default; the UI and API still allow the other three models to be compared and called.

## Preprocessing decisions

- Numeric charges are parsed as numbers; invalid values are removed because the model cannot interpret them safely.
- Numeric features use median imputation and StandardScaler. Scaling is essential for KNN and also keeps numeric magnitudes comparable for Logistic Regression.
- Categorical features use most-frequent imputation and One-Hot encoding. One-Hot is preferred to arbitrary integer labels because categories such as contract type do not have a natural numeric order.
- Identifier, geographic, target-derived and leakage-prone columns are removed: `CustomerID`, location fields, `Churn Value`, `Churn Score` and `Churn Reason`.
- The preprocessor is fit inside the sklearn pipeline after the split, preventing test information from leaking into training.
