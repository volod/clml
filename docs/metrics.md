# Metrics Reference and Method Interpretation Guide

Every `clml run` prints a one-line `metric_interpretation` and logs the same string as an MLflow tag.
This document defines every metric used, explains the quality thresholds, and provides per-method
interpretation notes so you can quickly judge whether a result is meaningful.

---

## Metrics Glossary

### Classification metrics

| Metric | Full name | Range | Definition |
|--------|-----------|-------|------------|
| **Accuracy** | Accuracy score | 0–100% | Fraction of samples classified correctly. Simple but misleading on imbalanced data — a model predicting the majority class always gets high accuracy without learning anything useful. |
| **Macro-F1** | Macro-averaged F1 score | 0–100% | Harmonic mean of precision and recall, averaged equally across all classes regardless of their frequency. More informative than accuracy when classes are imbalanced; a low Macro-F1 with high accuracy signals a class is being ignored. |
| **AUC** | Area Under the ROC Curve | 0–1 | Probability that the model ranks a random positive above a random negative. Threshold-independent: measures ranking quality, not classification quality. AUC 0.5 = random guessing; AUC 1.0 = perfect separation. Computed per class then macro-averaged for multiclass. |

### Regression metrics

| Metric | Full name | Range | Definition |
|--------|-----------|-------|------------|
| **R²** | Coefficient of determination | −∞ to 1 | Fraction of target variance explained by the model. R² = 1 means perfect fit; R² = 0 means the model predicts the mean for every sample; R² < 0 means the model is worse than the mean. Unitless — comparable across datasets. |
| **RMSE** | Root mean squared error | ≥ 0 | Square root of mean squared prediction error. Penalizes large errors more than MAE. Units match the target variable. Must be read in context of the target scale (e.g., RMSE 60 000 on house prices vs RMSE 1.5 on calibration data). |
| **MAE** | Mean absolute error | ≥ 0 | Average absolute prediction error. Robust to outliers; same units as the target. MAE ≤ RMSE always; a large gap between them means a few samples have very large errors. |

### Clustering metrics

| Metric | Full name | Range | Definition |
|--------|-----------|-------|------------|
| **Silhouette** | Silhouette coefficient | −1 to 1 | Measures how similar a sample is to its own cluster vs the nearest other cluster. Averaged over all samples. Higher = denser, better-separated clusters. Negative values indicate samples are likely assigned to the wrong cluster. |
| **ARI** | Adjusted Rand Index | −1 to 1 | Compares cluster assignments to ground-truth labels, corrected for chance. ARI = 1: perfect recovery; ARI = 0: random; ARI < 0: worse than random. Only meaningful when true labels are available (built-in `blobs`/`moons` datasets). |

### Dimensionality reduction metrics

| Metric | Definition |
|--------|------------|
| **Components** | Number of output dimensions requested (e.g., 2 for visualization). Not a quality metric — projection quality must be judged visually from the scatter plot artifact. |
| **Variance explained** | For linear methods (PCA, TruncatedSVD): cumulative fraction of input variance retained by the components. A single number for all requested components. |

### Anomaly detection metrics

Anomaly methods are evaluated on labeled datasets where ground truth is known.

| Metric | Definition |
|--------|------------|
| **AUC** | Same definition as classification AUC: how well the anomaly score separates true anomalies from normal samples. |
| **Macro-F1** | Classification F1 at the default threshold (0 decision boundary). Lower than AUC because the threshold is fixed. |

### Density estimation metrics

| Metric | Definition |
|--------|------------|
| **Mean log-likelihood** | Average log p(x) assigned to held-out samples under the fitted density model. Higher (less negative) = the model assigns higher probability to the test data. The ± value is the standard deviation across samples — large spread indicates the model is uncertain over parts of the space. |

### Time series metrics

| Metric | Full name | Definition |
|--------|-----------|------------|
| **MAE** | Mean absolute error | Average absolute forecast error in target units. |
| **RMSE** | Root mean squared error | Penalizes large forecast spikes more than MAE. |
| **MAPE** | Mean absolute percentage error | MAE as a fraction of the true value; scale-free. Expressed as %. Values below 5% are excellent for monthly data; below 15% is acceptable. Undefined when true values are zero. |

### Survival analysis metrics

| Metric | Definition |
|--------|------------|
| **Cox C-index** | Concordance index: fraction of comparable patient pairs where the model ranks the earlier-event patient higher. C-index = 0.5 = random; C-index = 1.0 = perfect. Analogous to AUC for time-to-event data. |
| **Log-rank p-value** | Tests whether survival curves for different groups are statistically different. p < 0.05 = significant; the smaller the p-value, the stronger the evidence of different survival. |
| **Event fraction** | Proportion of samples that experienced the event (death, churn, failure). Useful context for interpreting log-rank results. |

### Feature selection metrics

| Metric | Definition |
|--------|------------|
| **Selected features** | Number of features retained after the selection step. |
| **Downstream macro-F1** | F1 score of a classifier trained on only the selected features. The primary quality signal: higher means the retained features carry more discriminative information. |

### Optimization metrics

| Metric | Definition |
|--------|------------|
| **Optimal profit / Optimized sales** | Objective value at the solution. For maximization problems: higher = better. |
| **Resource utilization** | Average fraction of capacity consumed across resources. High utilization = binding constraints; low = slack in the model. |
| **Lift** | Improvement over the status-quo baseline (current budget, naive allocation). |

### Portfolio optimization metrics

| Metric | Definition |
|--------|------------|
| **Expected return** | Annualized portfolio return at the optimal allocation. |
| **Volatility** | Annualized portfolio standard deviation (risk). |
| **Sharpe-like** | Return divided by volatility. Higher = better risk-adjusted return. Not annualized vs a risk-free rate; purely comparative across runs. |

### Incremental learning metrics

| Metric | Definition |
|--------|------------|
| **Prequential accuracy** | Test-then-train accuracy: each sample is predicted before being used for training. Reflects real online performance without a separate test set. |
| **F1** | Prequential F1 (macro-averaged). Often lower than batch F1 because early samples are predicted by an untrained model. |
| **Samples** | Total samples processed. |

---

## Quality Thresholds

These thresholds are used to produce the qualitative labels in `metric_interpretation`.

### AUC

| Range | Label |
|-------|-------|
| ≥ 0.90 | excellent |
| ≥ 0.75 | good |
| < 0.75 | poor |

### R²

| Range | Label |
|-------|-------|
| ≥ 0.80 | strong fit |
| ≥ 0.50 | moderate fit |
| < 0.50 | weak fit |

### Silhouette

| Range | Label |
|-------|-------|
| > 0.50 | well-separated |
| > 0.20 | overlapping |
| ≤ 0.20 | poor |

### Cox C-index

| Range | Label |
|-------|-------|
| ≥ 0.70 | good |
| ≥ 0.60 | moderate |
| < 0.60 | poor |

---

## Per-Method Interpretation

### Classification

| Method | Purpose | Key metrics | How to interpret | Example result |
|--------|---------|-------------|-----------------|----------------|
| `adaboost_classifier` | Sequentially reweights weak decision stumps to focus on hard cases; sensitive to noisy labels. | Accuracy, Macro-F1, AUC | Good baseline for structured data. If AUC is good but Macro-F1 is low, check for class imbalance. | Accuracy 76.3%; Macro-F1 73.2%; AUC 0.809 (good) |
| `bagging_classifier` | Reduces variance by averaging predictions of many bootstrap-trained estimators. | Accuracy, Macro-F1, AUC | AUC should be close to or better than a single tree. Large gap between AUC and Macro-F1 = threshold sensitivity. | Accuracy 75.7%; Macro-F1 72.7%; AUC 0.783 (good) |
| `bernoulli_nb` | Naive Bayes for binary features (thresholded or count ≥ 1). | Accuracy, Macro-F1 | Best on text-like or binary-encoded data. No AUC when predict_proba is not calibrated. High accuracy + no AUC = no probability output. | Accuracy 84.4%; Macro-F1 84.4% |
| `calibrated_logistic_regression` | Logistic regression with isotonic/sigmoid calibration for reliable probability outputs. | Accuracy, Macro-F1, AUC | Choose over plain logistic regression when downstream decisions depend on well-calibrated probabilities. Small accuracy difference from baseline is acceptable. | Accuracy 77.0%; Macro-F1 73.8%; AUC 0.820 (good) |
| `catboost_classifier` | Gradient boosted trees (CatBoost) with native categorical support. | Accuracy, Macro-F1, AUC | Compare with XGBoost and LightGBM. Often wins on datasets with many categoricals. | Accuracy 77.5%; Macro-F1 75.0%; AUC 0.822 (good) |
| `category_hashing_encoder_sgd` | Feature hashing of categoricals fed to an SGD linear classifier. | Accuracy, Macro-F1, AUC | Fast and memory-efficient. Lower ceiling than tree methods; useful when cardinality is huge. | Accuracy 75.9%; Macro-F1 73.3%; AUC 0.764 (good) |
| `category_leave_one_out_encoder_logistic` | Leave-one-out target-leakage-safe encoding + logistic regression. | Accuracy, Macro-F1, AUC | Better than target encoding for avoiding leakage. Compare with plain logistic on same dataset. | Accuracy 77.3%; Macro-F1 74.5%; AUC 0.820 (good) |
| `category_target_encoder_logistic` | Mean-target encoding of categoricals + logistic regression. | Accuracy, Macro-F1, AUC | Prone to leakage with small categories; LOO encoder is safer. Useful to compare leakage impact. | Accuracy 72.5%; Macro-F1 70.4%; AUC 0.769 (good) |
| `decision_tree_classifier` | Interpretable if-then rules; high variance, typically underperforms ensembles. | Accuracy, Macro-F1, AUC | Low AUC relative to ensembles is expected. Examine the `plots/tree` artifact to inspect depth and splits. | Accuracy 68.7%; Macro-F1 66.2%; AUC 0.724 (poor) |
| `ebm_classifier` | Explainable Boosting Machine: GAM-like model with pairwise interactions; globally interpretable. | Accuracy, Macro-F1, AUC | Competitive with GBM on tabular data. Check the shape function plots in artifacts for feature-level explanations. | Accuracy 76.6%; Macro-F1 74.2%; AUC 0.806 (good) |
| `extra_trees_classifier` | Fully randomized tree ensemble; faster than Random Forest, often similar accuracy. | Accuracy, Macro-F1, AUC | High AUC with low Macro-F1 suggests good ranking ability but a skewed threshold. Try with balanced class weights. | Accuracy 62.5%; Macro-F1 38.5%; AUC 0.812 (good) |
| `gaussian_nb` | Naive Bayes assuming Gaussian feature likelihoods; extremely fast. | Accuracy, Macro-F1, AUC | Excellent when features are roughly independent. Very high AUC on `breast_cancer` is typical. Fails when features are strongly correlated. | Accuracy 93.7%; Macro-F1 93.2%; AUC 0.988 (excellent) |
| `gaussian_process_classifier` | Kernel probabilistic classifier with uncertainty estimates; O(n³) training. | Accuracy, Macro-F1 | Use on small datasets (< 5 000 samples). AUC is often omitted when multiclass approximation is used. | Accuracy 92.1%; Macro-F1 92.3% |
| `gradient_boosting_classifier` | Sklearn gradient boosted trees; also compares XGBoost, LightGBM, CatBoost. | Accuracy, Macro-F1, AUC | The comparison file shows all four variants. XGBoost/LightGBM usually train faster. | Accuracy 76.1%; Macro-F1 72.8%; AUC 0.824 (good) |
| `hist_gradient_boosting_classifier` | Histogram-based boosting (sklearn); handles missing values natively. | Accuracy, Macro-F1, AUC | Faster than `gradient_boosting_classifier` on larger data. Results should be similar or better. | Accuracy 75.4%; Macro-F1 73.0%; AUC 0.801 (good) |
| `imbalanced_balanced_random_forest` | Random forest with per-tree balanced resampling (imbalanced-learn). | Accuracy, Macro-F1, AUC | Compare Macro-F1 against plain Random Forest; if higher, the balancing is helping minority classes. | Accuracy 77.0%; Macro-F1 75.3%; AUC 0.816 (good) |
| `imbalanced_random_undersampling_logistic` | Random majority-class undersampling + logistic regression. | Accuracy, Macro-F1, AUC | Accuracy may drop vs plain logistic but Macro-F1 should increase. Useful to quantify the imbalance problem. | Accuracy 75.3%; Macro-F1 73.7%; AUC 0.819 (good) |
| `imbalanced_smote_logistic` | SMOTE synthetic minority oversampling + logistic regression. | Accuracy, Macro-F1, AUC | Compares favorably with undersampling when minority class is very small. | Accuracy 76.1%; Macro-F1 74.5%; AUC 0.817 (good) |
| `knn_classifier` | Lazy learner: predicts by majority vote of k nearest neighbors. | Accuracy, Macro-F1, AUC | Performance degrades in high-dimensional spaces (curse of dimensionality). Good on dense low-dimensional data. | Accuracy 76.2%; Macro-F1 73.2%; AUC 0.796 (good) |
| `label_propagation` | Spreads labels over a graph built from similarity; semi-supervised. | Accuracy, Macro-F1, AUC | Evaluated here on fully labeled data to upper-bound performance. In practice, use when only a fraction of labels are available. | Accuracy 95.1%; Macro-F1 95.1%; AUC 0.988 (excellent) |
| `lda_classifier` | Linear Discriminant Analysis: linear boundary based on shared covariance. | Accuracy, Macro-F1, AUC | Underperforms if classes have very different covariances (use QDA then). Fast dimensionality reduction side effect. | Accuracy 72.3%; Macro-F1 70.0%; AUC 0.769 (good) |
| `logistic_regression` | Regularized linear classifier; calibrated probabilities; interpretable coefficients. | Accuracy, Macro-F1, AUC | Strong general-purpose baseline. Coefficient artifacts show feature impact. | (in `list-methods` — not in run-all sample shown) |
| `mlp_classifier` | Feed-forward neural network for nonlinear classification. | Accuracy, Macro-F1, AUC | Needs scaled features (pipeline handles this). High variance — compare multiple trials. | Accuracy 96.5%; Macro-F1 96.3%; AUC 0.995 (excellent) |
| `multinomial_nb` | Count-based Naive Bayes for nonnegative integer features. | Accuracy, Macro-F1 | Ideal for bag-of-words text. Applied here to pixel count data (`digits`). | Accuracy 88.2%; Macro-F1 88.2% |
| `nu_svc_classifier` | SVM with ν parameter controlling margin error fraction. | Accuracy, Macro-F1, AUC | ν directly bounds the fraction of support vectors and margin errors; easier to tune than C. | Accuracy 97.9%; Macro-F1 97.8%; AUC 0.998 (excellent) |
| `passive_aggressive_classifier` | Online linear classifier (PA algorithm via SGD); updates only on errors. | Accuracy, Macro-F1 | Designed for streaming; evaluated in batch mode here. AUC omitted (no probability output). | Accuracy 77.3%; Macro-F1 74.3% |
| `qda_classifier` | Quadratic discriminant analysis: class-specific covariance for curved boundaries. | Accuracy, Macro-F1, AUC | Better than LDA when classes have different spread. Can overfit if n < features per class. | Accuracy 77.1%; Macro-F1 74.5%; AUC 0.817 (good) |
| `random_forest_classifier` | Bagged decision trees; robust, low variance, works well without tuning. | Accuracy, Macro-F1, AUC | Default first-try method for tabular classification. High AUC with low Macro-F1 = threshold issue (check class weights). | Accuracy 62.5%; Macro-F1 38.5%; AUC 0.793 (good) |
| `ridge_classifier` | Ridge regression on one-hot class labels; fast for many-class problems. | Accuracy, Macro-F1 | No probability output; AUC omitted. Often matches logistic regression accuracy at lower compute. | Accuracy 72.7%; Macro-F1 70.3% |
| `self_training_classifier` | Iteratively adds high-confidence pseudo-labels from unlabeled data. | Accuracy, Macro-F1, AUC | Evaluated in fully-labeled mode. Real-world use: supply 10–20% labels and let it generalize. | Accuracy 94.2%; Macro-F1 94.2%; AUC 0.984 (excellent) |
| `sgd_classifier` | Stochastic gradient descent on linear classifier (logistic / hinge loss). | Accuracy, Macro-F1, AUC | Scale-sensitive: pipeline normalizes features. Fast and scalable; may need more tuning than logistic regression. | Accuracy 72.3%; Macro-F1 69.1%; AUC 0.764 (good) |
| `stacking_classifier` | Meta-learner combining LogReg + RF + GBM out-of-fold predictions. | Accuracy, Macro-F1, AUC | Expect marginal improvement over the best base model; significant compute cost. | Accuracy 77.7%; Macro-F1 74.9%; AUC 0.825 (good) |
| `statsmodels_logit` | Logistic regression with full statistical inference (p-values, CIs). | Accuracy, Macro-F1, AUC | Use when coefficient significance matters (regulatory, academic). Check `summary.txt` artifact. | Accuracy 77.1%; Macro-F1 74.4%; AUC 0.822 (good) |
| `svc_classifier` | Kernel SVM with RBF kernel; high accuracy on small scaled datasets. | Accuracy, Macro-F1 | No probability output by default (AUC omitted). Excellent on `breast_cancer`; slow on > 10 k samples. | Accuracy 98.6%; Macro-F1 98.5% |
| `voting_classifier` | Soft-vote ensemble (LogReg + RF + SVC probability averages). | Accuracy, Macro-F1, AUC | Diversity of base models drives gains. Compare with individual members via `clml run`. | Accuracy 76.8%; Macro-F1 73.8%; AUC 0.822 (good) |
| `xgboost_classifier` | XGBoost regularized boosted trees. | Accuracy, Macro-F1, AUC | Compare with `gradient_boosting_classifier`. Usually faster; often similar or better. | Accuracy 77.4%; Macro-F1 74.9%; AUC 0.820 (good) |

---

### Regression

| Method | Purpose | Key metrics | How to interpret | Example result |
|--------|---------|-------------|-----------------|----------------|
| `adaboost_regressor` | Sequential boosting focusing later trees on large residuals. | R², RMSE, MAE | Sensitive to outliers — if RMSE >> MAE, a few samples have very large errors. Compare R² with Random Forest. | R² 0.644 (moderate fit); RMSE 97 033; MAE 75 676 |
| `ard_regression` | Bayesian sparse linear regression; learns per-feature regularization. | R², RMSE, MAE | Useful when you want automatic feature relevance weights. Similar R² to Ridge on dense data. | R² 0.498 (weak fit); RMSE 52.712; MAE 41.388 |
| `bagging_regressor` | Bootstrap aggregation for regression; reduces variance. | R², RMSE, MAE | Expect R² between a single tree and Random Forest. | R² 0.811 (strong fit); RMSE 70 709; MAE 53 275 |
| `bayesian_ridge_regression` | Probabilistic Ridge: infers regularization from data; per-coefficient uncertainty. | R², RMSE, MAE | Use when you need prediction intervals or principled regularization. | R² 0.862 (strong fit); RMSE 60 377; MAE 43 912 |
| `catboost_regressor` | CatBoost gradient boosting for regression. | R², RMSE, MAE | Compare with XGBoost and LightGBM. | R² 0.870 (strong fit); RMSE 58 632; MAE 43 105 |
| `decision_tree_regressor` | Piecewise-constant rule-based prediction. | R², RMSE, MAE | Lower R² than ensembles is expected; inspect the tree artifact for interpretability. | R² 0.720 (moderate fit); RMSE 86 020; MAE 66 784 |
| `ebm_regressor` | Interpretable GAM with pairwise interactions. | R², RMSE, MAE | Near-GBM accuracy with full feature-level explainability. | R² 0.834 (strong fit); RMSE 66 269; MAE 49 589 |
| `elastic_net_regression` | L1 + L2 regularized linear regression. | R², RMSE, MAE | Best for datasets with correlated predictors. Lasso zero-out: ElasticNet retains more features. | R² 0.485 (weak fit); RMSE 53.358; MAE 41.541 |
| `extra_trees_regressor` | Extremely randomized trees for regression. | R², RMSE, MAE | Often similar to Random Forest; faster. | R² 0.715 (moderate fit); RMSE 86 737; MAE 66 389 |
| `gamma_regression` | GLM with Gamma distribution for positive, right-skewed targets. | R², RMSE, MAE | Use for costs, durations, loss severity. R² here is the pseudo-R² — lower than OLS R² on same data is expected. | R² 0.788 (moderate fit); RMSE 74 831; MAE 52 185 |
| `gaussian_process_regressor` | Kernel probabilistic regressor with uncertainty bounds. | R², RMSE, MAE | Strong on small smooth datasets; degrades on large or noisy data. Negative R² signals the kernel is mismatched. | R² 0.306 (weak fit); RMSE 61.957; MAE 49.720 |
| `gradient_boosting_regressor` | Sklearn GBM with XGBoost / LightGBM / CatBoost comparison. | R², RMSE, MAE | Check `third_party_comparisons.json` artifact for all four variants side by side. | R² 0.785 (moderate fit); RMSE 75 413; MAE 57 069 |
| `hist_gradient_boosting_regressor` | Histogram-based GBM; handles missing values. | R², RMSE, MAE | Faster than `gradient_boosting_regressor`; handles large datasets. | R² 0.856 (strong fit); RMSE 61 807; MAE 47 134 |
| `huber_regression` | Robust linear regression; Huber loss reduces outlier influence. | R², RMSE, MAE | When RMSE >> MAE, outliers dominate — Huber will outperform OLS. Compare with RANSAC. | R² 0.864 (strong fit); RMSE 59 956; MAE 42 293 |
| `isotonic_regression` | Monotonic stepwise regression; 1-D only. | R², RMSE, MAE | Perfect fit possible because it can overfit to training order. Useful for calibration, not general regression. | R² 0.964 (strong fit); RMSE 1.540; MAE 1.229 |
| `kernel_ridge_regression` | Ridge regression in kernel space. | R², RMSE, MAE | Negative R² means the kernel/regularization is badly mismatched; try rbf with scaled features. | R² −0.863 (weak fit); RMSE 101.491; MAE 79.890 |
| `knn_regressor` | Average of k nearest neighbor target values. | R², RMSE, MAE | Degrades in high-dimensional spaces. Good when local structure dominates. | R² 0.684 (moderate fit); RMSE 91 418; MAE 64 144 |
| `lasso_regression` | L1-regularized linear regression; sparse coefficients. | R², RMSE, MAE | Use for automatic feature selection. Eliminated coefficients are exactly zero. | R² 0.489 (weak fit); RMSE 53.180; MAE 41.430 |
| `mlp_regressor` | Feed-forward neural network for regression. | R², RMSE, MAE | High variance; run with trials > 0 and compare R² across seeds. | R² 0.434 (weak fit); RMSE 55.927; MAE 43.255 |
| `ngboost_regressor` | Probabilistic GBM predicting a full conditional distribution. | R², RMSE, MAE | Same point metrics as other GBMs, but also logs uncertainty. Check artifacts for prediction interval plots. | R² 0.870 (strong fit); RMSE 58 678; MAE 44 079 |
| `pls_regression` | Latent components explaining predictors and target jointly. | R², RMSE, MAE | Useful for many correlated predictors and few samples. Check number of components in params. | R² 0.494 (weak fit); RMSE 52.881; MAE 41.307 |
| `poisson_regression` | GLM with Poisson distribution for non-negative count-like targets. | R², RMSE, MAE | Use for counts, rates, or frequency data. Log link keeps predictions positive. | R² 0.751 (moderate fit); RMSE 81 088; MAE 58 468 |
| `quantile_regression` | Predicts a conditional quantile (default: median) rather than mean. | R², RMSE, MAE | R² and RMSE penalize non-mean predictions — expect lower values than OLS. Purpose is robustness, not R² maximization. | R² −0.929 (weak fit); RMSE 225 837; MAE 191 710 |
| `random_forest_regressor` | Bagged trees; robust nonlinear regression. | R², RMSE, MAE | Strong default choice. Inspect feature importance artifact. | R² 0.751 (moderate fit); RMSE 81 066; MAE 61 493 |
| `ransac_regression` | Fits linear model on consensus inlier subset; rejects outliers. | R², RMSE, MAE | Compare RMSE with Huber. If RANSAC wins, data has large structural outliers. | R² 0.802 (strong fit); RMSE 72 376; MAE 51 417 |
| `ridge_regression` | L2-regularized linear regression; stable on correlated features. | R², RMSE, MAE | Baseline for linear modeling. Check `coefficients.csv` artifact for feature weights. | R² 0.485 (weak fit); RMSE 53.369; MAE 41.548 |
| `sgd_regressor` | Stochastic gradient descent linear regressor. | R², RMSE, MAE | Scalable; needs feature scaling (pipeline provides it). | R² 0.857 (strong fit); RMSE 61 573; MAE 43 252 |
| `stacking_regressor` | Ridge + RF + GBM out-of-fold predictions fed to a Ridge meta-learner. | R², RMSE, MAE | Marginal improvement over best base model; expensive. | R² 0.871 (strong fit); RMSE 58 505; MAE 43 903 |
| `statsmodels_glm_regression` | GLM with inference tables (p-values, CIs). | R², RMSE, MAE | Use when coefficient significance is required. Check `summary.txt`. | R² 0.868 (strong fit); RMSE 59 022; MAE 42 466 |
| `statsmodels_ols` | OLS with full inference output. | R², RMSE, MAE | Gold standard for interpretable linear regression. Identical point metrics to `ridge_regression` with λ=0. | R² 0.868 (strong fit); RMSE 59 022; MAE 42 466 |
| `statsmodels_robust_regression` | Iteratively reweighted least squares (Huber M-estimator). | R², RMSE, MAE | When OLS and robust are close, data has few outliers. Large gap signals outlier influence. | R² 0.866 (strong fit); RMSE 59 522; MAE 41 646 |
| `svr_regression` | Kernel SVM regressor; strong on small scaled data. | R², RMSE, MAE | Very sensitive to feature scaling and hyperparameters. Near-zero R² without tuning is common — run with trials > 0. | R² 0.009 (weak fit); RMSE 74.009; MAE 63.595 |
| `theil_sen_regression` | Median-slope robust linear regression. | R², RMSE, MAE | Highly robust but slow (O(n²)). Use for small datasets with significant outliers. | R² 0.478 (weak fit); RMSE 53.715; MAE 41.513 |
| `tweedie_regression` | GLM for non-negative possibly zero-inflated targets. | R², RMSE, MAE | Power parameter: 0=Gaussian, 1=Poisson, 2=Gamma. Tuned automatically. | R² 0.744 (moderate fit); RMSE 82 256; MAE 59 025 |
| `voting_regressor` | Ridge + RF + GBM prediction average. | R², RMSE, MAE | Diversity drives ensemble gains. | R² 0.862 (strong fit); RMSE 60 494; MAE 45 115 |
| `xgboost_regressor` | XGBoost regularized boosted trees. | R², RMSE, MAE | Compare with `gradient_boosting_regressor` and `hist_gradient_boosting_regressor`. | R² 0.855 (strong fit); RMSE 61 813; MAE 46 094 |

---

### Clustering

| Method | Purpose | Key metrics | How to interpret | Example result |
|--------|---------|-------------|-----------------|----------------|
| `affinity_propagation` | Message-passing clustering; no k required. | clusters, silhouette, ARI | Tends to over-segment — high cluster count with moderate silhouette is common. | 10 clusters; silhouette 0.195 (poor); ARI 0.523 |
| `agglomerative_clustering` | Bottom-up hierarchical clustering. | clusters, silhouette, ARI | ARI close to 1.0 = accurate cluster recovery. Check the dendrogram artifact. | 6 clusters; silhouette 0.481 (overlapping); ARI 0.825 |
| `bayesian_gaussian_mixture` | Bayesian GMM that shrinks redundant components. | clusters, silhouette, ARI | ARI = 1.000 means perfect cluster recovery. Fewer components than requested = Bayesian pruning worked. | 4 clusters; silhouette 0.705 (well-separated); ARI 1.000 |
| `birch` | Incremental compressed-tree clustering. | clusters, silhouette, ARI | Good for large datasets. Lower silhouette than k-means expected due to CF-tree approximation. | 8 clusters; silhouette 0.466 (overlapping); ARI 0.810 |
| `bisecting_kmeans` | Divisive hierarchical k-means. | clusters, silhouette, ARI | More deterministic than affinity propagation; better silhouette than agglomerative on compact blobs. | 5 clusters; silhouette 0.617 (well-separated); ARI 0.913 |
| `dbscan` | Density-based; finds arbitrary shapes and labels noise. | clusters, silhouette, ARI | Silhouette excludes noise points (label −1). High ARI = correct number and membership recovered. | 4 clusters; silhouette 0.655 (well-separated); ARI 0.999 |
| `gaussian_mixture` | Soft probabilistic clustering via EM. | clusters, silhouette, ARI | ARI 0.499 at only 2 clusters = GMM merged two true blobs. Increase n_components or use Bayesian version. | 2 clusters; silhouette 0.545 (well-separated); ARI 0.499 |
| `hdbscan` | Hierarchical DBSCAN; auto-selects stable cluster count. | clusters, silhouette, ARI | ARI near 1.0 with noise tolerance makes this the strongest density-based method. | 4 clusters; silhouette 0.705 (well-separated); ARI 1.000 |
| `kmeans` | Centroid clustering; requires k. | clusters, silhouette, ARI | High silhouette + ARI > 0.9 = well-recovered structure. Fails on non-convex clusters. | 5 clusters; silhouette 0.587 (well-separated); ARI 0.914 |
| `mean_shift` | Mode-seeking; bandwidth auto-estimated. | clusters, silhouette, ARI | (not in run-all sample; see `list-methods`) Bandwidth controls granularity — too small = many small clusters. |  |
| `minibatch_kmeans` | Mini-batch k-means for large datasets. | clusters, silhouette, ARI | Slightly lower silhouette than full k-means; much faster. | 8 clusters; silhouette 0.368 (overlapping); ARI 0.679 |
| `optics` | Density ordering that generalizes DBSCAN across densities. | clusters, silhouette, ARI | ARI near 1.0 without specifying epsilon. | 4 clusters; silhouette 0.705 (well-separated); ARI 1.000 |
| `spectral_clustering` | Graph affinity matrix + eigenvector cut. | clusters, silhouette, ARI | Excellent on non-convex clusters (rings, moons). Memory O(n²). | 3 clusters; silhouette 0.636 (well-separated); ARI 0.714 |

---

### Anomaly Detection

| Method | Purpose | Key metrics | How to interpret | Example result |
|--------|---------|-------------|-----------------|----------------|
| `elliptic_envelope` | Robust covariance outlier detector; assumes elliptical distribution. | AUC, Macro-F1 | AUC 1.000 is achievable on the built-in anomaly dataset; lower Macro-F1 is due to threshold sensitivity. | AUC 1.000 (excellent); Macro-F1 93.8% |
| `isolation_forest` | Random-partitioning anomaly detector; efficient on high-dimensional data. | AUC, Macro-F1 | AUC 1.000 on the synthetic anomaly dataset. Use contamination parameter to tune the threshold. | AUC 1.000 (excellent); Macro-F1 94.9% |
| `local_outlier_factor` | Local density deviation anomaly detector. | AUC, Macro-F1 | Sensitive to neighborhood size. Best for local density anomalies (not global outliers). | (in `list-methods`) |
| `one_class_svm` | Kernel boundary for the support of normal data. | AUC, Macro-F1 | Kernel and nu require tuning; AUC can be high with low Macro-F1 due to threshold. | AUC 1.000 (excellent); Macro-F1 86.0% |

---

### Dimensionality Reduction

| Method | Purpose | Key metrics | How to interpret | Example result |
|--------|---------|-------------|-----------------|----------------|
| `factor_analysis` | Latent factor model explaining observed covariance. | components | Variance explained is not reported because FA models noise explicitly. Inspect factor loading artifact. | Projected to 2 components |
| `fast_ica` | Separates statistically independent components. | components | Use on signals with non-Gaussian marginals (e.g., audio, sensor data). | Projected to 2 components |
| `isomap` | Geodesic distance-preserving nonlinear reduction. | components | Preserves global structure better than t-SNE. Good for data on smooth manifolds. | Projected to 2 components |
| `kernel_pca` | Nonlinear PCA via kernel trick. | components | `rbf` kernel can separate curved clusters invisible to linear PCA. Compare scatter plot with PCA. | Projected to 2 components |
| `mds` | Multidimensional scaling: preserves pairwise distances. | components | Metric MDS minimizes stress over distances. Inspect stress value in artifact. | (in `list-methods`) |
| `nmf` | Nonnegative matrix factorization; additive parts representation. | components | Components are strictly non-negative — interpretable as "topics" or "parts". | Projected to 2 components |
| `pca` | Linear variance-maximizing projection. | components, variance explained | 21.6% variance in 2 components is low for `digits` (10 classes) — expected at 2 components; add more to retain more variance. | 2 components explain 21.6% of variance |
| `spectral_embedding` | Graph Laplacian embedding for nonlinear structure. | components | Shows local connectivity structure. Compare scatter plot with t-SNE. | Projected to 2 components |
| `truncated_svd` | Low-rank SVD; works on sparse matrices. | components, variance explained | Same variance % as PCA on dense data; preferred for sparse/text data (no centering needed). | 2 components explain 21.6% of variance |
| `tsne` | Neighborhood-preserving 2D visualization. | components | Not a metric — inspect the scatter plot. High perplexity = global structure; low perplexity = local. | Projected to 2 components |
| `umap` | Fast nonlinear embedding preserving local and global topology. | components | Faster than t-SNE, better global structure. Inspect scatter plot for cluster separation. | Projected to 2 components |

---

### Feature Selection

| Method | Purpose | Key metrics | How to interpret | Example result |
|--------|---------|-------------|-----------------|----------------|
| `feature_rfe_logistic` | Recursive elimination ranked by logistic regression coefficients. | selected features, downstream macro-F1 | Higher downstream F1 with fewer features = better. Compare selected count with `feature_rfecv_logistic`. | Selected 12; downstream macro-F1 97.0% |
| `feature_rfecv_logistic` | RFE with CV to automatically choose the number of features. | selected features, downstream macro-F1 | More features selected than fixed-k RFE because CV optimizes the count. | Selected 26; downstream macro-F1 98.5% |
| `feature_select_from_l1_linear_svc` | Embedded selection via sparse L1 SVM coefficients. | selected features, downstream macro-F1 | Embedded method: selection and model training are joint. | Selected 12; downstream macro-F1 99.2% |
| `feature_select_k_best_f_classif` | Univariate ANOVA F-test filter. | selected features, downstream macro-F1 | Filters features independently of the model; fast but ignores interactions. | Selected 12; downstream macro-F1 94.7% |
| `feature_select_k_best_mutual_info` | Univariate mutual-information filter. | selected features, downstream macro-F1 | Captures nonlinear feature-target dependence unlike F-test. | Selected 12; downstream macro-F1 94.7% |

---

### Time Series

| Method | Purpose | Key metrics | How to interpret | Example result |
|--------|---------|-------------|-----------------|----------------|
| `statsmodels_autoreg` | Autoregressive model using lagged target values. | forecast steps, MAE, RMSE, MAPE | Baseline for time series. MAPE 1.5% on airline data = solid. Compare with exponential smoothing. | 24-step forecast; MAE 6.774; RMSE 8.129; MAPE 1.5% |
| `statsmodels_exponential_smoothing` | Holt-Winters: trend + seasonal smoothing. | forecast steps, MAE, RMSE, MAPE | MAPE 1.1% = excellent. Best for data with clear additive/multiplicative seasonality. | 24-step forecast; MAE 4.977; RMSE 6.032; MAPE 1.1% |
| `statsmodels_sarimax` | State-space ARIMA with seasonal terms. | forecast steps, MAE, RMSE, MAPE | Most flexible — handles trend, seasonality, and exogenous variables. | 24-step forecast; MAE 4.995; RMSE 6.109; MAPE 1.1% |

---

### Survival Analysis

| Method | Purpose | Key metrics | How to interpret | Example result |
|--------|---------|-------------|-----------------|----------------|
| `lifelines_cox_model` | Cox proportional hazards; estimates covariate effect on survival. | C-index (test + train), event fraction | C-index 0.663 = moderate discrimination; 0.731+ = good. Large train-test gap = overfitting. | Cox C-index 0.663 test (moderate), 0.678 train |
| `lifelines_kaplan_meier` | Non-parametric survival curve by group; log-rank test. | log-rank p-value, event fraction | p < 0.05 = groups have significantly different survival. Inspect KM curve artifact. | Log-rank p=0.0000 (significant); event fraction 28.4% |

---

### Optimization

| Method | Purpose | Key metrics | How to interpret | Example result |
|--------|---------|-------------|-----------------|----------------|
| `scipy_linear_programming` | Product-mix LP: maximize profit subject to resource constraints. | optimal profit, resource utilization | Resource utilization near 100% = binding constraints. Check shadow prices in artifact for bottlenecks. | Optimal profit 43 264; avg resource utilization 96.2% |
| `scipy_nonlinear_optimization` | Marketing-mix budget allocation with diminishing-return response curves. | optimized sales, lift % | Positive lift confirms the optimizer beat the current allocation. Re-run with more iterations for higher lift. | Optimized sales 37 031 (+3.8% lift vs current budget) |
| `cvxpy_portfolio_optimization` | Mean-variance portfolio with return, risk, and weight constraints. | expected return, volatility, Sharpe-like | Higher Sharpe-like = better risk-adjusted allocation. Compare with `cvxpy_quadratic_programming`. | Expected return 6.4%; volatility 8.1%; Sharpe-like 0.78 |
| `cvxpy_quadratic_programming` | Pure risk minimization with target-return constraint. | expected return, volatility, Sharpe-like | Lower volatility than portfolio optimization at cost of lower return is the expected trade-off. | Expected return 7.2%; volatility 9.9%; Sharpe-like 0.73 |

---

### Incremental Learning

| Method | Purpose | Key metrics | How to interpret | Example result |
|--------|---------|-------------|-----------------|----------------|
| `river_logistic_regression` | Online logistic regression evaluated prequentially on a streaming dataset. | prequential accuracy, F1, samples | Early samples drag F1 down — accuracy stabilizes after ~200 samples. Low F1 = class imbalance in stream. | Prequential accuracy 83.0%; F1 30.1%; samples 1200 |

---

## Common Diagnostic Patterns

| Symptom | Likely cause | What to do |
|---------|-------------|------------|
| High AUC, low Macro-F1 | Imbalanced classes; threshold is biased toward majority | Try `class_weight="balanced"`, SMOTE, or balanced random forest |
| R² << 0 | Target or features are on incompatible scales, or kernel mismatch | Check pipeline scaling, try different kernel, or tune regularization |
| Silhouette < 0.2 | Wrong k, or cluster structure doesn't match algorithm assumptions | Try HDBSCAN (no k needed), or explore range of k values |
| RMSE >> MAE | A few samples have very large errors | Use Huber or RANSAC regression, or remove/investigate outliers |
| MAPE undefined or very high | Target contains zeros or near-zeros | Use RMSE/MAE instead; Poisson/Tweedie GLMs handle count targets better |
| C-index ≈ 0.5 | Covariates carry no survival signal | Inspect coefficients; check for correlated or leaky features |
| ARI ≈ 0 with good silhouette | True labels don't match the geometric structure found | The clustering may be valid but represents a different grouping than the labels |
| Prequential F1 << accuracy | Streaming class imbalance or slow model warm-up | Increase initial training window or use class-weighted online loss |
