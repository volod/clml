"""
Named constants for the clml codebase.

Magic numbers must never appear directly in source code; use the names defined here.
Each constant is documented with its purpose and rationale.
"""

# ---------------------------------------------------------------------------
# Output directory structure
# ---------------------------------------------------------------------------

TASK_GROUPS: dict[str, str] = {
    "classification": "classification",
    "imbalanced_classification": "classification",
    "incremental_classification": "classification",
    "statsmodels_classification": "classification",
    "categorical_encoding": "classification",
    "feature_selection": "classification",
    "regression": "regression",
    "statsmodels_regression": "regression",
    "regression_1d": "regression",
    "timeseries": "timeseries",
    "clustering": "clustering",
    "anomaly": "anomaly",
    "density": "density",
    "dimensionality": "dimensionality",
    "linear_programming": "optimization",
    "nonlinear_optimization": "optimization",
    "cvxpy_portfolio": "optimization",
    "cvxpy_quadratic": "optimization",
    "survival_cox": "survival",
    "survival_kaplan_meier": "survival",
}
"""Map each task type to a top-level group folder under DATA_DIR/runs/."""

# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

TEST_SIZE = 0.25
"""Hold-out fraction for supervised and statistical task runners."""

ANOMALY_NORMAL_TEST_SIZE = 0.30
"""Fraction of normal-class samples held out for the anomaly detection test set."""

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

CV_FOLDS = 3
"""K-fold splits for Optuna cross-validation and stacking / voting base-learner CV."""

# ---------------------------------------------------------------------------
# Metric quality thresholds  (used by interpret_metrics)
# ---------------------------------------------------------------------------

AUC_EXCELLENT = 0.90
"""ROC-AUC at or above this value is labelled 'excellent'."""

AUC_GOOD = 0.75
"""ROC-AUC at or above this value (but below AUC_EXCELLENT) is labelled 'good'."""

R2_STRONG = 0.80
"""R² at or above this value is labelled a 'strong' fit."""

R2_MODERATE = 0.50
"""R² at or above this value (but below R2_STRONG) is labelled a 'moderate' fit."""

SILHOUETTE_WELL_SEPARATED = 0.50
"""Silhouette score above this value is labelled 'well-separated'."""

SILHOUETTE_OVERLAPPING = 0.20
"""Silhouette score above this value (but below SILHOUETTE_WELL_SEPARATED) is labelled
'overlapping'."""

CONCORDANCE_GOOD = 0.70
"""Cox C-index at or above this value is labelled 'good'."""

CONCORDANCE_MODERATE = 0.60
"""Cox C-index at or above this value (but below CONCORDANCE_GOOD) is labelled 'moderate'."""

LOGRANK_P_THRESHOLD = 0.05
"""p-value below which a log-rank test result is declared statistically significant."""

# ---------------------------------------------------------------------------
# Boosting library defaults  (XGBoost / LightGBM / CatBoost / ExtraTrees / RF)
# ---------------------------------------------------------------------------

GBM_N_ESTIMATORS = 120
"""Default number of boosting rounds / trees for all gradient-boosted and forest methods."""

GBM_LEARNING_RATE = 0.08
"""Default learning rate for XGBoost, LightGBM, and CatBoost."""

GBM_MAX_DEPTH = 3
"""Default maximum tree depth for XGBoost and LightGBM."""

GBM_SUBSAMPLE = 0.9
"""Row-sample fraction for XGBoost and LightGBM to reduce overfitting."""

# ---------------------------------------------------------------------------
# Interpretable boosting (EBM)
# ---------------------------------------------------------------------------

EBM_INTERACTIONS = 5
"""Number of pairwise interaction terms for ExplainableBoostingClassifier/Regressor."""

EBM_MAX_ROUNDS = 400
"""Maximum boosting rounds for EBM."""

NGBOOST_N_ESTIMATORS = 180
"""Number of boosting stages for NGBoost probabilistic regression."""

NGBOOST_LEARNING_RATE = 0.05
"""Learning rate for NGBoost; kept conservative for stable distribution fitting."""

# ---------------------------------------------------------------------------
# Ensemble composition — base-learner sizes
# (small so stacking / voting trains quickly inside CV folds)
# ---------------------------------------------------------------------------

ENSEMBLE_BASE_N_ESTIMATORS = 50
"""Trees per RandomForest / GradientBoosting base learner inside stacking and voting ensembles."""

ENSEMBLE_BASE_MAX_ITER = 500
"""Max solver iterations for LogisticRegression base / meta-learners in stacking and voting."""

# ---------------------------------------------------------------------------
# Specific model defaults
# ---------------------------------------------------------------------------

LOGISTIC_MAX_ITER = 1000
"""Max solver iterations for standalone LogisticRegression and categorical-encoding pipelines."""

ELASTIC_NET_MAX_ITER = 5_000
"""Max iterations for ElasticNet and Lasso coordinate descent."""

NMF_MAX_ITER = 500
"""Max iterations for Non-negative Matrix Factorization."""

GPR_N_RESTARTS = 3
"""Kernel hyper-parameter optimizer restarts for GaussianProcessRegressor."""

MLP_CLF_HIDDEN_LAYERS = (32,)
"""Hidden layer sizes for MLPClassifier."""

MLP_CLF_MAX_ITER = 2_000
"""Maximum training epochs for MLPClassifier."""

MLP_REG_HIDDEN_LAYERS = (64, 32)
"""Hidden layer sizes for MLPRegressor (wider than classifier to handle regression scale)."""

MLP_REG_MAX_ITER = 2_000
"""Maximum training epochs for MLPRegressor."""

QDA_REG_PARAM = 0.1
"""Regularisation strength for QuadraticDiscriminantAnalysis."""

SVR_DEFAULT_C = 10.0
"""Default regularisation for SVR; higher than sklearn's C=1 for better untuned fit."""

HUBER_DEFAULT_EPSILON = 1.35
"""Threshold between quadratic and linear loss in HuberRegressor."""

TWEEDIE_DEFAULT_POWER = 1
"""Tweedie family power for TweedieRegressor (1 = Poisson)."""

TWEEDIE_DEFAULT_ALPHA = 0.5
"""Default L2 regularisation for TweedieRegressor."""

TWEEDIE_MAX_ITER = 1_000
"""Maximum solver iterations for TweedieRegressor."""

PASSIVE_AGGRESSIVE_MAX_ITER = 1_000
"""Maximum training passes for PassiveAggressiveClassifier."""

# ---------------------------------------------------------------------------
# Clustering defaults
# ---------------------------------------------------------------------------

CLUSTER_DEFAULT_K = 4
"""Default number of clusters for K-Means, BIRCH, Agglomerative, Spectral, GMM variants."""

HDBSCAN_MIN_CLUSTER_SIZE = 15
"""Minimum cluster size for HDBSCAN."""

DBSCAN_EPS = 0.8
"""Neighbourhood radius for DBSCAN."""

DBSCAN_MIN_SAMPLES = 6
"""Minimum neighbours required to form a core point in DBSCAN."""

OPTICS_MIN_SAMPLES = 8
"""Minimum samples parameter for OPTICS."""

KDE_DEFAULT_BANDWIDTH = 1.0
"""Starting bandwidth for KernelDensity estimation."""

# ---------------------------------------------------------------------------
# Dimensionality reduction defaults
# ---------------------------------------------------------------------------

DIM_DEFAULT_COMPONENTS = 2
"""Default output dimensions for PCA, ICA, NMF, KernelPCA, FactorAnalysis, etc."""

ISOMAP_N_NEIGHBORS = 10
"""Number of neighbours used to build the geodesic graph in Isomap."""

# t-SNE
TSNE_PERPLEXITY = 30
"""Neighbourhood size for t-SNE joint probability (perplexity ≈ effective number of neighbours)."""

TSNE_PERPLEXITY_MIN = 5
"""Lower bound for t-SNE perplexity in hyperparameter search."""

TSNE_PERPLEXITY_MAX = 50
"""Upper bound for t-SNE perplexity in hyperparameter search."""

TSNE_MAX_ITER = 300
"""Maximum gradient-descent iterations for t-SNE (300 is sufficient for exploratory embeddings)."""

# UMAP
UMAP_N_NEIGHBORS = 15
"""Default UMAP local-neighbourhood size."""

UMAP_N_NEIGHBORS_MIN = 5
"""Lower bound for UMAP n_neighbors in hyperparameter search."""

UMAP_N_NEIGHBORS_MAX = 40
"""Upper bound for UMAP n_neighbors in hyperparameter search."""

UMAP_MIN_DIST = 0.1
"""Default UMAP minimum embedding distance; lower values form tighter clusters."""

UMAP_MIN_DIST_MIN = 0.0
"""Lower bound for UMAP min_dist in hyperparameter search."""

UMAP_MIN_DIST_MAX = 0.8
"""Upper bound for UMAP min_dist in hyperparameter search."""

# Locally Linear Embedding
LLE_N_NEIGHBORS = 10
"""Default neighbourhood size for Locally Linear Embedding geometry reconstruction."""

LLE_N_NEIGHBORS_MIN = 5
"""Lower bound for LLE n_neighbors in hyperparameter search."""

LLE_N_NEIGHBORS_MAX = 30
"""Upper bound for LLE n_neighbors in hyperparameter search."""

# Multidimensional Scaling
MDS_MAX_ITER = 300
"""Maximum SMACOF stress-minimisation iterations for MDS."""

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

ANOMALY_CONTAMINATION = 0.06
"""Assumed fraction of anomalies used for IsolationForest, OneClassSVM, EllipticEnvelope."""

ONE_CLASS_SVM_NU = 0.06
"""Upper bound on the fraction of training errors for OneClassSVM (≈ contamination rate)."""

# ---------------------------------------------------------------------------
# NuSVC
# ---------------------------------------------------------------------------

NU_SVC_NU = 0.5
"""Default nu for NuSVC: upper bound on the fraction of margin errors and support vectors."""

NU_MIN = 0.05
"""Lower bound for nu in NuSVC hyperparameter search."""

NU_MAX = 0.50
"""Upper bound for nu in NuSVC hyperparameter search (keeps the problem feasible)."""

# ---------------------------------------------------------------------------
# LinearSVC
# ---------------------------------------------------------------------------

LINEAR_SVC_MAX_ITER = 2_000
"""Max dual-coordinate iterations for LinearSVC; higher than default for reliable convergence."""

# ---------------------------------------------------------------------------
# QuantileRegressor
# ---------------------------------------------------------------------------

QUANTILE_DEFAULT = 0.5
"""Default quantile for QuantileRegressor — 0.5 produces median (L1) regression."""

QUANTILE_MIN = 0.1
"""Lower bound for the quantile hyperparameter in search."""

QUANTILE_MAX = 0.9
"""Upper bound for the quantile hyperparameter in search."""

# ---------------------------------------------------------------------------
# Imbalanced / categorical encoding
# ---------------------------------------------------------------------------

BALANCED_RF_N_ESTIMATORS = 120
"""Number of trees in BalancedRandomForestClassifier."""

TARGET_ENCODER_SMOOTHING = 10
"""Smoothing factor for TargetEncoder (higher = more shrinkage towards the global mean)."""

LOO_ENCODER_SIGMA = 0.05
"""Gaussian noise standard deviation added by LeaveOneOutEncoder to prevent target leakage."""

HASHING_ENCODER_COMPONENTS = 12
"""Number of hash buckets (output features) for HashingEncoder."""

# ---------------------------------------------------------------------------
# Hyperparameter search bounds
# ---------------------------------------------------------------------------

# Regularisation alpha (Ridge, Lasso, KernelRidge)
ALPHA_MIN = 1e-4
ALPHA_MAX = 10.0

# ElasticNet — tighter upper bound than general ALPHA_MAX because the L1 component
# already provides strong sparsity; very large alpha collapses all coefficients.
ELASTIC_NET_ALPHA_MAX = 2.0

# ElasticNet L1 ratio
L1_RATIO_MIN = 0.05
L1_RATIO_MAX = 0.95

# Huber regression
HUBER_EPSILON_MIN = 1.1
HUBER_EPSILON_MAX = 2.5
HUBER_ALPHA_MIN = 1e-4
HUBER_ALPHA_MAX = 1.0

# Tweedie GLM
TWEEDIE_POWER_MIN = 0.0
TWEEDIE_POWER_MAX = 3.0
TWEEDIE_ALPHA_MIN = 0.01
TWEEDIE_ALPHA_MAX = 10.0

# Decision tree
TREE_MAX_DEPTH_MIN = 2
TREE_MAX_DEPTH_MAX = 12
TREE_MIN_LEAF_MIN = 1
TREE_MIN_LEAF_MAX = 12

# Random forest / extra trees
FOREST_N_ESTIMATORS_MIN = 50
FOREST_N_ESTIMATORS_MAX = 180
FOREST_MAX_DEPTH_MAX = 14
FOREST_MIN_LEAF_MAX = 8

# Gradient boosting (sklearn GBT)
GB_N_ESTIMATORS_MIN = 40
GB_N_ESTIMATORS_MAX = 180
GB_LEARNING_RATE_MIN = 0.02
GB_LEARNING_RATE_MAX = 0.25
GB_MAX_DEPTH_MIN = 2
GB_MAX_DEPTH_MAX = 6

# HistGradientBoosting
HGBT_MAX_ITER_MIN = 40
HGBT_MAX_ITER_MAX = 180
HGBT_MAX_LEAF_NODES_MIN = 8
HGBT_MAX_LEAF_NODES_MAX = 48

# K-Nearest Neighbors
KNN_N_NEIGHBORS_MIN = 3
KNN_N_NEIGHBORS_MAX = 25

# SVM / SVR
SVM_C_MIN = 1e-2
SVM_C_MAX = 100.0

# PassiveAggressive C
PA_C_MIN = 1e-3
PA_C_MAX = 10.0

# Logistic regression C
LR_C_MIN = 1e-3
LR_C_MAX = 100.0

# SGD alpha
SGD_ALPHA_MIN = 1e-5
SGD_ALPHA_MAX = 1e-1

# KernelRidge gamma
KERNEL_RIDGE_GAMMA_MIN = 1e-4
KERNEL_RIDGE_GAMMA_MAX = 1.0

# Cluster k search
CLUSTER_K_MIN = 2
CLUSTER_K_MAX = 8

# Components search
DIM_COMPONENTS_MIN = 2
DIM_COMPONENTS_MAX = 12

# PLS components
PLS_COMPONENTS_MIN = 1
PLS_COMPONENTS_MAX = 8

# KernelDensity bandwidth
KDE_BANDWIDTH_MIN = 0.2
KDE_BANDWIDTH_MAX = 2.0

# ---------------------------------------------------------------------------
# Statistical modeling (statsmodels)
# ---------------------------------------------------------------------------

SM_LOGIT_MAX_ITER = 250
"""Maximum Newton-Raphson iterations for statsmodels Logit."""

SM_HIGH_CARDINALITY_THRESHOLD = 20
"""Drop categorical columns with more unique values than this before dummy-encoding
to avoid near-singular Hessians from date / ID columns."""

# ---------------------------------------------------------------------------
# Survival analysis
# ---------------------------------------------------------------------------

COX_PENALIZER = 0.05
"""L2 regularisation strength (penalizer) for CoxPHFitter."""

SURVIVAL_EVAL_MONTH = 12
"""Calendar month at which Kaplan-Meier survival probability is extracted and reported."""

KM_PLOT_DPI = 140
"""Resolution (dots per inch) for the Kaplan-Meier survival curve output image."""

KM_PLOT_FIGSIZE = (8, 6)
"""Figure dimensions (width, height) in inches for the Kaplan-Meier plot."""

# ---------------------------------------------------------------------------
# Portfolio optimisation (CVXPY)
# ---------------------------------------------------------------------------

PORTFOLIO_CROSS_SECTOR_CORR = 0.22
"""Default pairwise correlation between assets from different sectors."""

PORTFOLIO_SAME_SECTOR_CORR = 0.58
"""Default pairwise correlation between assets within the same sector."""

PORTFOLIO_RISK_AVERSION = 3.0
"""Risk-aversion coefficient multiplied by the variance term in the mean-variance objective."""

PORTFOLIO_MIN_LIQUIDITY = 0.84
"""Minimum required weighted-average liquidity score for the optimal portfolio."""

PORTFOLIO_TARGET_RETURN = 0.072
"""Minimum required expected return for the minimum-variance (quadratic) portfolio."""

# ---------------------------------------------------------------------------
# Marketing mix optimisation (SciPy nonlinear)
# ---------------------------------------------------------------------------

MARKETING_BUDGET = 550_000.0
"""Total marketing spend budget to allocate across channels."""

SLSQP_MAX_ITER = 400
"""Maximum iterations for the SLSQP nonlinear solver."""

SLSQP_FTOL = 1e-8
"""Function-value convergence tolerance for SLSQP."""

# ---------------------------------------------------------------------------
# Production planning — linear programming (SciPy linprog)
# ---------------------------------------------------------------------------

LP_LABOR_CAPACITY = 1_300.0
"""Available labour-hours per planning period."""

LP_MACHINE_CAPACITY = 820.0
"""Available machine-hours per planning period."""

LP_MATERIAL_CAPACITY = 2_800.0
"""Available raw-material (kg) per planning period."""

LP_BINDING_SLACK_TOL = 1e-6
"""A constraint is treated as binding when its slack falls below this tolerance."""

# ---------------------------------------------------------------------------
# Synthetic dataset generation  (catalog.py)
# ---------------------------------------------------------------------------

CATALOG_SEED = 42
"""Shared random seed for all catalog dataset builders (ensures reproducibility)."""

# --- moons ---
MOONS_N_SAMPLES = 900
MOONS_NOISE = 0.25

# --- blobs ---
BLOBS_N_SAMPLES = 900
BLOBS_N_FEATURES = 6
BLOBS_CENTERS = 4
BLOBS_CLUSTER_STD = [0.8, 1.2, 1.0, 1.5]

# --- anomaly ---
ANOMALY_N_NORMAL = 700
ANOMALY_N_OUTLIERS = 45
ANOMALY_N_FEATURES = 5
ANOMALY_OUTLIER_MEAN = 5.0
ANOMALY_OUTLIER_STD = 0.8

# --- streaming churn ---
STREAMING_CHURN_N_ROWS = 1_200
STREAMING_TREND_SCALE = 0.8
STREAMING_NOISE_STD = 0.35

# --- airline-style time series ---
AIRLINE_N_PERIODS = 144
AIRLINE_SEASONAL_PERIODS = 12
AIRLINE_BASE_PASSENGERS = 120.0
AIRLINE_MONTHLY_TREND = 2.4
AIRLINE_SEASONAL_AMPLITUDE = 22.0
AIRLINE_NOISE_STD = 5.0
TIMESERIES_HOLDOUT = 24

# --- feature selection ---
FEATURE_SELECTION_K = 12
FEATURE_SELECTION_MIN_FEATURES = 5
FEATURE_SELECTION_L1_C = 0.2
FEATURE_SELECTION_MAX_ITER = 10_000

# --- credit_risk ---
CREDIT_RISK_N_ROWS = 3_500
CREDIT_RISK_AGE_MIN = 21
CREDIT_RISK_AGE_MAX = 73
CREDIT_RISK_DATE_DAYS = 730  # ≈ 2 years of application history
CREDIT_RISK_INCOME_LOG_MEAN = 10.8
CREDIT_RISK_INCOME_LOG_STD = 0.45
CREDIT_RISK_LOAN_LOG_MEAN = 10.1
CREDIT_RISK_LOAN_LOG_STD = 0.6
CREDIT_RISK_DTI_NOISE_MEAN = 0.18
CREDIT_RISK_DTI_NOISE_STD = 0.09
CREDIT_RISK_DTI_MIN = 0.02
CREDIT_RISK_DTI_MAX = 1.6
CREDIT_RISK_CREDIT_HIST_BETA_A = 2.0
CREDIT_RISK_CREDIT_HIST_BETA_B = 4.0
CREDIT_RISK_DELINQ_CLIP_MAX = 3.0
CREDIT_RISK_EMP_LENGTH_MEAN = 6.5
CREDIT_RISK_EMP_LENGTH_STD = 4.0
CREDIT_RISK_EMP_LENGTH_MAX = 32.0
CREDIT_RISK_UTIL_BETA_A = 2.2
CREDIT_RISK_UTIL_BETA_B = 4.5
CREDIT_RISK_UTIL_DTI_WEIGHT = 0.18
# Logistic risk formula coefficients
CREDIT_RISK_INTERCEPT = -2.4
CREDIT_RISK_DTI_COEF = 2.6
CREDIT_RISK_UTIL_COEF = 1.5
CREDIT_RISK_DELINQ_COEF = 0.33
CREDIT_RISK_LOG_INCOME_COEF = 0.13
CREDIT_RISK_CREDIT_HIST_COEF = 0.025
CREDIT_RISK_RENT_COEF = 0.35
CREDIT_RISK_SMALL_BIZ_COEF = 0.45
CREDIT_RISK_MEDICAL_COEF = 0.28
CREDIT_RISK_NOISE_STD = 0.45
# Categorical proportions
CREDIT_RISK_HOME_PROBS = [0.42, 0.44, 0.14]  # rent / mortgage / own
CREDIT_RISK_PURPOSE_PROBS = [0.46, 0.18, 0.12, 0.08, 0.16]
CREDIT_RISK_REGION_PROBS = [0.22, 0.34, 0.25, 0.19]

# --- housing_prices ---
HOUSING_N_ROWS = 3_200
HOUSING_LIVING_AREA_LOG_MEAN = 7.35
HOUSING_LIVING_AREA_LOG_STD = 0.36
HOUSING_LIVING_AREA_PER_BEDROOM = 620.0
HOUSING_BEDROOM_MEAN = 1.2
HOUSING_BEDROOM_STD = 0.9
HOUSING_BEDROOM_MIN = 1
HOUSING_BEDROOM_MAX = 8
HOUSING_BATHROOM_RATIO = 0.55
HOUSING_BATHROOM_MEAN = 0.6
HOUSING_BATHROOM_STD = 0.55
HOUSING_BATHROOM_MIN = 1.0
HOUSING_BATHROOM_MAX = 5.0
HOUSING_AGE_MAX = 105
HOUSING_LOT_LOG_MEAN = 8.55
HOUSING_LOT_LOG_STD = 0.55
HOUSING_DIST_SHAPE = 2.1
HOUSING_DIST_SCALE = 4.0
HOUSING_SCHOOL_MEAN = 6.5
HOUSING_SCHOOL_STD = 1.6
HOUSING_SCHOOL_DIST_WEIGHT = 0.05
HOUSING_SCHOOL_MIN = 1.0
HOUSING_SCHOOL_MAX = 10.0
HOUSING_DATE_DAYS = 1_095  # ≈ 3 years of listing history
HOUSING_NEIGHBORHOOD_PROBS = [0.18, 0.47, 0.17, 0.08, 0.10]
HOUSING_CONDITION_PROBS = [0.12, 0.46, 0.32, 0.10]
# Neighborhood price effects (USD)
HOUSING_CENTRAL_EFFECT = 95_000.0
HOUSING_SUBURBAN_EFFECT = 45_000.0
HOUSING_RURAL_EFFECT = -35_000.0
HOUSING_WATERFRONT_EFFECT = 185_000.0
HOUSING_INDUSTRIAL_EFFECT = -65_000.0
# Condition price effects (USD)
HOUSING_NEEDS_WORK_EFFECT = -55_000.0
HOUSING_GOOD_EFFECT = 42_000.0
HOUSING_RENOVATED_EFFECT = 98_000.0
# Price formula coefficients
HOUSING_BASE_PRICE = 65_000.0
HOUSING_LIVING_AREA_COEF = 145.0
HOUSING_LOT_AREA_COEF = 4.5
HOUSING_BATHROOM_COEF = 22_000.0
HOUSING_SCHOOL_COEF = 18_000.0
HOUSING_AGE_DEPRECIATION = 950.0
HOUSING_DISTANCE_COEF = 5_200.0
HOUSING_PRICE_NOISE_STD = 45_000.0
HOUSING_LUXURY_QUANTILE = 0.75
HOUSING_LUXURY_MULTIPLIER = 1.28
HOUSING_OUTLIER_COUNT = 35
HOUSING_OUTLIER_MULT_MIN = 1.35
HOUSING_OUTLIER_MULT_MAX = 1.90

# --- customer_survival ---
SURVIVAL_N_ROWS = 1_400
SURVIVAL_AGE_MIN = 22
SURVIVAL_AGE_MAX = 78
SURVIVAL_SPEND_LOG_MEAN = 4.2
SURVIVAL_SPEND_LOG_STD = 0.45
SURVIVAL_TENURE_MAX = 72
SURVIVAL_TICKETS_RATE = 1.2
SURVIVAL_TICKETS_SPEND_REF = 90.0  # reference spend for ticket-rate formula
SURVIVAL_TICKETS_SPEND_MAX = 80.0  # clip ceiling in ticket-rate formula
SURVIVAL_CONTRACT_PROBS = [0.52, 0.34, 0.14]  # monthly / annual / two_year
SURVIVAL_SEGMENT_PROBS = [0.58, 0.30, 0.12]  # consumer / small_business / enterprise
# Log-hazard formula coefficients
SURVIVAL_HAZARD_INTERCEPT = -3.0
SURVIVAL_MONTHLY_COEF = 0.65
SURVIVAL_TWO_YEAR_COEF = 0.55
SURVIVAL_TICKETS_COEF = 0.16
SURVIVAL_LOG_SPEND_COEF = 0.12
SURVIVAL_TENURE_COEF = 0.012
SURVIVAL_CONSUMER_COEF = 0.18
SURVIVAL_HAZARD_NOISE_STD = 0.25
SURVIVAL_CENSOR_MIN = 8.0
SURVIVAL_CENSOR_MAX = 42.0
SURVIVAL_DURATION_MIN = 1.0
SURVIVAL_DURATION_MAX = 60.0

# --- monotone_1d ---
MONOTONE_N_SAMPLES = 600

MONOTONE_X_MAX = 10.0
MONOTONE_LINEAR_COEF = 2.0
MONOTONE_LOG_COEF = 3.0
MONOTONE_NOISE_STD = 1.5

# --- credit risk extra generation parameters ---
CREDIT_RISK_DATE_START = "2023-01-01"
"""ISO date string for the earliest synthetic credit-risk application date."""

CREDIT_RISK_MIN_CREDIT_AGE = 18
"""Age (years) at which a borrower's credit history starts accumulating."""

CREDIT_RISK_CREDIT_HIST_MAX = 45
"""Upper clip for credit history years after beta sampling."""

CREDIT_RISK_DELINQ_POISSON_SCALE = 1.8
"""Multiplier applied to debt_to_income to produce the Poisson rate for delinquencies."""

CREDIT_RISK_DELINQ_POISSON_MIN = 0.05
"""Minimum Poisson rate for delinquencies (floor clip before sampling)."""

# --- housing prices extra generation parameters ---
HOUSING_DATE_START = "2022-01-01"
"""ISO date string for the earliest synthetic housing listing date."""

# --- airline passengers date anchor ---
AIRLINE_DATE_START = "2010-01-01"
"""ISO date string for the first month in the synthetic airline-passengers time series."""

# ---------------------------------------------------------------------------
# Run artifact file names
# (All output files written by task runners use these names.)
# ---------------------------------------------------------------------------

RUN_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
"""strftime format for the run directory timestamp (ISO-8601 compact UTC)."""

# Core run metadata
ARTIFACT_RUN_JSON = "run.json"
"""Serialised RunResult written at the end of every run."""

ARTIFACT_THIRD_PARTY_JSON = "third_party_comparisons.json"
"""Side-by-side metrics from XGBoost / LightGBM / CatBoost comparison runs."""

ARTIFACT_RUN_ALL_LEARNING_MD = "learning_insights.md"
"""Human-readable learning insights generated from a run-all summary."""

ARTIFACT_RUN_ALL_LEARNING_CSV = "learning_insights.csv"
"""Tabular learning signals generated from a run-all summary."""

ARTIFACT_MODEL_SKOPS = "model.skops"
"""Primary model serialisation (skops format)."""

ARTIFACT_MODEL_JOBLIB = "model.joblib"
"""Fallback model serialisation used when skops cannot handle the estimator type."""

# Supervised learning
ARTIFACT_PREDICTIONS_CSV = "predictions.csv"
"""Actual vs predicted values for classification and regression tasks."""

ARTIFACT_DISTRIBUTION_PREDICTIONS_CSV = "distribution_predictions.csv"
"""Predicted distribution parameters and quantiles for probabilistic regressors (e.g. NGBoost)."""

ARTIFACT_CLASSIFICATION_REPORT_JSON = "classification_report.json"
"""Per-class precision, recall, and F1 from sklearn classification_report."""

ARTIFACT_CONFUSION_MATRIX_PNG = "confusion_matrix.png"
"""Confusion matrix heatmap for classification tasks."""

ARTIFACT_PREDICTION_SCATTER_PNG = "prediction_scatter.png"
"""Actual vs predicted scatter plot for regression tasks."""

# Clustering / dimensionality / anomaly / density
ARTIFACT_CLUSTERS_CSV = "clusters.csv"
"""Cluster label per sample produced by clustering algorithms."""

ARTIFACT_CLUSTERS_PNG = "clusters.png"
"""2-D PCA projection coloured by cluster label."""

ARTIFACT_EMBEDDING_CSV = "embedding.csv"
"""Low-dimensional embedding coordinates from dimensionality-reduction methods."""

ARTIFACT_EMBEDDING_PNG = "embedding.png"
"""2-D scatter of the embedding coloured by true labels (if available)."""

ARTIFACT_ANOMALY_SCORES_CSV = "anomaly_scores.csv"
"""Actual labels, predictions, and raw anomaly scores for anomaly-detection methods."""

ARTIFACT_ANOMALIES_PNG = "anomalies.png"
"""2-D projection with anomalous samples highlighted."""

ARTIFACT_DENSITY_SCORES_CSV = "density_scores.csv"
"""Per-sample log-likelihood from kernel density or GMM density estimation."""

ARTIFACT_DENSITY_PNG = "density.png"
"""2-D projection coloured by log-likelihood value."""

# Time series
ARTIFACT_FORECAST_CSV = "forecast.csv"
"""Actual vs forecast values for the hold-out horizon."""

ARTIFACT_SERIES_CSV = "series.csv"
"""Full observed training series used for fitting."""

ARTIFACT_FORECAST_PNG = "forecast.png"
"""Observed + forecast line chart."""

# Optimisation / portfolio
ARTIFACT_ALLOCATION_CSV = "allocation.csv"
"""Optimised allocation (portfolio weights or production units) per asset / product."""

ARTIFACT_COVARIANCE_CSV = "covariance.csv"
"""Asset return covariance matrix used in portfolio optimisation."""

ARTIFACT_RESOURCE_USAGE_CSV = "resource_usage.csv"
"""Capacity, used, slack, and utilisation for each resource in the LP model."""

# Survival analysis
ARTIFACT_COX_COEFFICIENTS_CSV = "cox_coefficients.csv"
"""Cox model coefficient table with standard errors and p-values."""

ARTIFACT_SURVIVAL_PREDICTIONS_CSV = "survival_predictions.csv"
"""Test-set rows with appended partial-hazard risk scores."""

# Feature selection
ARTIFACT_SELECTED_FEATURES_CSV = "selected_features.csv"
"""Names of the features retained after the selection step."""

ARTIFACT_FEATURE_SELECTION_DIAGNOSTICS_CSV = "feature_selection_diagnostics.csv"
"""Per-feature diagnostics: selected flag, selector score, and ranking where available."""

# Exploration and feature-engineering reports
ARTIFACT_EXPLORATION_JSON = "exploration.json"
"""Serialised ExplorationReport with dataset profile, missing fractions, and suggestions."""

ARTIFACT_NUMERIC_SUMMARY_CSV = "numeric_summary.csv"
"""pandas describe() transposed for numeric features."""

ARTIFACT_CORRELATION_PNG = "correlation.png"
"""Pearson correlation heatmap of numeric features."""

ARTIFACT_FEATURES_PNG = "features.png"
"""Histogram grid of feature distributions."""

ARTIFACT_FE_RECOMMENDATIONS_JSON = "recommendations.json"
"""Feature-engineering advice JSON written by the feature-engineering reporter."""

ARTIFACT_FE_CANDIDATE_RULES_JSON = "candidate_rules.json"
"""Candidate feature-engineering rule list derived from the dataset profile."""

# Dataset cache
ARTIFACT_DATASET_DATA_CSV = "data.csv"
"""Cached dataset frame written to <DATA_DIR>/datasets/<name>/data.csv."""

ARTIFACT_DATASET_METADATA_JSON = "metadata.json"
"""Cached DatasetInfo JSON written alongside data.csv."""

# Suggest-methods output
ARTIFACT_METHOD_ADVICE_JSON = "method_advice.json"
"""Serialised DatasetMethodAdvice written by the suggest-methods command."""

# ---------------------------------------------------------------------------
# Plot configuration
# (All matplotlib / seaborn output uses these values for consistency.)
# ---------------------------------------------------------------------------

PLOT_DPI = 140
"""Output resolution (dots per inch) for all saved plot images."""

PLOT_FIGSIZE_HEATMAP = (10, 8)
"""Figure dimensions (width, height) in inches for correlation heatmaps."""

PLOT_FIGSIZE_DISTRIBUTIONS = (12, 8)
"""Figure dimensions for the feature-distribution histogram grid."""

PLOT_DISTRIBUTION_MAX_COLUMNS = 8
"""Maximum number of feature columns shown in the distribution histogram grid."""

PLOT_DISTRIBUTION_BINS = 30
"""Number of histogram bins for feature-distribution plots."""

PLOT_FIGSIZE_SCATTER = (7, 6)
"""Figure dimensions for regression actual-vs-predicted scatter plots."""

PLOT_SCATTER_MARKER_SIZE = 35
"""Scatter-plot marker size (s parameter) for regression predictions."""

PLOT_SCATTER_LINE_WIDTH = 1
"""Line width of the identity (y=x) reference line on regression scatter plots."""

PLOT_FIGSIZE_TIMESERIES = (10, 5)
"""Figure dimensions for time-series forecast plots."""

PLOT_TIMESERIES_OBSERVED_LINE_WIDTH = 1.8
"""Line width for the observed series in forecast plots."""

PLOT_TIMESERIES_FORECAST_LINE_WIDTH = 2.0
"""Line width for the forecast series in forecast plots (slightly bolder than observed)."""

PLOT_FIGSIZE_PROJECTION = (7, 6)
"""Figure dimensions for 2-D dimensionality-reduction projection scatter plots."""

PLOT_PROJECTION_MARKER_SIZE = 28
"""Scatter marker size for 2-D projection plots."""

PLOT_PROJECTION_PALETTE = "tab10"
"""Colour palette used for cluster / class labels in projection scatter plots."""

PLOT_FIGSIZE_BARS = (9, 5)
"""Figure dimensions for named bar charts (feature importances, resource utilisation, etc.)."""

PLOT_BARS_LABEL_ROTATION = 25
"""X-axis label rotation angle (degrees) for bar charts."""

PLOT_FIGSIZE_RESPONSE_CURVES = (9, 6)
"""Figure dimensions for marketing response-curve plots."""

PLOT_RESPONSE_GRID_POINTS = 80
"""Number of evenly-spaced spend values used to draw each response curve."""

PLOT_RESPONSE_SCATTER_SIZE = 35
"""Scatter marker size for the optimised spend point on response-curve plots."""

PLOT_FIGSIZE_SKEW = (9, 5)
"""Figure dimensions for the numeric-skew bar chart in feature-engineering reports."""

PLOT_FIGSIZE_MUTUAL_INFO = (9, 6)
"""Figure dimensions for the mutual-information bar chart in feature-engineering reports."""

PLOT_TOP_FEATURES_COUNT = 15
"""Maximum number of features shown in skew and mutual-information plots."""

# ---------------------------------------------------------------------------
# Data quality and inference thresholds
# ---------------------------------------------------------------------------

DATETIME_PARSE_MIN_NOTNA_FRACTION = 0.8
"""Minimum successful parse fraction for treating a column as a datetime candidate."""

TARGET_CONTINUOUS_MIN_NUNIQUE = 20
"""Unique-value count above which a numeric target column is treated as continuous (regression)
rather than categorical (classification)."""

CORRELATION_WARN_THRESHOLD_INFERENCE = 0.95
"""Pairwise absolute correlation above this value triggers a collinearity note in dataset notes."""

CORRELATION_WARN_THRESHOLD_EXPLORE = 0.85
"""Pairwise absolute correlation above this value triggers the 'check correlated predictors'
suggestion in the exploration report."""

SKEW_THRESHOLD = 1.0
"""Absolute skewness above this value indicates a feature is right-skewed and may benefit from
a log or quantile transform."""

OUTLIER_MIN_SAMPLES = 10
"""Minimum non-null values required before running IQR outlier detection."""

OUTLIER_IQR_Q1 = 0.25
"""First quartile used in IQR-fence outlier detection."""

OUTLIER_IQR_Q3 = 0.75
"""Third quartile used in IQR-fence outlier detection."""

OUTLIER_IQR_MULTIPLIER = 1.5
"""Tukey multiplier for the IQR fence (standard 1.5 × IQR rule)."""

IMBALANCE_MINORITY_THRESHOLD = 0.2
"""Minority class frequency below this value is flagged as class imbalance."""

EXPLORE_MANY_FEATURES_THRESHOLD = 25
"""Feature count above this value triggers a dimensionality-reduction suggestion."""

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

DISTRIBUTION_QUANTILE_P05 = 0.05
"""5th-percentile quantile extracted from probabilistic regression distributions."""

DISTRIBUTION_QUANTILE_P50 = 0.50
"""Median (50th-percentile) quantile for probabilistic regression distributions."""

DISTRIBUTION_QUANTILE_P95 = 0.95
"""95th-percentile quantile for probabilistic regression distributions."""

MAPE_DENOMINATOR_FLOOR = 1e-9
"""Floor applied to the denominator in MAPE to avoid division by zero on near-zero actuals."""

TIMESERIES_HORIZON_DIVISOR = 4
"""The hold-out horizon is min(TIMESERIES_HOLDOUT, series_length // TIMESERIES_HORIZON_DIVISOR)."""

PROB_BINARY_THRESHOLD = 0.5
"""Default decision threshold for converting predicted probabilities to binary class labels."""

# ---------------------------------------------------------------------------
# Specific model parameters
# ---------------------------------------------------------------------------

RANSAC_MIN_SAMPLES_FRACTION = 0.5
"""Fraction of the dataset used as the random consensus subset in each RANSAC iteration."""

RIVER_LOGISTIC_LEARNING_RATE = 0.05
"""SGD learning rate for the online logistic regression model from the River library."""
