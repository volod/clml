import clml.methods._context as ctx

METHODS = {
    "category_hashing_encoder_sgd": ctx.MethodSpec(
        "category_hashing_encoder_sgd",
        "Hashing Encoder + SGD Classifier",
        "Extended categorical encoding",
        "categorical_encoding",
        "credit_risk",
        lambda seed: None,
        notes="Uses category-encoders HashingEncoder with an SGD classifier.",
    ),
    "category_leave_one_out_encoder_logistic": ctx.MethodSpec(
        "category_leave_one_out_encoder_logistic",
        "Leave-One-Out Encoder + Logistic Regression",
        "Extended categorical encoding",
        "categorical_encoding",
        "credit_risk",
        lambda seed: None,
        notes="Uses category-encoders LeaveOneOutEncoder for categorical features.",
    ),
    "category_target_encoder_logistic": ctx.MethodSpec(
        "category_target_encoder_logistic",
        "Target Encoder + Logistic Regression",
        "Extended categorical encoding",
        "categorical_encoding",
        "credit_risk",
        lambda seed: None,
        notes="Uses category-encoders TargetEncoder for categorical features.",
    ),
    "feature_select_k_best_f_classif": ctx.MethodSpec(
        "feature_select_k_best_f_classif",
        "SelectKBest f-test Feature Selection",
        "1.13 Feature selection",
        "feature_selection",
        "breast_cancer",
        ctx._select_k_best_f_classif,
        notes=(
            "Univariate feature filter using ANOVA F-statistics before a downstream "
            "classification model."
        ),
    ),
    "feature_select_k_best_mutual_info": ctx.MethodSpec(
        "feature_select_k_best_mutual_info",
        "SelectKBest Mutual Information Feature Selection",
        "1.13 Feature selection",
        "feature_selection",
        "breast_cancer",
        ctx._select_k_best_mutual_info,
        notes=(
            "Univariate feature filter using mutual information, which can capture "
            "nonlinear feature-target dependence."
        ),
    ),
    "feature_rfe_logistic": ctx.MethodSpec(
        "feature_rfe_logistic",
        "Recursive Feature Elimination",
        "1.13 Feature selection",
        "feature_selection",
        "breast_cancer",
        ctx._rfe_logistic,
        notes=(
            "Wrapper feature selection that repeatedly removes the weakest logistic-regression "
            "features."
        ),
    ),
    "feature_rfecv_logistic": ctx.MethodSpec(
        "feature_rfecv_logistic",
        "Recursive Feature Elimination with CV",
        "1.13 Feature selection",
        "feature_selection",
        "breast_cancer",
        ctx._rfecv_logistic,
        notes=(
            "RFE with cross-validation to choose the selected feature count based on "
            "downstream macro-F1."
        ),
    ),
    "feature_select_from_l1_linear_svc": ctx.MethodSpec(
        "feature_select_from_l1_linear_svc",
        "SelectFromModel L1 Linear SVC Feature Selection",
        "1.13 Feature selection",
        "feature_selection",
        "breast_cancer",
        ctx._select_from_l1_linear_svc,
        notes=(
            "Embedded feature selection using an L1-regularized linear SVM before "
            "downstream evaluation."
        ),
    ),
    "imbalanced_balanced_random_forest": ctx.MethodSpec(
        "imbalanced_balanced_random_forest",
        "Balanced Random Forest",
        "Extended imbalanced learning",
        "imbalanced_classification",
        "credit_risk",
        lambda seed: None,
        notes="Uses imbalanced-learn BalancedRandomForestClassifier.",
    ),
    "imbalanced_random_undersampling_logistic": ctx.MethodSpec(
        "imbalanced_random_undersampling_logistic",
        "Random Under-Sampling + Logistic Regression",
        "Extended imbalanced learning",
        "imbalanced_classification",
        "credit_risk",
        lambda seed: None,
        notes="Uses imbalanced-learn RandomUnderSampler before logistic regression.",
    ),
    "imbalanced_smote_logistic": ctx.MethodSpec(
        "imbalanced_smote_logistic",
        "SMOTE + Logistic Regression",
        "Extended imbalanced learning",
        "imbalanced_classification",
        "credit_risk",
        lambda seed: None,
        notes="Uses imbalanced-learn SMOTE before logistic regression.",
    ),
    "lifelines_cox_model": ctx.MethodSpec(
        "lifelines_cox_model",
        "Cox Proportional Hazards Model",
        "Extended survival analysis",
        "survival_cox",
        "customer_survival",
        lambda seed: None,
        notes="Uses lifelines CoxPHFitter for time-to-event modeling.",
    ),
    "lifelines_kaplan_meier": ctx.MethodSpec(
        "lifelines_kaplan_meier",
        "Kaplan-Meier Survival Curves",
        "Extended survival analysis",
        "survival_kaplan_meier",
        "customer_survival",
        lambda seed: None,
        notes="Uses lifelines KaplanMeierFitter for group survival curves.",
    ),
    "cvxpy_portfolio_optimization": ctx.MethodSpec(
        "cvxpy_portfolio_optimization",
        "CVXPY Portfolio Optimization",
        "Extended convex optimization",
        "cvxpy_portfolio",
        "portfolio_assets",
        lambda seed: None,
        notes="Mean-variance portfolio allocation with CVXPY.",
    ),
    "cvxpy_quadratic_programming": ctx.MethodSpec(
        "cvxpy_quadratic_programming",
        "CVXPY Quadratic Programming",
        "Extended convex optimization",
        "cvxpy_quadratic",
        "portfolio_assets",
        lambda seed: None,
        notes="Constrained quadratic risk minimization with CVXPY.",
    ),
    "river_logistic_regression": ctx.MethodSpec(
        "river_logistic_regression",
        "River Online Logistic Regression",
        "Extended incremental learning",
        "incremental_classification",
        "streaming_churn",
        ctx._river_logistic_regression,
        notes=(
            "Online binary classifier from River; evaluates each sample before learning it "
            "to demonstrate prequential streaming evaluation."
        ),
    ),
    "scipy_linear_programming": ctx.MethodSpec(
        "scipy_linear_programming",
        "SciPy Linear Programming",
        "Extended mathematical optimization",
        "linear_programming",
        "production_planning",
        lambda seed: None,
        notes="Solves product-mix profit maximization with scipy.optimize.linprog.",
    ),
    "scipy_nonlinear_optimization": ctx.MethodSpec(
        "scipy_nonlinear_optimization",
        "SciPy Nonlinear Constrained Optimization",
        "Extended mathematical optimization",
        "nonlinear_optimization",
        "marketing_mix",
        lambda seed: None,
        notes="Solves bounded marketing-budget allocation with scipy.optimize.minimize.",
    ),
    "statsmodels_glm_regression": ctx.MethodSpec(
        "statsmodels_glm_regression",
        "statsmodels GLM Gaussian Regression",
        "Extended statistical modeling",
        "statsmodels_regression",
        "housing_prices",
        lambda seed: None,
        notes="Uses statsmodels GLM for inference-oriented regression.",
    ),
    "statsmodels_logit": ctx.MethodSpec(
        "statsmodels_logit",
        "statsmodels Logistic Regression",
        "Extended statistical modeling",
        "statsmodels_classification",
        "credit_risk",
        lambda seed: None,
        notes="Uses statsmodels Logit with coefficient inference.",
    ),
    "statsmodels_ols": ctx.MethodSpec(
        "statsmodels_ols",
        "statsmodels Ordinary Least Squares",
        "Extended statistical modeling",
        "statsmodels_regression",
        "housing_prices",
        lambda seed: None,
        notes="Uses statsmodels OLS with coefficient inference.",
    ),
    "statsmodels_robust_regression": ctx.MethodSpec(
        "statsmodels_robust_regression",
        "statsmodels Robust Linear Model",
        "Extended statistical modeling",
        "statsmodels_regression",
        "housing_prices",
        lambda seed: None,
        notes="Uses statsmodels RLM to reduce outlier influence.",
    ),
    "statsmodels_exponential_smoothing": ctx.MethodSpec(
        "statsmodels_exponential_smoothing",
        "statsmodels Exponential Smoothing",
        "Extended time series forecasting",
        "timeseries",
        "airline_passengers",
        lambda seed: None,
        notes=(
            "Holt-Winters exponential smoothing for trend and seasonal forecasting on a "
            "single ordered series."
        ),
    ),
    "statsmodels_sarimax": ctx.MethodSpec(
        "statsmodels_sarimax",
        "statsmodels SARIMAX",
        "Extended time series forecasting",
        "timeseries",
        "airline_passengers",
        lambda seed: None,
        notes=(
            "Seasonal ARIMA state-space model for autocorrelation, trend differencing, and "
            "annual seasonality."
        ),
    ),
    "statsmodels_autoreg": ctx.MethodSpec(
        "statsmodels_autoreg",
        "statsmodels AutoReg",
        "Extended time series forecasting",
        "timeseries",
        "airline_passengers",
        lambda seed: None,
        notes="Autoregressive forecasting model using lagged values of the target series.",
    ),
}
