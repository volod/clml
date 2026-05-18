import clml.methods._context as ctx

METHODS = {
    "elastic_net_regression": ctx.MethodSpec(
        "elastic_net_regression",
        "Elastic-Net Regression",
        "1.1 Linear Models",
        "regression",
        "diabetes",
        lambda seed: ctx.linear_model.ElasticNet(
            max_iter=ctx.ELASTIC_NET_MAX_ITER, random_state=seed
        ),
        lambda trial: {
            "model__alpha": trial.suggest_float(
                "alpha", ctx.ALPHA_MIN, ctx.ELASTIC_NET_ALPHA_MAX, log=True
            ),
            "model__l1_ratio": trial.suggest_float("l1_ratio", ctx.L1_RATIO_MIN, ctx.L1_RATIO_MAX),
        },
    ),
    "ard_regression": ctx.MethodSpec(
        "ard_regression",
        "Automatic Relevance Determination Regression",
        "1.1 Linear Models",
        "regression",
        "diabetes",
        lambda seed: ctx.linear_model.ARDRegression(),
        notes=(
            "Bayesian linear regression with per-feature precision parameters; irrelevant "
            "features are shrunk toward zero through automatic relevance determination."
        ),
    ),
    "isotonic_regression": ctx.MethodSpec(
        "isotonic_regression",
        "Isotonic Regression",
        "1.15 Isotonic regression",
        "regression_1d",
        "monotone_1d",
        lambda seed: ctx.isotonic.IsotonicRegression(out_of_bounds="clip"),
    ),
    "kernel_ridge_regression": ctx.MethodSpec(
        "kernel_ridge_regression",
        "Kernel Ridge Regression",
        "1.3 Kernel ridge regression",
        "regression",
        "diabetes",
        lambda seed: ctx.kernel_ridge.KernelRidge(kernel="rbf"),
        lambda trial: {
            "model__alpha": trial.suggest_float("alpha", ctx.ALPHA_MIN, ctx.ALPHA_MAX, log=True),
            "model__gamma": trial.suggest_float(
                "gamma", ctx.KERNEL_RIDGE_GAMMA_MIN, ctx.KERNEL_RIDGE_GAMMA_MAX, log=True
            ),
        },
    ),
    "lasso_regression": ctx.MethodSpec(
        "lasso_regression",
        "Lasso Regression",
        "1.1 Linear Models",
        "regression",
        "diabetes",
        lambda seed: ctx.linear_model.Lasso(max_iter=ctx.ELASTIC_NET_MAX_ITER, random_state=seed),
        ctx._regularization_space,
    ),
    "linear_regression": ctx.MethodSpec(
        "linear_regression",
        "Ordinary Least Squares",
        "1.1 Linear Models",
        "regression",
        "housing_prices",
        lambda seed: ctx.linear_model.LinearRegression(),
    ),
    "pls_regression": ctx.MethodSpec(
        "pls_regression",
        "Partial Least Squares Regression",
        "1.8 Cross decomposition",
        "regression",
        "diabetes",
        lambda seed: ctx.PLSRegression(n_components=ctx.DIM_DEFAULT_COMPONENTS),
        lambda trial: {
            "model__n_components": trial.suggest_int(
                "n_components", ctx.PLS_COMPONENTS_MIN, ctx.PLS_COMPONENTS_MAX
            )
        },
    ),
    "poisson_regression": ctx.MethodSpec(
        "poisson_regression",
        "Poisson Regression",
        "1.1 Linear Models",
        "regression",
        "housing_prices",
        lambda seed: ctx.linear_model.PoissonRegressor(
            alpha=ctx.TWEEDIE_DEFAULT_ALPHA, max_iter=ctx.TWEEDIE_MAX_ITER
        ),
        ctx._glm_alpha_space,
        notes=(
            "Generalized linear model with log link for non-negative count or exposure-like "
            "targets."
        ),
    ),
    "quantile_regression": ctx.MethodSpec(
        "quantile_regression",
        "Quantile Regression",
        "1.1 Linear Models",
        "regression",
        "housing_prices",
        lambda seed: ctx.linear_model.QuantileRegressor(
            quantile=ctx.QUANTILE_DEFAULT, alpha=ctx.TWEEDIE_DEFAULT_ALPHA
        ),
        ctx._quantile_space,
        notes=(
            "Estimates a conditional quantile instead of the conditional mean; the default "
            "median model is useful for asymmetric errors and uncertainty bands."
        ),
    ),
    "ransac_regression": ctx.MethodSpec(
        "ransac_regression",
        "RANSAC Regression",
        "1.1 Linear Models",
        "regression",
        "housing_prices",
        lambda seed: ctx.linear_model.RANSACRegressor(
            min_samples=ctx.RANSAC_MIN_SAMPLES_FRACTION,
            random_state=seed,
        ),
        notes=(
            "Robust regression by repeatedly fitting on random subsets and selecting the "
            "largest consensus set of inliers."
        ),
    ),
    "ridge_regression": ctx.MethodSpec(
        "ridge_regression",
        "Ridge Regression",
        "1.1 Linear Models",
        "regression",
        "diabetes",
        lambda seed: ctx.linear_model.Ridge(random_state=seed),
        ctx._regularization_space,
    ),
    "sgd_regressor": ctx.MethodSpec(
        "sgd_regressor",
        "Stochastic Gradient Descent Regressor",
        "1.5 Stochastic Gradient Descent",
        "regression",
        "housing_prices",
        lambda seed: ctx.linear_model.SGDRegressor(random_state=seed),
        ctx._regularization_space,
    ),
    "gamma_regression": ctx.MethodSpec(
        "gamma_regression",
        "Gamma Regression",
        "1.1 Linear Models",
        "regression",
        "housing_prices",
        lambda seed: ctx.linear_model.GammaRegressor(
            alpha=ctx.TWEEDIE_DEFAULT_ALPHA, max_iter=ctx.TWEEDIE_MAX_ITER
        ),
        ctx._glm_alpha_space,
        notes=(
            "Generalized linear model for strictly positive, right-skewed targets such as "
            "loss severity, costs, and durations."
        ),
    ),
    "svr_regression": ctx.MethodSpec(
        "svr_regression",
        "Support Vector Regressor",
        "1.4 Support Vector Machines",
        "regression",
        "diabetes",
        lambda seed: ctx.svm.SVR(C=ctx.SVR_DEFAULT_C),
        ctx._svc_space,
    ),
    "theil_sen_regression": ctx.MethodSpec(
        "theil_sen_regression",
        "Theil-Sen Regression",
        "1.1 Linear Models",
        "regression",
        "diabetes",
        lambda seed: ctx.linear_model.TheilSenRegressor(random_state=seed),
        notes=(
            "Robust linear regression based on median slopes across subsamples; less "
            "sensitive to outliers than ordinary least squares."
        ),
    ),
    "bayesian_ridge_regression": ctx.MethodSpec(
        "bayesian_ridge_regression",
        "Bayesian Ridge Regression",
        "1.1 Linear Models",
        "regression",
        "housing_prices",
        lambda seed: ctx.linear_model.BayesianRidge(),
        notes=(
            "Probabilistic Ridge that infers regularization strength from data via evidence "
            "maximization. Provides posterior mean and variance per coefficient; useful for "
            "uncertainty quantification without cross-validating alpha manually."
        ),
    ),
    "huber_regression": ctx.MethodSpec(
        "huber_regression",
        "Huber Regression",
        "1.1 Linear Models",
        "regression",
        "housing_prices",
        lambda seed: ctx.linear_model.HuberRegressor(epsilon=ctx.HUBER_DEFAULT_EPSILON),
        ctx._huber_space,
        notes=(
            "Robust linear regression using Huber loss: quadratic for small residuals, linear "
            "for large ones. Less sensitive to outliers than OLS. epsilon controls the boundary "
            "between quadratic and linear regime; smaller epsilon = more robust."
        ),
    ),
    "tweedie_regression": ctx.MethodSpec(
        "tweedie_regression",
        "Tweedie GLM Regression",
        "1.1 Linear Models",
        "regression",
        "housing_prices",
        lambda seed: ctx.linear_model.TweedieRegressor(
            power=ctx.TWEEDIE_DEFAULT_POWER,
            alpha=ctx.TWEEDIE_DEFAULT_ALPHA,
            max_iter=ctx.TWEEDIE_MAX_ITER,
        ),
        ctx._tweedie_space,
        notes=(
            "Generalized linear model for non-negative, possibly zero-inflated targets. "
            "power=0: Normal; power=1: Poisson (counts); power=2: Gamma (skewed positives); "
            "power in (1,2): Tweedie compound. Useful for insurance losses, claim counts."
        ),
    ),
}
