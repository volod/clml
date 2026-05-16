import clml.methods._context as ctx

METHODS = {
    "bagging_regressor": ctx.MethodSpec(
        "bagging_regressor",
        "Bagging Regressor",
        "1.11 Ensembles",
        "regression",
        "housing_prices",
        lambda seed: ctx.ensemble.BaggingRegressor(random_state=seed),
        notes=(
            "Bootstrap aggregation for regression: fits base regressors on resampled data "
            "and averages their predictions to reduce variance."
        ),
    ),
    "catboost_regressor": ctx.MethodSpec(
        "catboost_regressor",
        "CatBoost Regressor",
        "Extended gradient boosting",
        "regression",
        "housing_prices",
        ctx._catboost_regressor,
        notes="Third-party gradient boosting implementation for tabular regression.",
    ),
    "decision_tree_regressor": ctx.MethodSpec(
        "decision_tree_regressor",
        "Decision Tree Regressor",
        "1.10 Decision Trees",
        "regression",
        "housing_prices",
        lambda seed: ctx.tree.DecisionTreeRegressor(random_state=seed),
        ctx._tree_space,
    ),
    "ebm_regressor": ctx.MethodSpec(
        "ebm_regressor",
        "Explainable Boosting Regressor",
        "Extended interpretable models",
        "regression",
        "housing_prices",
        ctx._ebm_regressor,
        notes="Interpretable generalized additive boosted model from interpret.",
    ),
    "adaboost_regressor": ctx.MethodSpec(
        "adaboost_regressor",
        "AdaBoost Regressor",
        "1.11 Ensembles",
        "regression",
        "housing_prices",
        lambda seed: ctx.ensemble.AdaBoostRegressor(random_state=seed),
        notes=(
            "Sequentially reweights regression examples so later weak learners focus on "
            "larger residuals."
        ),
    ),
    "extra_trees_regressor": ctx.MethodSpec(
        "extra_trees_regressor",
        "Extra Trees Regressor",
        "1.11 Ensembles",
        "regression",
        "housing_prices",
        lambda seed: ctx.ensemble.ExtraTreesRegressor(
            random_state=seed, n_estimators=ctx.GBM_N_ESTIMATORS
        ),
        ctx._forest_space,
        notes=(
            "Extremely randomized trees for regression: random split thresholds reduce "
            "variance and often train quickly."
        ),
    ),
    "gradient_boosting_regressor": ctx.MethodSpec(
        "gradient_boosting_regressor",
        "Gradient Boosting Regressor",
        "1.11 Ensembles",
        "regression",
        "housing_prices",
        lambda seed: ctx.ensemble.GradientBoostingRegressor(random_state=seed),
        ctx._gradient_boosting_space,
        third_party_factories=ctx.GBM_COMPARISONS_REGRESSOR,
    ),
    "hist_gradient_boosting_regressor": ctx.MethodSpec(
        "hist_gradient_boosting_regressor",
        "Histogram Gradient Boosting Regressor",
        "1.11 Ensembles",
        "regression",
        "housing_prices",
        lambda seed: ctx.ensemble.HistGradientBoostingRegressor(random_state=seed),
        lambda trial: {
            "model__learning_rate": trial.suggest_float(
                "learning_rate", ctx.GB_LEARNING_RATE_MIN, ctx.GB_LEARNING_RATE_MAX, log=True
            ),
            "model__max_iter": trial.suggest_int(
                "max_iter", ctx.HGBT_MAX_ITER_MIN, ctx.HGBT_MAX_ITER_MAX
            ),
            "model__max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes", ctx.HGBT_MAX_LEAF_NODES_MIN, ctx.HGBT_MAX_LEAF_NODES_MAX
            ),
        },
        third_party_factories=ctx.GBM_COMPARISONS_REGRESSOR,
    ),
    "lightgbm_regressor": ctx.MethodSpec(
        "lightgbm_regressor",
        "LightGBM Regressor",
        "Extended gradient boosting",
        "regression",
        "housing_prices",
        ctx._lightgbm_regressor,
        ctx._gradient_boosting_space,
        notes="Third-party histogram gradient boosting implementation.",
    ),
    "ngboost_regressor": ctx.MethodSpec(
        "ngboost_regressor",
        "NGBoost Probabilistic Regressor",
        "Extended probabilistic boosting",
        "regression",
        "housing_prices",
        ctx._ngboost_regressor,
        notes=(
            "Third-party probabilistic gradient boosting regressor that predicts a full "
            "conditional distribution rather than only point estimates."
        ),
    ),
    "random_forest_regressor": ctx.MethodSpec(
        "random_forest_regressor",
        "Random Forest Regressor",
        "1.11 Ensembles",
        "regression",
        "housing_prices",
        lambda seed: ctx.ensemble.RandomForestRegressor(
            random_state=seed, n_estimators=ctx.GBM_N_ESTIMATORS
        ),
        ctx._forest_space,
    ),
    "xgboost_regressor": ctx.MethodSpec(
        "xgboost_regressor",
        "XGBoost Regressor",
        "Extended gradient boosting",
        "regression",
        "housing_prices",
        ctx._xgboost_regressor,
        ctx._gradient_boosting_space,
        notes="Third-party gradient boosting implementation for tabular regression.",
    ),
    "stacking_regressor": ctx.MethodSpec(
        "stacking_regressor",
        "Stacking Regressor",
        "1.11 Ensembles",
        "regression",
        "housing_prices",
        ctx._stacking_regressor,
        notes=(
            "Stacks Ridge, RandomForest, and GradientBoosting with a Ridge meta-learner. "
            "Out-of-fold base predictions serve as training data for the meta-learner, "
            "reducing overfitting risk compared to in-sample stacking."
        ),
    ),
    "voting_regressor": ctx.MethodSpec(
        "voting_regressor",
        "Voting Regressor",
        "1.11 Ensembles",
        "regression",
        "housing_prices",
        ctx._voting_regressor,
        notes=(
            "Averages predictions from Ridge, RandomForest, and GradientBoosting. "
            "Reduces variance through model diversity; complementary to boosting because "
            "it benefits from structurally different base learners."
        ),
    ),
}
