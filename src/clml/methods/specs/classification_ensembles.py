import clml.methods._context as ctx

METHODS = {
    "adaboost_classifier": ctx.MethodSpec(
        "adaboost_classifier",
        "AdaBoost Classifier",
        "1.11 Ensembles",
        "classification",
        "credit_risk",
        lambda seed: ctx.ensemble.AdaBoostClassifier(random_state=seed),
    ),
    "bagging_classifier": ctx.MethodSpec(
        "bagging_classifier",
        "Bagging Classifier",
        "1.11 Ensembles",
        "classification",
        "credit_risk",
        lambda seed: ctx.ensemble.BaggingClassifier(random_state=seed),
    ),
    "catboost_classifier": ctx.MethodSpec(
        "catboost_classifier",
        "CatBoost Classifier",
        "Extended gradient boosting",
        "classification",
        "credit_risk",
        ctx._catboost_classifier,
        notes="Third-party gradient boosting implementation for tabular classification.",
    ),
    "decision_tree_classifier": ctx.MethodSpec(
        "decision_tree_classifier",
        "Decision Tree Classifier",
        "1.10 Decision Trees",
        "classification",
        "credit_risk",
        lambda seed: ctx.tree.DecisionTreeClassifier(random_state=seed),
        ctx._tree_space,
    ),
    "ebm_classifier": ctx.MethodSpec(
        "ebm_classifier",
        "Explainable Boosting Classifier",
        "Extended interpretable models",
        "classification",
        "credit_risk",
        ctx._ebm_classifier,
        notes="Interpretable generalized additive boosted model from interpret.",
    ),
    "extra_trees_classifier": ctx.MethodSpec(
        "extra_trees_classifier",
        "Extra Trees Classifier",
        "1.11 Ensembles",
        "classification",
        "credit_risk",
        lambda seed: ctx.ensemble.ExtraTreesClassifier(
            random_state=seed, n_estimators=ctx.GBM_N_ESTIMATORS
        ),
        ctx._forest_space,
    ),
    "gradient_boosting_classifier": ctx.MethodSpec(
        "gradient_boosting_classifier",
        "Gradient Boosting Classifier",
        "1.11 Ensembles",
        "classification",
        "credit_risk",
        lambda seed: ctx.ensemble.GradientBoostingClassifier(random_state=seed),
        ctx._gradient_boosting_space,
        third_party_factories=ctx.GBM_COMPARISONS_CLASSIFIER,
    ),
    "hist_gradient_boosting_classifier": ctx.MethodSpec(
        "hist_gradient_boosting_classifier",
        "Histogram Gradient Boosting Classifier",
        "1.11 Ensembles",
        "classification",
        "credit_risk",
        lambda seed: ctx.ensemble.HistGradientBoostingClassifier(random_state=seed),
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
        third_party_factories=ctx.GBM_COMPARISONS_CLASSIFIER,
    ),
    "lightgbm_classifier": ctx.MethodSpec(
        "lightgbm_classifier",
        "LightGBM Classifier",
        "Extended gradient boosting",
        "classification",
        "credit_risk",
        ctx._lightgbm_classifier,
        ctx._gradient_boosting_space,
        notes="Third-party histogram gradient boosting implementation.",
    ),
    "random_forest_classifier": ctx.MethodSpec(
        "random_forest_classifier",
        "Random Forest Classifier",
        "1.11 Ensembles",
        "classification",
        "credit_risk",
        lambda seed: ctx.ensemble.RandomForestClassifier(
            random_state=seed, n_estimators=ctx.GBM_N_ESTIMATORS
        ),
        ctx._forest_space,
    ),
    "xgboost_classifier": ctx.MethodSpec(
        "xgboost_classifier",
        "XGBoost Classifier",
        "Extended gradient boosting",
        "classification",
        "credit_risk",
        ctx._xgboost_classifier,
        ctx._gradient_boosting_space,
        notes="Third-party gradient boosting implementation for tabular classification.",
    ),
    "stacking_classifier": ctx.MethodSpec(
        "stacking_classifier",
        "Stacking Classifier",
        "1.11 Ensembles",
        "classification",
        "credit_risk",
        ctx._stacking_classifier,
        notes=(
            "Meta-learning ensemble: LogisticRegression, RandomForest, and GradientBoosting "
            "generate out-of-fold predictions used as features for a Logistic meta-learner. "
            "More expressive than voting; requires cross-validation fitting of base learners."
        ),
    ),
    "voting_classifier": ctx.MethodSpec(
        "voting_classifier",
        "Voting Classifier",
        "1.11 Ensembles",
        "classification",
        "credit_risk",
        ctx._voting_classifier,
        notes=(
            "Soft-voting ensemble: averages predicted probabilities from LogisticRegression, "
            "RandomForest, and SVC. Reduces variance compared to any single learner; "
            "simpler than stacking and less prone to overfitting the training set."
        ),
    ),
}
