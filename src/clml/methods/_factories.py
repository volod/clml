from functools import partial

from sklearn import ensemble, feature_selection, linear_model, svm

from clml import constants
from clml.config.settings import get_settings


def _xgboost_classifier(seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=constants.GBM_N_ESTIMATORS,
        max_depth=constants.GBM_MAX_DEPTH,
        learning_rate=constants.GBM_LEARNING_RATE,
        subsample=constants.GBM_SUBSAMPLE,
        eval_metric="logloss",
        random_state=seed,
    )


def _xgboost_regressor(seed: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=constants.GBM_N_ESTIMATORS,
        max_depth=constants.GBM_MAX_DEPTH,
        learning_rate=constants.GBM_LEARNING_RATE,
        subsample=constants.GBM_SUBSAMPLE,
        random_state=seed,
    )


def _lightgbm_classifier(seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=constants.GBM_N_ESTIMATORS,
        learning_rate=constants.GBM_LEARNING_RATE,
        random_state=seed,
        verbose=-1,
    )


def _lightgbm_regressor(seed: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=constants.GBM_N_ESTIMATORS,
        learning_rate=constants.GBM_LEARNING_RATE,
        random_state=seed,
        verbose=-1,
    )


def _catboost_classifier(seed: int):
    from catboost import CatBoostClassifier

    train_dir = str(get_settings().data_dir / "catboost_info")
    return CatBoostClassifier(
        iterations=constants.GBM_N_ESTIMATORS,
        learning_rate=constants.GBM_LEARNING_RATE,
        random_seed=seed,
        verbose=False,
        train_dir=train_dir,
    )


def _catboost_regressor(seed: int):
    from catboost import CatBoostRegressor

    train_dir = str(get_settings().data_dir / "catboost_info")
    return CatBoostRegressor(
        iterations=constants.GBM_N_ESTIMATORS,
        learning_rate=constants.GBM_LEARNING_RATE,
        random_seed=seed,
        verbose=False,
        train_dir=train_dir,
    )


def _ebm_classifier(seed: int):
    from interpret.glassbox import ExplainableBoostingClassifier

    return ExplainableBoostingClassifier(
        random_state=seed,
        interactions=constants.EBM_INTERACTIONS,
        max_rounds=constants.EBM_MAX_ROUNDS,
    )


def _ebm_regressor(seed: int):
    from interpret.glassbox import ExplainableBoostingRegressor

    return ExplainableBoostingRegressor(
        random_state=seed,
        interactions=constants.EBM_INTERACTIONS,
        max_rounds=constants.EBM_MAX_ROUNDS,
    )


def _umap(seed: int):
    from umap import UMAP

    return UMAP(
        n_components=constants.DIM_DEFAULT_COMPONENTS,
        n_neighbors=constants.UMAP_N_NEIGHBORS,
        min_dist=constants.UMAP_MIN_DIST,
        random_state=seed,
    )


def _ngboost_regressor(seed: int):
    from ngboost import NGBRegressor

    return NGBRegressor(
        n_estimators=constants.NGBOOST_N_ESTIMATORS,
        learning_rate=constants.NGBOOST_LEARNING_RATE,
        random_state=seed,
        verbose=False,
    )


def _river_logistic_regression(seed: int):
    from river import compose, linear_model, optim, preprocessing

    _ = seed
    return compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        (
            "model",
            linear_model.LogisticRegression(
                optimizer=optim.SGD(constants.RIVER_LOGISTIC_LEARNING_RATE),
            ),
        ),
    )


def _select_k_best_f_classif(seed: int):
    _ = seed
    return feature_selection.SelectKBest(
        score_func=feature_selection.f_classif,
        k=constants.FEATURE_SELECTION_K,
    )


def _select_k_best_mutual_info(seed: int):
    score_func = partial(feature_selection.mutual_info_classif, random_state=seed)
    return feature_selection.SelectKBest(score_func=score_func, k=constants.FEATURE_SELECTION_K)


def _rfe_logistic(seed: int):
    estimator = linear_model.LogisticRegression(
        max_iter=constants.LOGISTIC_MAX_ITER,
        solver="liblinear",
        random_state=seed,
    )
    return feature_selection.RFE(
        estimator=estimator,
        n_features_to_select=constants.FEATURE_SELECTION_K,
        step=1,
    )


def _rfecv_logistic(seed: int):
    estimator = linear_model.LogisticRegression(
        max_iter=constants.LOGISTIC_MAX_ITER,
        solver="liblinear",
        random_state=seed,
    )
    return feature_selection.RFECV(
        estimator=estimator,
        min_features_to_select=constants.FEATURE_SELECTION_MIN_FEATURES,
        cv=constants.CV_FOLDS,
        scoring="f1_macro",
        step=1,
    )


def _select_from_l1_linear_svc(seed: int):
    estimator = svm.LinearSVC(
        penalty="l1",
        dual=False,
        C=constants.FEATURE_SELECTION_L1_C,
        max_iter=constants.FEATURE_SELECTION_MAX_ITER,
        random_state=seed,
    )
    return feature_selection.SelectFromModel(
        estimator=estimator,
        max_features=constants.FEATURE_SELECTION_K,
    )


def _stacking_classifier(seed: int):
    return ensemble.StackingClassifier(
        estimators=[
            (
                "lr",
                linear_model.LogisticRegression(
                    max_iter=constants.ENSEMBLE_BASE_MAX_ITER, random_state=seed
                ),
            ),
            (
                "rf",
                ensemble.RandomForestClassifier(
                    n_estimators=constants.ENSEMBLE_BASE_N_ESTIMATORS, random_state=seed
                ),
            ),
            (
                "gb",
                ensemble.GradientBoostingClassifier(
                    n_estimators=constants.ENSEMBLE_BASE_N_ESTIMATORS, random_state=seed
                ),
            ),
        ],
        final_estimator=linear_model.LogisticRegression(
            max_iter=constants.ENSEMBLE_BASE_MAX_ITER, random_state=seed
        ),
        cv=constants.CV_FOLDS,
        n_jobs=1,
    )


def _stacking_regressor(seed: int):
    return ensemble.StackingRegressor(
        estimators=[
            ("ridge", linear_model.Ridge()),
            (
                "rf",
                ensemble.RandomForestRegressor(
                    n_estimators=constants.ENSEMBLE_BASE_N_ESTIMATORS, random_state=seed
                ),
            ),
            (
                "gb",
                ensemble.GradientBoostingRegressor(
                    n_estimators=constants.ENSEMBLE_BASE_N_ESTIMATORS, random_state=seed
                ),
            ),
        ],
        final_estimator=linear_model.Ridge(),
        cv=constants.CV_FOLDS,
        n_jobs=1,
    )


def _voting_classifier(seed: int):
    return ensemble.VotingClassifier(
        estimators=[
            (
                "lr",
                linear_model.LogisticRegression(
                    max_iter=constants.ENSEMBLE_BASE_MAX_ITER, random_state=seed
                ),
            ),
            (
                "rf",
                ensemble.RandomForestClassifier(
                    n_estimators=constants.ENSEMBLE_BASE_N_ESTIMATORS, random_state=seed
                ),
            ),
            ("svc", svm.SVC(probability=True, random_state=seed)),
        ],
        voting="soft",
    )


def _voting_regressor(seed: int):
    return ensemble.VotingRegressor(
        estimators=[
            ("ridge", linear_model.Ridge()),
            (
                "rf",
                ensemble.RandomForestRegressor(
                    n_estimators=constants.ENSEMBLE_BASE_N_ESTIMATORS, random_state=seed
                ),
            ),
            (
                "gb",
                ensemble.GradientBoostingRegressor(
                    n_estimators=constants.ENSEMBLE_BASE_N_ESTIMATORS, random_state=seed
                ),
            ),
        ]
    )


GBM_COMPARISONS_CLASSIFIER = {
    "xgboost": _xgboost_classifier,
    "lightgbm": _lightgbm_classifier,
    "catboost": _catboost_classifier,
}
GBM_COMPARISONS_REGRESSOR = {
    "xgboost": _xgboost_regressor,
    "lightgbm": _lightgbm_regressor,
    "catboost": _catboost_regressor,
}
