from typing import Any

from clml import constants


def _regularization_space(trial) -> dict[str, Any]:
    return {
        "model__alpha": trial.suggest_float(
            "alpha", constants.ALPHA_MIN, constants.ALPHA_MAX, log=True
        )
    }


def _huber_space(trial) -> dict[str, Any]:
    return {
        "model__epsilon": trial.suggest_float(
            "epsilon", constants.HUBER_EPSILON_MIN, constants.HUBER_EPSILON_MAX
        ),
        "model__alpha": trial.suggest_float(
            "alpha", constants.HUBER_ALPHA_MIN, constants.HUBER_ALPHA_MAX, log=True
        ),
    }


def _tweedie_space(trial) -> dict[str, Any]:
    return {
        "model__power": trial.suggest_float(
            "power", constants.TWEEDIE_POWER_MIN, constants.TWEEDIE_POWER_MAX
        ),
        "model__alpha": trial.suggest_float(
            "alpha", constants.TWEEDIE_ALPHA_MIN, constants.TWEEDIE_ALPHA_MAX, log=True
        ),
    }


def _glm_alpha_space(trial) -> dict[str, Any]:
    return {
        "model__alpha": trial.suggest_float(
            "alpha", constants.TWEEDIE_ALPHA_MIN, constants.TWEEDIE_ALPHA_MAX, log=True
        )
    }


def _quantile_space(trial) -> dict[str, Any]:
    return {
        "model__quantile": trial.suggest_float(
            "quantile", constants.QUANTILE_MIN, constants.QUANTILE_MAX
        ),
        "model__alpha": trial.suggest_float(
            "alpha", constants.TWEEDIE_ALPHA_MIN, constants.TWEEDIE_ALPHA_MAX, log=True
        ),
    }


def _nu_svc_space(trial) -> dict[str, Any]:
    return {
        "model__nu": trial.suggest_float("nu", constants.NU_MIN, constants.NU_MAX),
        "model__gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def _linear_svc_space(trial) -> dict[str, Any]:
    return {
        "model__C": trial.suggest_float("C", constants.SVM_C_MIN, constants.SVM_C_MAX, log=True)
    }


def _lle_space(trial) -> dict[str, Any]:
    return {
        "model__n_neighbors": trial.suggest_int(
            "n_neighbors", constants.LLE_N_NEIGHBORS_MIN, constants.LLE_N_NEIGHBORS_MAX
        )
    }


def _tsne_space(trial) -> dict[str, Any]:
    return {
        "model__perplexity": trial.suggest_float(
            "perplexity", constants.TSNE_PERPLEXITY_MIN, constants.TSNE_PERPLEXITY_MAX
        )
    }


def _umap_space(trial) -> dict[str, Any]:
    return {
        "model__n_neighbors": trial.suggest_int(
            "n_neighbors", constants.UMAP_N_NEIGHBORS_MIN, constants.UMAP_N_NEIGHBORS_MAX
        ),
        "model__min_dist": trial.suggest_float(
            "min_dist", constants.UMAP_MIN_DIST_MIN, constants.UMAP_MIN_DIST_MAX
        ),
    }


def _tree_space(trial) -> dict[str, Any]:
    return {
        "model__max_depth": trial.suggest_int(
            "max_depth", constants.TREE_MAX_DEPTH_MIN, constants.TREE_MAX_DEPTH_MAX
        ),
        "model__min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", constants.TREE_MIN_LEAF_MIN, constants.TREE_MIN_LEAF_MAX
        ),
    }


def _forest_space(trial) -> dict[str, Any]:
    return {
        "model__n_estimators": trial.suggest_int(
            "n_estimators", constants.FOREST_N_ESTIMATORS_MIN, constants.FOREST_N_ESTIMATORS_MAX
        ),
        "model__max_depth": trial.suggest_int(
            "max_depth", constants.TREE_MAX_DEPTH_MIN, constants.FOREST_MAX_DEPTH_MAX
        ),
        "model__min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", constants.TREE_MIN_LEAF_MIN, constants.FOREST_MIN_LEAF_MAX
        ),
    }


def _gradient_boosting_space(trial) -> dict[str, Any]:
    return {
        "model__n_estimators": trial.suggest_int(
            "n_estimators", constants.GB_N_ESTIMATORS_MIN, constants.GB_N_ESTIMATORS_MAX
        ),
        "model__learning_rate": trial.suggest_float(
            "learning_rate",
            constants.GB_LEARNING_RATE_MIN,
            constants.GB_LEARNING_RATE_MAX,
            log=True,
        ),
        "model__max_depth": trial.suggest_int(
            "max_depth", constants.GB_MAX_DEPTH_MIN, constants.GB_MAX_DEPTH_MAX
        ),
    }


def _neighbors_space(trial) -> dict[str, Any]:
    return {
        "model__n_neighbors": trial.suggest_int(
            "n_neighbors", constants.KNN_N_NEIGHBORS_MIN, constants.KNN_N_NEIGHBORS_MAX
        ),
        "model__weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
    }


def _svc_space(trial) -> dict[str, Any]:
    return {
        "model__C": trial.suggest_float("C", constants.SVM_C_MIN, constants.SVM_C_MAX, log=True),
        "model__gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def _cluster_space(trial) -> dict[str, Any]:
    return {
        "model__n_clusters": trial.suggest_int(
            "n_clusters", constants.CLUSTER_K_MIN, constants.CLUSTER_K_MAX
        )
    }


def _component_space(trial) -> dict[str, Any]:
    return {
        "model__n_components": trial.suggest_int(
            "n_components", constants.DIM_COMPONENTS_MIN, constants.DIM_COMPONENTS_MAX
        )
    }
