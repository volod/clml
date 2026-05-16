import clml.methods._context as ctx

METHODS = {
    "gaussian_process_regressor": ctx.MethodSpec(
        "gaussian_process_regressor",
        "Gaussian Process Regressor",
        "1.7 Gaussian Processes",
        "regression",
        "diabetes",
        lambda seed: ctx.gaussian_process.GaussianProcessRegressor(
            normalize_y=True,
            n_restarts_optimizer=ctx.GPR_N_RESTARTS,
            random_state=seed,
        ),
    ),
    "knn_regressor": ctx.MethodSpec(
        "knn_regressor",
        "Nearest Neighbors Regressor",
        "1.6 Nearest Neighbors",
        "regression",
        "housing_prices",
        lambda seed: ctx.neighbors.KNeighborsRegressor(),
        ctx._neighbors_space,
    ),
    "mlp_regressor": ctx.MethodSpec(
        "mlp_regressor",
        "Multi-layer Perceptron Regressor",
        "1.17 Neural network models (supervised)",
        "regression",
        "diabetes",
        lambda seed: ctx.neural_network.MLPRegressor(
            hidden_layer_sizes=ctx.MLP_REG_HIDDEN_LAYERS,
            max_iter=ctx.MLP_REG_MAX_ITER,
            random_state=seed,
        ),
    ),
}
