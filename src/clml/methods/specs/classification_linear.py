import clml.methods._context as ctx

METHODS = {
    "calibrated_logistic_regression": ctx.MethodSpec(
        "calibrated_logistic_regression",
        "Calibrated Logistic Regression",
        "1.16 Probability calibration",
        "classification",
        "credit_risk",
        lambda seed: ctx.CalibratedClassifierCV(
            ctx.linear_model.LogisticRegression(max_iter=ctx.LOGISTIC_MAX_ITER)
        ),
    ),
    "linear_svc_classifier": ctx.MethodSpec(
        "linear_svc_classifier",
        "Linear Support Vector Classifier",
        "1.4 Support Vector Machines",
        "classification",
        "credit_risk",
        lambda seed: ctx.svm.LinearSVC(
            dual="auto", max_iter=ctx.LINEAR_SVC_MAX_ITER, random_state=seed
        ),
        ctx._linear_svc_space,
        notes=(
            "Linear SVM optimized by liblinear; efficient for high-dimensional or larger "
            "linear classification problems."
        ),
    ),
    "logistic_regression": ctx.MethodSpec(
        "logistic_regression",
        "Logistic Regression",
        "1.1 Linear Models",
        "classification",
        "credit_risk",
        lambda seed: ctx.linear_model.LogisticRegression(
            max_iter=ctx.LOGISTIC_MAX_ITER, random_state=seed
        ),
        lambda trial: {"model__C": trial.suggest_float("C", ctx.LR_C_MIN, ctx.LR_C_MAX, log=True)},
    ),
    "nu_svc_classifier": ctx.MethodSpec(
        "nu_svc_classifier",
        "Nu Support Vector Classifier",
        "1.4 Support Vector Machines",
        "classification",
        "breast_cancer",
        lambda seed: ctx.svm.NuSVC(nu=ctx.NU_SVC_NU, probability=True, random_state=seed),
        ctx._nu_svc_space,
        notes=(
            "Kernel SVM parameterized by nu, which controls an upper bound on margin "
            "errors and a lower bound on support vectors."
        ),
    ),
    "sgd_classifier": ctx.MethodSpec(
        "sgd_classifier",
        "Stochastic Gradient Descent Classifier",
        "1.5 Stochastic Gradient Descent",
        "classification",
        "credit_risk",
        lambda seed: ctx.linear_model.SGDClassifier(random_state=seed),
        lambda trial: {
            "model__alpha": trial.suggest_float(
                "alpha", ctx.SGD_ALPHA_MIN, ctx.SGD_ALPHA_MAX, log=True
            ),
            "model__loss": trial.suggest_categorical(
                "loss", ["hinge", "log_loss", "modified_huber"]
            ),
        },
    ),
    "svc_classifier": ctx.MethodSpec(
        "svc_classifier",
        "Support Vector Classifier",
        "1.4 Support Vector Machines",
        "classification",
        "breast_cancer",
        lambda seed: ctx.svm.SVC(random_state=seed),
        ctx._svc_space,
    ),
    "passive_aggressive_classifier": ctx.MethodSpec(
        "passive_aggressive_classifier",
        "Passive-Aggressive Classifier",
        "1.5 Stochastic Gradient Descent",
        "classification",
        "credit_risk",
        lambda seed: ctx.linear_model.SGDClassifier(
            loss="hinge",
            penalty=None,
            learning_rate="pa1",
            eta0=1.0,
            max_iter=ctx.PASSIVE_AGGRESSIVE_MAX_ITER,
            random_state=seed,
        ),
        lambda trial: {
            "model__eta0": trial.suggest_float("eta0", ctx.PA_C_MIN, ctx.PA_C_MAX, log=True)
        },
        notes=(
            "Online linear classifier: stays passive when prediction is correct, makes the "
            "minimal aggressive update when wrong. Natural fit for streaming or incremental "
            "learning; does not produce probability estimates."
        ),
    ),
    "ridge_classifier": ctx.MethodSpec(
        "ridge_classifier",
        "Ridge Classifier",
        "1.1 Linear Models",
        "classification",
        "credit_risk",
        lambda seed: ctx.linear_model.RidgeClassifier(),
        ctx._regularization_space,
        notes=(
            "Classification via Ridge regression: fits Ridge to one-hot class labels and "
            "predicts the highest-score class. Faster than LogisticRegression for many-class "
            "problems; does not produce probability estimates."
        ),
    ),
}
