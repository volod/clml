import clml.methods._context as ctx

METHODS = {
    "bernoulli_nb": ctx.MethodSpec(
        "bernoulli_nb",
        "Bernoulli Naive Bayes",
        "1.9 Naive Bayes",
        "classification",
        "digits",
        lambda seed: ctx.naive_bayes.BernoulliNB(),
        needs_nonnegative=True,
    ),
    "gaussian_nb": ctx.MethodSpec(
        "gaussian_nb",
        "Gaussian Naive Bayes",
        "1.9 Naive Bayes",
        "classification",
        "breast_cancer",
        lambda seed: ctx.naive_bayes.GaussianNB(),
    ),
    "gaussian_process_classifier": ctx.MethodSpec(
        "gaussian_process_classifier",
        "Gaussian Process Classifier",
        "1.7 Gaussian Processes",
        "classification",
        "iris",
        lambda seed: ctx.gaussian_process.GaussianProcessClassifier(random_state=seed),
    ),
    "knn_classifier": ctx.MethodSpec(
        "knn_classifier",
        "Nearest Neighbors Classifier",
        "1.6 Nearest Neighbors",
        "classification",
        "credit_risk",
        lambda seed: ctx.neighbors.KNeighborsClassifier(),
        ctx._neighbors_space,
    ),
    "label_propagation": ctx.MethodSpec(
        "label_propagation",
        "Label Propagation",
        "1.14 Semi-supervised learning",
        "classification",
        "moons",
        lambda seed: ctx.semi_supervised.LabelPropagation(),
    ),
    "lda_classifier": ctx.MethodSpec(
        "lda_classifier",
        "Linear Discriminant Analysis",
        "1.2 Linear and Quadratic Discriminant Analysis",
        "classification",
        "credit_risk",
        lambda seed: ctx.discriminant_analysis.LinearDiscriminantAnalysis(),
    ),
    "mlp_classifier": ctx.MethodSpec(
        "mlp_classifier",
        "Multi-layer Perceptron Classifier",
        "1.17 Neural network models (supervised)",
        "classification",
        "breast_cancer",
        lambda seed: ctx.neural_network.MLPClassifier(
            hidden_layer_sizes=ctx.MLP_CLF_HIDDEN_LAYERS,
            max_iter=ctx.MLP_CLF_MAX_ITER,
            random_state=seed,
        ),
    ),
    "multinomial_nb": ctx.MethodSpec(
        "multinomial_nb",
        "Multinomial Naive Bayes",
        "1.9 Naive Bayes",
        "classification",
        "digits",
        lambda seed: ctx.naive_bayes.MultinomialNB(),
        needs_nonnegative=True,
    ),
    "qda_classifier": ctx.MethodSpec(
        "qda_classifier",
        "Quadratic Discriminant Analysis",
        "1.2 Linear and Quadratic Discriminant Analysis",
        "classification",
        "credit_risk",
        lambda seed: ctx.discriminant_analysis.QuadraticDiscriminantAnalysis(
            reg_param=ctx.QDA_REG_PARAM
        ),
    ),
    "self_training_classifier": ctx.MethodSpec(
        "self_training_classifier",
        "Self Training Classifier",
        "1.14 Semi-supervised learning",
        "classification",
        "moons",
        lambda seed: ctx.semi_supervised.SelfTrainingClassifier(
            ctx.svm.SVC(probability=True, gamma="scale")
        ),
    ),
}
