import clml.methods._context as ctx

METHODS = {
    "affinity_propagation": ctx.MethodSpec(
        "affinity_propagation",
        "Affinity Propagation",
        "2.3 Clustering",
        "clustering",
        "blobs",
        lambda seed: ctx.cluster.AffinityPropagation(random_state=seed),
    ),
    "agglomerative_clustering": ctx.MethodSpec(
        "agglomerative_clustering",
        "Agglomerative Clustering",
        "2.3 Clustering",
        "clustering",
        "blobs",
        lambda seed: ctx.cluster.AgglomerativeClustering(n_clusters=ctx.CLUSTER_DEFAULT_K),
        ctx._cluster_space,
    ),
    "bayesian_gaussian_mixture": ctx.MethodSpec(
        "bayesian_gaussian_mixture",
        "Bayesian Gaussian Mixture",
        "2.1 Gaussian mixture models",
        "clustering",
        "blobs",
        lambda seed: ctx.mixture.BayesianGaussianMixture(
            n_components=ctx.CLUSTER_DEFAULT_K, random_state=seed
        ),
        ctx._component_space,
    ),
    "birch": ctx.MethodSpec(
        "birch",
        "BIRCH",
        "2.3 Clustering",
        "clustering",
        "blobs",
        lambda seed: ctx.cluster.Birch(n_clusters=ctx.CLUSTER_DEFAULT_K),
        ctx._cluster_space,
    ),
    "dbscan": ctx.MethodSpec(
        "dbscan",
        "DBSCAN",
        "2.3 Clustering",
        "clustering",
        "blobs",
        lambda seed: ctx.cluster.DBSCAN(eps=ctx.DBSCAN_EPS, min_samples=ctx.DBSCAN_MIN_SAMPLES),
    ),
    "elliptic_envelope": ctx.MethodSpec(
        "elliptic_envelope",
        "Robust Covariance / Elliptic Envelope",
        "2.6 Covariance estimation",
        "anomaly",
        "anomaly",
        lambda seed: ctx.covariance.EllipticEnvelope(
            random_state=seed, contamination=ctx.ANOMALY_CONTAMINATION
        ),
    ),
    "gaussian_mixture": ctx.MethodSpec(
        "gaussian_mixture",
        "Gaussian Mixture",
        "2.1 Gaussian mixture models",
        "clustering",
        "blobs",
        lambda seed: ctx.mixture.GaussianMixture(
            n_components=ctx.CLUSTER_DEFAULT_K, random_state=seed
        ),
        ctx._component_space,
    ),
    "isolation_forest": ctx.MethodSpec(
        "isolation_forest",
        "Isolation Forest",
        "2.7 Novelty and Outlier Detection",
        "anomaly",
        "anomaly",
        lambda seed: ctx.ensemble.IsolationForest(
            random_state=seed, contamination=ctx.ANOMALY_CONTAMINATION
        ),
    ),
    "kernel_density": ctx.MethodSpec(
        "kernel_density",
        "Kernel Density Estimation",
        "2.8 Density Estimation",
        "density",
        "blobs",
        lambda seed: ctx.neighbors.KernelDensity(
            kernel="gaussian", bandwidth=ctx.KDE_DEFAULT_BANDWIDTH
        ),
        lambda trial: {
            "model__bandwidth": trial.suggest_float(
                "bandwidth", ctx.KDE_BANDWIDTH_MIN, ctx.KDE_BANDWIDTH_MAX
            )
        },
    ),
    "kmeans": ctx.MethodSpec(
        "kmeans",
        "K-Means",
        "2.3 Clustering",
        "clustering",
        "blobs",
        lambda seed: ctx.cluster.KMeans(
            n_clusters=ctx.CLUSTER_DEFAULT_K, n_init="auto", random_state=seed
        ),
        ctx._cluster_space,
    ),
    "local_outlier_factor": ctx.MethodSpec(
        "local_outlier_factor",
        "Local Outlier Factor",
        "2.7 Novelty and Outlier Detection",
        "anomaly",
        "anomaly",
        lambda seed: ctx.neighbors.LocalOutlierFactor(
            novelty=True, contamination=ctx.ANOMALY_CONTAMINATION
        ),
    ),
    "mean_shift": ctx.MethodSpec(
        "mean_shift",
        "Mean Shift",
        "2.3 Clustering",
        "clustering",
        "blobs",
        lambda seed: ctx.cluster.MeanShift(),
    ),
    "minibatch_kmeans": ctx.MethodSpec(
        "minibatch_kmeans",
        "Mini-Batch K-Means",
        "2.3 Clustering",
        "clustering",
        "blobs",
        lambda seed: ctx.cluster.MiniBatchKMeans(
            n_clusters=ctx.CLUSTER_DEFAULT_K, random_state=seed, n_init="auto"
        ),
        ctx._cluster_space,
    ),
    "one_class_svm": ctx.MethodSpec(
        "one_class_svm",
        "One-Class SVM",
        "2.7 Novelty and Outlier Detection",
        "anomaly",
        "anomaly",
        lambda seed: ctx.svm.OneClassSVM(nu=ctx.ONE_CLASS_SVM_NU, kernel="rbf", gamma="scale"),
    ),
    "optics": ctx.MethodSpec(
        "optics",
        "OPTICS",
        "2.3 Clustering",
        "clustering",
        "blobs",
        lambda seed: ctx.cluster.OPTICS(min_samples=ctx.OPTICS_MIN_SAMPLES),
    ),
    "spectral_clustering": ctx.MethodSpec(
        "spectral_clustering",
        "Spectral Clustering",
        "2.3 Clustering",
        "clustering",
        "blobs",
        lambda seed: ctx.cluster.SpectralClustering(
            n_clusters=ctx.CLUSTER_DEFAULT_K, random_state=seed, assign_labels="kmeans"
        ),
        ctx._cluster_space,
    ),
    "bisecting_kmeans": ctx.MethodSpec(
        "bisecting_kmeans",
        "Bisecting K-Means",
        "2.3 Clustering",
        "clustering",
        "blobs",
        lambda seed: ctx.cluster.BisectingKMeans(
            n_clusters=ctx.CLUSTER_DEFAULT_K, random_state=seed
        ),
        ctx._cluster_space,
        notes=(
            "Divisive hierarchical clustering: each step bisects the largest cluster using "
            "k-means. Produces more balanced clusters than standard k-means and avoids "
            "the empty-cluster issue common in random initialization."
        ),
    ),
    "hdbscan": ctx.MethodSpec(
        "hdbscan",
        "HDBSCAN",
        "2.3 Clustering",
        "clustering",
        "blobs",
        lambda seed: ctx.cluster.HDBSCAN(min_cluster_size=ctx.HDBSCAN_MIN_CLUSTER_SIZE),
        notes=(
            "Hierarchical DBSCAN that selects the most stable clusters across density levels. "
            "Automatically determines cluster count; noise points are labeled -1. More robust "
            "than DBSCAN to varying density and parameter sensitivity."
        ),
    ),
}
