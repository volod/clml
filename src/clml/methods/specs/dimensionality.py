import clml.methods._context as ctx

METHODS = {
    "factor_analysis": ctx.MethodSpec(
        "factor_analysis",
        "Factor Analysis",
        "2.5 Matrix factorization",
        "dimensionality",
        "digits",
        lambda seed: ctx.decomposition.FactorAnalysis(
            n_components=ctx.DIM_DEFAULT_COMPONENTS, random_state=seed
        ),
        ctx._component_space,
    ),
    "fast_ica": ctx.MethodSpec(
        "fast_ica",
        "Independent Component Analysis",
        "2.5 Matrix factorization",
        "dimensionality",
        "digits",
        lambda seed: ctx.decomposition.FastICA(
            n_components=ctx.DIM_DEFAULT_COMPONENTS,
            random_state=seed,
            whiten="unit-variance",
        ),
        ctx._component_space,
    ),
    "isomap": ctx.MethodSpec(
        "isomap",
        "Isomap",
        "2.2 Manifold learning",
        "dimensionality",
        "digits",
        lambda seed: ctx.manifold.Isomap(
            n_components=ctx.DIM_DEFAULT_COMPONENTS, n_neighbors=ctx.ISOMAP_N_NEIGHBORS
        ),
    ),
    "kernel_pca": ctx.MethodSpec(
        "kernel_pca",
        "Kernel PCA",
        "2.5 Matrix factorization",
        "dimensionality",
        "digits",
        lambda seed: ctx.decomposition.KernelPCA(
            n_components=ctx.DIM_DEFAULT_COMPONENTS, kernel="rbf", random_state=seed
        ),
    ),
    "locally_linear_embedding": ctx.MethodSpec(
        "locally_linear_embedding",
        "Locally Linear Embedding",
        "2.2 Manifold learning",
        "dimensionality",
        "digits",
        lambda seed: ctx.manifold.LocallyLinearEmbedding(
            n_components=ctx.DIM_DEFAULT_COMPONENTS,
            n_neighbors=ctx.LLE_N_NEIGHBORS,
            random_state=seed,
        ),
        ctx._lle_space,
        notes=(
            "Manifold embedding that preserves local linear reconstruction weights, useful "
            "for learning curved low-dimensional structure from neighborhoods."
        ),
    ),
    "mds": ctx.MethodSpec(
        "mds",
        "Multidimensional Scaling",
        "2.2 Manifold learning",
        "dimensionality",
        "digits",
        lambda seed: ctx.manifold.MDS(
            n_components=ctx.DIM_DEFAULT_COMPONENTS,
            max_iter=ctx.MDS_MAX_ITER,
            random_state=seed,
            n_init=1,
        ),
        notes=(
            "Distance-preserving embedding that minimizes stress between original and "
            "projected pairwise distances."
        ),
    ),
    "nmf": ctx.MethodSpec(
        "nmf",
        "Non-negative Matrix Factorization",
        "2.5 Matrix factorization",
        "dimensionality",
        "digits",
        lambda seed: ctx.decomposition.NMF(
            n_components=ctx.DIM_DEFAULT_COMPONENTS,
            init="nndsvda",
            random_state=seed,
            max_iter=ctx.NMF_MAX_ITER,
        ),
        needs_nonnegative=True,
    ),
    "pca": ctx.MethodSpec(
        "pca",
        "Principal Component Analysis",
        "2.5 Matrix factorization",
        "dimensionality",
        "digits",
        lambda seed: ctx.decomposition.PCA(
            n_components=ctx.DIM_DEFAULT_COMPONENTS, random_state=seed
        ),
        ctx._component_space,
    ),
    "spectral_embedding": ctx.MethodSpec(
        "spectral_embedding",
        "Spectral Embedding",
        "2.2 Manifold learning",
        "dimensionality",
        "digits",
        lambda seed: ctx.manifold.SpectralEmbedding(
            n_components=ctx.DIM_DEFAULT_COMPONENTS, random_state=seed
        ),
    ),
    "tsne": ctx.MethodSpec(
        "tsne",
        "t-SNE",
        "2.2 Manifold learning",
        "dimensionality",
        "digits",
        lambda seed: ctx.manifold.TSNE(
            n_components=ctx.DIM_DEFAULT_COMPONENTS,
            perplexity=ctx.TSNE_PERPLEXITY,
            max_iter=ctx.TSNE_MAX_ITER,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ),
        ctx._tsne_space,
        notes=(
            "Nonlinear neighborhood embedding for visualization; emphasizes local neighbor "
            "structure rather than global distances."
        ),
    ),
    "umap": ctx.MethodSpec(
        "umap",
        "UMAP",
        "Extended manifold learning",
        "dimensionality",
        "digits",
        ctx._umap,
        ctx._umap_space,
        notes=(
            "Third-party nonlinear manifold embedding that is often faster than t-SNE and "
            "preserves more global neighborhood structure."
        ),
    ),
    "truncated_svd": ctx.MethodSpec(
        "truncated_svd",
        "Truncated SVD",
        "2.5 Matrix factorization",
        "dimensionality",
        "digits",
        lambda seed: ctx.decomposition.TruncatedSVD(
            n_components=ctx.DIM_DEFAULT_COMPONENTS, random_state=seed
        ),
        ctx._component_space,
    ),
}
