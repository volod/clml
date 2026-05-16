from clml.recommendations._common import _rec
from clml.recommendations._models import MethodRecommendation


def recommend_unsupervised() -> list[MethodRecommendation]:
    return [
        _rec(
            "pca",
            "No reliable target was found, so structure-first projection is appropriate.",
            "2D structure exploration.",
            "Embedding plot.",
            "Look for gradients/groups.",
            None,
            "high",
        ),
        _rec(
            "tsne",
            "No reliable target was found, so t-SNE is useful for neighborhood visualization.",
            "Visualize local sample neighborhoods.",
            "2D embedding plot.",
            "Use for exploratory visualization; distances between far groups are not reliable.",
            None,
            "medium",
        ),
        _rec(
            "umap",
            "UMAP is a fast modern default for high-dimensional visualization without labels.",
            "Explore local neighborhoods and broader manifold structure.",
            "2D embedding plot.",
            "Compare with PCA and t-SNE because manifold views can be parameter-sensitive.",
            None,
            "medium",
        ),
        _rec(
            "locally_linear_embedding",
            "LLE can reveal curved manifolds by preserving local geometry.",
            "Manifold visualization from local neighborhoods.",
            "2D embedding plot.",
            "Neighborhood structure is the main signal; tune neighbors for stability.",
            None,
            "medium",
        ),
        _rec(
            "mds",
            "MDS preserves pairwise distances and complements neighborhood-focused embeddings.",
            "Distance-preserving 2D projection.",
            "2D embedding plot.",
            "Stress-like visual distortions indicate distances are hard to preserve in 2D.",
            None,
            "medium",
        ),
        _rec(
            "kmeans",
            "No target was found, so clustering can provide exploratory segmentation.",
            "Segment rows into clusters.",
            "Cluster labels and silhouette.",
            "Interpret clusters by feature profiles.",
            None,
            "high",
        ),
        _rec(
            "isolation_forest",
            "No target was found, so anomaly detection can flag unusual rows for review.",
            "Detect unusual rows.",
            "Anomaly scores and plot.",
            "Review top anomalies manually.",
            None,
            "medium",
        ),
    ]
